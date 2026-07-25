"""
dependency_refinement.py
========================
Progressively refine the schema-matching similarity matrix using data
dependencies (FD / IND / CFD) from the source side.
Method name: Progressive Dependency Mapping-Validation Refinement (PDMVR)

────────────────────────────────────────────────────────────────────
Attribute Index Convention
────────────────────────────────────────────────────────────────────
Source attribute indices: integers 0 ~ n_source-1, written as strings
    "0", "1", ... in dependency files.
Target attribute indices: integers n_source ~ n_source+n_target-1;
    target instance column k corresponds to attribute index n_source+k.

Similarity matrix similarity_matrix[i][j]:
    Row i -> target attribute index n_source+i
    Column j -> source attribute index j

────────────────────────────────────────────────────────────────────
Iteration Mechanism (PDMVR)
────────────────────────────────────────────────────────────────────
Within each iteration:
  1. Compute the "sharpness" of the current matrix: the mean gap between
     the top-1 and top-2 scores per column.
  2. Dynamically determine top_k for this round based on sharpness and
     the current iteration number.
  3. Process dependencies in order (FD -> IND); after validating each
     dependency, update the matrix immediately (online update).
  4. The reward for each dependency is adaptively determined by two
     convergence-signal factors (no fixed base_reward).
  5. After a round ends, check the multi-signal convergence condition
     (matrix change + top-1 stability + patience).
  6. When oscillation is detected, take the element-wise mean of the
     most recent rounds and exit.

top_k adjustment rules:
    k_by_iter      = k_max - t * (k_max - k_min) / max_iter   (linear decay over rounds)
    k_by_sharpness = k_max - sharpness * (k_max - k_min)       (shrink when sharpness is high)
    top_k          = max(k_min, int(min(k_by_iter, k_by_sharpness)))

reward adjustment rules (two-phase):
    Round 0: reward = scale(S) * decay_factor^t
        scale(S) is automatically estimated from the matrix standard deviation;
        no manual setting required.
    Round 1+: reward = scale(validated) * stability * exploration
        scale(validated) is inversely proportional to the number of validated
        assignments from the previous round; automatically amplified when
        dependencies are sparse.
        stability   = exp(-k_s * delta)              (suppress reward when matrix is unstable)
        exploration = 1 - exp(-k_e * top1_ratio)     (naturally decays to zero as matching stabilizes)
    Reward cap: cumulative score at a single matrix cell <= score_ceiling

────────────────────────────────────────────────────────────────────
Dependency File Formats (attribute names are source attribute index strings)
────────────────────────────────────────────────────────────────────
FD : 0,1 -> 2
IND: 0[=1   or   0,1[=2,3
CFD: (0, 1=1996) => 2
     Items with = inside parentheses are pattern constants
     (attribute_index=constant_value); items without = are wildcard LHS.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from util import parse_fds, parse_inds, parse_cfds, build_target_relation

# ── Import dependency validators ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/mnt/project")

from dependency_validator import (
    FD, CFD, IND,
    Relation,
    validate_fd, validate_cfd, validate_ind,
)

# Assignment type: source attribute index (int) -> target attribute index (int)
Assignment = Dict[int, int]


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _top_n_targets(
    sim_matrix: List[List[float]],
    src_idx: int,
    n_source: int,
    top_n: int,
) -> List[int]:
    """
    For source attribute src_idx, return the global indices of the top_n
    target attributes with the highest similarity scores.

    Target attribute global index = matrix row index + n_source, sorted
    in descending order of similarity.
    """
    # Iterate over each row, extract the score for this source column,
    # and record the corresponding target global index
    scored: List[Tuple[float, int]] = [
        (row[src_idx] if src_idx < len(row) else 0.0, row_i + n_source)
        for row_i, row in enumerate(sim_matrix)
    ]
    scored.sort(reverse=True)
    return [tgt for _, tgt in scored[:top_n]]


def _remap_attrs(attrs: List[str], assignment: Assignment) -> List[str]:
    """
    Replace source attribute index strings in the given list with the
    corresponding target attribute index strings according to the assignment.

    assignment: source attribute index (int) -> target attribute index (int)
    """
    return [str(assignment[int(a)]) for a in attrs]


def _apply_reward(
    updated: List[List[float]],
    assignment: Assignment,
    n_source: int,
    reward: float,
    original: List[List[float]],
    score_ceiling: float,
) -> None:
    """
    For each (source column index, target row index) pair in the assignment,
    add the reward value to the corresponding cell in the matrix.

    Matrix row index = target attribute global index - n_source
    Matrix column index = source attribute index

    score_ceiling limits the cumulative reward per cell:
        actual_reward = min(reward, ceiling - current_value)
    This prevents certain matches from being repeatedly rewarded across
    multiple iterations, which would cause scores to grow unboundedly.
    original is fixed as the matrix snapshot at the start of the current
    round, used to compute the remaining rewardable headroom, ensuring
    that cumulative rewards from multiple dependencies on the same cell
    within a single round remain bounded by the ceiling.
    """
    n_target = len(updated)
    for src_idx, tgt_idx in assignment.items():
        row = tgt_idx - n_source    # matrix row
        col = src_idx               # matrix column
        if 0 <= row < n_target and 0 <= col < len(updated[row]):
            # Remaining rewardable headroom = ceiling - current value; ensure non-negative
            headroom = score_ceiling - updated[row][col]
            actual_reward = min(reward, max(headroom, 0.0))
            updated[row][col] += actual_reward


def _enumerate_assignments(
    src_indices: List[int],
    sim_matrix: List[List[float]],
    n_source: int,
    top_n: int,
) -> List[Assignment]:
    """
    For the given list of source attribute indices, enumerate the Cartesian
    product of all top-n candidate target attributes and return a list of
    all possible assignments (source index -> target global index).

    With only 5 target attributes, the Cartesian product for top_n=3 has
    size 3^|src_indices|, which remains tractable.
    """
    # Top-n candidate target list for each source attribute
    candidates = [
        _top_n_targets(sim_matrix, src_idx, n_source, top_n)
        for src_idx in src_indices
    ]
    # Cartesian product enumeration; zip aligns each combination with source indices into a dict
    return [
        dict(zip(src_indices, combo))
        for combo in itertools.product(*candidates)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Dependency Validation Functions
# Signatures are extended with two new parameters compared to the original:
# original (round snapshot) and score_ceiling (reward cap).
# Both are passed through to _apply_reward; the validation logic itself
# is identical to the original. The return value is the number of
# validated assignments in this call, used for iteration-level statistics.
# ══════════════════════════════════════════════════════════════════════════════

def _validate_fd(
    fd: FD,
    sim_matrix: List[List[float]],
    updated: List[List[float]],
    target_relation: Relation,
    n_source: int,
    top_n: int,
    reward: float,
    original: List[List[float]],
    score_ceiling: float,
) -> int:
    """
    For a single FD, enumerate all candidate assignments involving the
    source attributes, replace source attribute indices in the FD with
    the corresponding target attribute indices, and validate. Apply
    reward if the FD holds. Return the number of validated assignments.
    """
    # All source attribute indices involved in the FD (deduplicated, order preserved)
    involved = list(dict.fromkeys(int(a) for a in fd.lhs + fd.rhs))
    validated = 0
    for assignment in _enumerate_assignments(involved, sim_matrix, n_source, top_n):
        # Replace FD attribute names from source indices to target indices
        mapped_lhs = _remap_attrs(fd.lhs, assignment)
        mapped_rhs = _remap_attrs(fd.rhs, assignment)
        # Duplicate column filter: if A=B after mapping in A,B -> C, the assignment is invalid; skip
        if len(set(mapped_lhs)) < len(mapped_lhs):
            continue
        # Triviality filter: if any RHS attribute appears in LHS, the dependency always holds; skip
        if set(mapped_rhs) & set(mapped_lhs):
            continue
        mapped_fd = FD(lhs=mapped_lhs, rhs=mapped_rhs)
        result = validate_fd(target_relation, mapped_fd)
        if result.satisfied:
            _apply_reward(updated, assignment, n_source, reward, original, score_ceiling)
            validated += 1
    return validated


def _validate_ind(
    ind: IND,
    sim_matrix: List[List[float]],
    updated: List[List[float]],
    target_relation: Relation,
    n_source: int,
    top_n: int,
    reward: float,
    original: List[List[float]],
    score_ceiling: float,
) -> int:
    """
    For a single IND, enumerate candidate assignments for LHS/RHS source
    attributes separately and validate. Return the number of validated
    assignment combinations.

    IND semantics: lhs_relation[lhs_attrs] is a subset of rhs_relation[rhs_attrs].
    Here lhs_relation and rhs_relation are both the same target_relation,
    while lhs/rhs are mapped to different target attribute columns (which
    may or may not be the same).
    """
    lhs_src = [int(a) for a in ind.lhs_attrs]
    rhs_src = [int(a) for a in ind.rhs_attrs]
    validated = 0
    for lhs_assign in _enumerate_assignments(lhs_src, sim_matrix, n_source, top_n):
        for rhs_assign in _enumerate_assignments(rhs_src, sim_matrix, n_source, top_n):
            mapped_lhs_attrs = _remap_attrs(ind.lhs_attrs, lhs_assign)
            mapped_rhs_attrs = _remap_attrs(ind.rhs_attrs, rhs_assign)
            # Triviality filter: if LHS and RHS attribute columns overlap, the dependency has no constraining power; skip
            if set(mapped_lhs_attrs) & set(mapped_rhs_attrs):
                continue
            mapped_ind = IND(
                lhs_attrs=mapped_lhs_attrs,
                rhs_attrs=mapped_rhs_attrs,
            )
            result = validate_ind(mapped_ind,
                                  lhs_relation=target_relation,
                                  rhs_relation=target_relation)
            if result.satisfied:
                # Attributes on both sides of the IND participated in this validation; reward both assignments
                _apply_reward(updated, lhs_assign, n_source, reward, original, score_ceiling)
                _apply_reward(updated, rhs_assign, n_source, reward, original, score_ceiling)
                validated += 1
    return validated


def _validate_cfd(
    cfd: CFD,
    sim_matrix: List[List[float]],
    updated: List[List[float]],
    target_relation: Relation,
    n_source: int,
    top_n: int,
    reward: float,
    original: List[List[float]],
    score_ceiling: float,
) -> int:
    """
    For a single CFD, enumerate all candidate assignments, replace source
    attribute indices in the CFD (including pattern keys) with target
    attribute indices, and validate. Apply reward only when substantively
    satisfied (i.e., at least one tuple matches the pattern). Return the
    number of validated assignments.

    Note: In practice CFD validation is disabled by default (cfd_file=None);
    this function is retained for future extension.
    """
    involved = list(dict.fromkeys(int(a) for a in cfd.lhs + cfd.rhs))
    validated = 0
    for assignment in _enumerate_assignments(involved, sim_matrix, n_source, top_n):
        mapped_lhs = _remap_attrs(cfd.lhs, assignment)
        mapped_rhs = _remap_attrs(cfd.rhs, assignment)
        # Duplicate column filter: same as FD
        if len(set(mapped_lhs)) < len(mapped_lhs):
            continue
        # Triviality filter: inherits FD triviality; skip if RHS intersect LHS is non-empty
        if set(mapped_rhs) & set(mapped_lhs):
            continue
        # Pattern keys also need to be remapped from source indices to target indices
        mapped_pattern = {
            str(assignment[int(k)]): v
            for k, v in cfd.pattern.items()
        }
        mapped_cfd = CFD(
            lhs=mapped_lhs,
            rhs=mapped_rhs,
            pattern=mapped_pattern,
        )
        result = validate_cfd(target_relation, mapped_cfd)
        # Exclude vacuous satisfaction (no tuple matches the pattern; dependency holds trivially; do not reward)
        vacuous = any("vacuously" in v for v in result.violations)
        if result.satisfied and not vacuous:
            _apply_reward(updated, assignment, n_source, reward, original, score_ceiling)
            validated += 1
    return validated


# ══════════════════════════════════════════════════════════════════════════════
# Iteration Control Utility Functions
# ══════════════════════════════════════════════════════════════════════════════

def _compute_sharpness(sim_matrix: List[List[float]]) -> float:
    """
    Compute the matrix "sharpness": the mean gap between the top-1 and
    top-2 scores per column, then compress to [0, 1] via tanh.

    High sharpness  -> the best target for each source attribute is fairly
                       clear -> shrink top_k and enter the refinement phase.
    Low sharpness   -> matching is still uncertain
                       -> keep a larger top_k and continue broad exploration.

    Uses tanh(x * 3) for normalization so that common similarity gap
    ranges map roughly uniformly to [0, 1]:
        raw_sharpness=0.1 -> ~0.29, raw_sharpness=0.3 -> ~0.72.
    """
    gaps = []
    for row in sim_matrix:
        if len(row) < 2:
            continue
        sorted_row = sorted(row, reverse=True)
        gaps.append(sorted_row[0] - sorted_row[1])
    if not gaps:
        return 0.0
    raw_sharpness = sum(gaps) / len(gaps)
    # tanh normalization: raw_sharpness=0.1 -> ~0.29, raw_sharpness=0.3 -> ~0.72
    return math.tanh(raw_sharpness * 3)


def _compute_dynamic_k(
    t: int,
    sharpness: float,
    k_min: int,
    k_max: int,
    max_iter: int,
) -> int:
    """
    Dual-factor dynamic top_k: take the minimum of two factors
    (conservative refinement strategy).

    Factor 1 (iteration): k_by_iter = k_max - t * (k_max - k_min) / max_iter
        At t=0, k=k_max (maximum exploration); at t=max_iter, k=k_min
        (minimum refinement).

    Factor 2 (sharpness): k_by_sharpness = k_max - sharpness * (k_max - k_min)
        At sharpness=0, k=k_max; at sharpness=1, k=k_min.
        When the matrix is already highly differentiated, shrink the
        candidate set early even at earlier iterations.

    Taking min ensures that as soon as either factor indicates shrinking,
    we shrink immediately, avoiding low-quality candidates in the Cartesian
    product at later stages.
    """
    # Iteration factor: t=0 -> k_max, t=max_iter -> k_min
    k_by_iter = k_max - t * (k_max - k_min) / max(max_iter, 1)
    # Sharpness factor: sharpness in [0,1]; higher sharpness yields smaller k
    k_by_sharpness = k_max - sharpness * (k_max - k_min)
    k = min(k_by_iter, k_by_sharpness)
    return max(k_min, int(round(k)))


def _estimate_reward_scale(
    sim_matrix: Optional[List[List[float]]] = None,
    prev_validated: int = 0,
    target_delta: float = 0.015,
    reward_min: float = 0.001,
    reward_max: float = 0.05,
    matrix_scale: float = 0.07,
) -> float:
    """
    Automatically estimate the reward scale for the current round;
    no manual base_reward setting required.

    Two estimation modes:

    Round 0 (prev_validated=0, sim_matrix is valid):
        Estimate an appropriate scale from the global standard deviation
        of the matrix.
        Large std -> scores are already dispersed; top-1 is clearly ahead
                     of other candidates; a small reward suffices.
        Small std -> scores are concentrated; a larger reward is needed
                     to drive ranking changes.
        Formula: scale = target_delta * matrix_scale / std

    Round 1+ (prev_validated > 0):
        Determined inversely by the number of validated assignments from
        the previous round.
        More validated (rich dependencies) -> smaller per-assignment reward
            to prevent over-modification.
        Fewer validated (sparse dependencies) -> larger per-assignment reward
            to ensure the refinement signal is strong enough.
        Formula: scale = target_delta / validated

    Both modes are clamped to [reward_min, reward_max].

    Parameters
    ----
    sim_matrix      : Current similarity matrix; used for Round 0 estimation
    prev_validated  : Number of validated assignments from the previous round
    target_delta    : Desired relative matrix change per round; controls the
                      overall reward strength baseline
    reward_min      : Lower bound on reward scale; prevents reward from
                      approaching zero when dependencies are extremely numerous
    reward_max      : Upper bound on reward scale; prevents excessive reward
                      when dependencies are very few
    matrix_scale    : Tuning coefficient for Round 0 estimation; default 0.07
    """
    if prev_validated <= 0:
        # Round 0: auto-estimate from matrix standard deviation
        if sim_matrix:
            all_values = [v for row in sim_matrix for v in row]
            mean = sum(all_values) / len(all_values)
            variance = sum((v - mean) ** 2 for v in all_values) / len(all_values)
            std = math.sqrt(variance) if variance > 0 else 1.0
            # Larger std means scores are already dispersed -> reward should be smaller;
            # smaller std means scores are concentrated -> reward should be larger
            scale = (target_delta * matrix_scale) / std
        else:
            scale = reward_max
    else:
        # Round 1+: inversely proportional to validated count; auto-amplify when dependencies are sparse
        scale = target_delta / max(prev_validated, 1)

    return float(max(reward_min, min(reward_max, scale)))


def _compute_reward(
    sim_matrix: Optional[List[List[float]]],
    prev_validated: int,
    t: int,
    decay_factor: float,
    prev_delta: float = 0.0,
    prev_top1_change_ratio: float = 1.0,
    k_stability: float = 10.0,
    k_exploration: float = 3.0,
    target_delta: float = 0.015,
    reward_min: float = 0.001,
    reward_max: float = 0.05,
    matrix_scale: float = 0.07,
) -> float:
    """
    Two-phase adaptive reward, addressing two issues: (1) overly rapid
    decay from triple-factor multiplicative stacking, and (2) insufficient
    signal when dependencies are sparse.

    Round 0 (no historical signal):
        reward = scale(S) * decay_factor^t
        scale is auto-estimated from the matrix standard deviation; decay
        provides initial-round decay protection.
        At this point prev_delta=0 and prev_top1_change_ratio=1, so
        convergence signals are meaningless; the formula degrades to
        pure scale decay, consistent with the original behavior.

    Round 1+ (historical signal available):
        reward = scale(validated) * stability * exploration
        Convergence-signal factors fully replace iteration-based decay;
        no additional stacking:
        - stability   = exp(-k_s * delta): suppress reward when the matrix
          fluctuates significantly
        - exploration = 1 - exp(-k_e * ratio): naturally decay to zero
          as matching stabilizes
        The late-stage suppression effect of iteration decay is naturally
        handled by exploration; no separate multiplicative factor needed.

    Parameters
    ----
    sim_matrix             : Current matrix; used for Round 0 scale estimation
    prev_validated         : Number of validated assignments from the previous
                             round; drives adaptive scale
    t                      : Current iteration round
    decay_factor           : Iteration decay coefficient for Round 0 only;
                             affects only the initial round
    prev_delta             : Relative matrix change from the previous round
    prev_top1_change_ratio : Proportion of source attributes whose top-1
                             changed in the previous round
    k_stability            : Stability factor sensitivity; recommended 10.0
    k_exploration          : Exploration factor sensitivity; recommended 3.0
    target_delta / reward_min / reward_max / matrix_scale :
                             Passed through to _estimate_reward_scale
    """
    # Auto-estimate reward scale for this round (replaces fixed base_reward)
    scale = _estimate_reward_scale(
        sim_matrix=sim_matrix,
        prev_validated=prev_validated,
        target_delta=target_delta,
        reward_min=reward_min,
        reward_max=reward_max,
        matrix_scale=matrix_scale,
    )

    # Round 0: no historical signal; degrade to pure scale decay
    if prev_validated == 0:
        return scale * (decay_factor ** t)

    # Round 1+: convergence signals replace iteration decay, avoiding the
    # multiplicative amplification effect of three factors decaying in sync
    # stability: suppress reward when matrix change is large, preventing
    # overly aggressive updates during the unstable phase
    stability   = math.exp(-k_stability * prev_delta)
    # exploration: maintain reward while top-1 is still changing frequently;
    # naturally decay to zero as it stabilizes
    exploration = 1.0 - math.exp(-k_exploration * prev_top1_change_ratio)

    return scale * stability * exploration


def _frobenius_relative_delta(mat_new: List[List[float]], mat_old: List[List[float]]) -> float:
    """
    Compute the relative Frobenius distance between two matrices for
    convergence detection.

    relative_delta = ||S_new - S_old||_F / ||S_old||_F

    Uses relative distance rather than absolute distance to prevent the
    overall magnitude of the matrix from affecting the effectiveness of
    the threshold. A small epsilon (1e-12) is added to the denominator
    to avoid division by zero for zero matrices.
    """
    sq_diff = sum(
        (a - b) ** 2
        for row_new, row_old in zip(mat_new, mat_old)
        for a, b in zip(row_new, row_old)
    )
    sq_old = sum(v ** 2 for row in mat_old for v in row)
    return math.sqrt(sq_diff) / (math.sqrt(sq_old) + 1e-12)


def _count_top1_changes(mat_new: List[List[float]], mat_old: List[List[float]]) -> int:
    """
    Count the number of columns whose top-1 row has changed between the
    two matrices. That is, how many source attributes had their "best
    matching target" change in this round.

    This is a more direct matching-stability metric than Frobenius distance:
    even if the matrix values still exhibit minor fluctuations, as long as
    the final matching results no longer change, convergence can be declared.
    Both conditions (delta < eps AND top1_changes == 0) must be satisfied
    simultaneously to trigger the convergence signal.
    """
    n_col = len(mat_new[0]) if mat_new else 0
    changes = 0
    for col in range(n_col):
        top1_new = max(range(len(mat_new)), key=lambda r: mat_new[r][col])
        top1_old = max(range(len(mat_old)), key=lambda r: mat_old[r][col])
        if top1_new != top1_old:
            changes += 1
    return changes


def _detect_oscillation(matrix_history: List[List[List[float]]], window: int = 4) -> bool:
    """
    Detect iteration oscillation: the matrix bounces back and forth
    between two states and cannot converge naturally.

    Detection method: take the sharpness values of the most recent
    ``window`` rounds, split the sequence into odd-indexed and
    even-indexed groups. If the odd rounds are internally stable and
    the even rounds are internally stable, but the means of the two
    groups differ significantly, oscillation is declared.

    Post-oscillation handling (see _refinement_iterative): take the
    element-wise mean of the most recent ``window`` matrices and exit,
    effectively compromising between the two oscillating states and
    eliminating the jitter.
    """
    if len(matrix_history) < window:
        return False
    recent = matrix_history[-window:]
    sharpnesses = [_compute_sharpness(m) for m in recent]
    # Odd-even split: recent[0,2,...] is the odd group, recent[1,3,...] is the even group
    odd  = sharpnesses[0::2]
    even = sharpnesses[1::2]

    def _std(lst):
        if len(lst) < 2:
            return 0.0
        mean = sum(lst) / len(lst)
        return math.sqrt(sum((x - mean) ** 2 for x in lst) / len(lst))

    odd_stable  = _std(odd)  < 0.005   # Sharpness is stable within odd-indexed rounds
    even_stable = _std(even) < 0.005   # Sharpness is stable within even-indexed rounds
    cross_diff  = abs(sum(odd) / len(odd) - sum(even) / len(even)) > 0.01  # Difference between odd and even groups
    return odd_stable and even_stable and cross_diff


def _average_matrices(matrices: List[List[List[float]]]) -> List[List[float]]:
    """
    Compute the element-wise arithmetic mean of multiple matrices of the
    same shape. Used after oscillation detection to produce a compromise
    value from recent rounds as the final output.
    """
    n_row = len(matrices[0])
    n_col = len(matrices[0][0])
    avg = [[0.0] * n_col for _ in range(n_row)]
    for mat in matrices:
        for i in range(n_row):
            for j in range(n_col):
                avg[i][j] += mat[i][j]
    k = len(matrices)
    return [[v / k for v in row] for row in avg]


# ══════════════════════════════════════════════════════════════════════════════
# Core Iteration Function
# ══════════════════════════════════════════════════════════════════════════════

def _refinement_iterative(
    similarity_matrix: List[List[float]],
    target_instances: List[List[Any]],
    fd_file: Optional[str] = None,
    ind_file: Optional[str] = None,
    cfd_file: Optional[str] = None,
    # ── Iteration parameters ──
    max_iter: int = 5,
    patience: int = 2,
    matrix_eps: float = 1e-4,
    # ── top_k parameters ──
    k_min: int = 1,
    k_max: int = 3,
    # ── Reward parameters (no fixed base_reward; reward scale is auto-estimated from matrix and validation count) ──
    decay_factor: float = 0.6,
    score_ceiling: float = 1.5,
    k_stability: float = 10.0,
    k_exploration: float = 3.0,
    target_delta: float = 0.015,
    reward_min: float = 0.001,
    reward_max: float = 0.05,
    matrix_scale: float = 0.07,
    # ── Ablation control ──
    dep_mode: str = "all",   # 'all'=FD+IND (V0), 'fd'=FD only (V3), 'ind'=IND only (V4)
    # ── Logging ──
    verbose: bool = False,
) -> List[List[float]]:
    """
    PDMVR core iteration function, combining batch iteration (round
    structure) with online updates (immediate updates within a round).

    Parameters
    ----
    max_iter      : Maximum iteration rounds (hard upper limit); prevents
                    infinite loops in extreme cases
    patience      : Number of consecutive rounds that must satisfy the
                    convergence condition before truly stopping; prevents
                    false convergence during oscillation
    matrix_eps    : Relative matrix change threshold (Frobenius); below
                    this value the matrix is considered stable
    k_min / k_max : Dynamic range for top_k; with only 5 target attributes,
                    k_min=1 and k_max=3 are recommended
    decay_factor  : Iteration decay coefficient for Round 0; affects only
                    the initial round; recommended 0.4~0.6
    score_ceiling : Per-cell score cap; recommended 1.5 when embedding
                    similarities are within ~1.0
    k_stability   : Stability factor sensitivity; controls the suppression
                    strength of delta on reward; recommended 10.0
    k_exploration : Exploration factor sensitivity; controls the boost
                    effect of top-1 change rate on reward; recommended 3.0
    target_delta  : Desired relative matrix change per round; controls the
                    reward strength baseline; recommended 0.015
    reward_min    : Lower bound on reward scale; prevents reward from
                    approaching zero when dependencies are extremely numerous
    reward_max    : Upper bound on reward scale; prevents excessive reward
                    when dependencies are very few; also serves as the
                    Round 0 estimation upper bound
    matrix_scale  : Tuning coefficient for Round 0 matrix std estimation;
                    recommended 0.07
    dep_mode      : Ablation control. 'all'=validate both FD and IND
                    (V0/V2 full method); 'fd'=validate FD only (V3);
                    'ind'=validate IND only (V4).
                    For datasets without INDs (e.g., MusicRecordings),
                    'ind' mode validates no dependencies and the output
                    is element-wise identical to the unrefined input
                    matrix (V1).
    verbose       : Whether to print sharpness, top_k, reward value, and
                    convergence signals for each round
    """
    n_source = len(similarity_matrix[0]) if similarity_matrix else 0
    # Build target Relation only once; keys are target attribute global index strings; reused throughout
    target_relation = build_target_relation(target_instances, n_source)

    # Decide which dependency types to load based on dep_mode (ablation V3/V4 filter here only; validation loop below needs no changes)
    use_fd  = dep_mode in ("all", "fd")
    use_ind = dep_mode in ("all", "ind")
    fds:  List[FD]  = parse_fds(fd_file)   if (use_fd  and fd_file  and os.path.exists(fd_file))  else []
    inds: List[IND] = parse_inds(ind_file) if (use_ind and ind_file and os.path.exists(ind_file)) else []
    # cfds: List[CFD] = parse_cfds(cfd_file) if os.path.exists(cfd_file) else []

    # Current working matrix (deep copy; do not modify original input)
    current = copy.deepcopy(similarity_matrix)

    # matrix_history stores the snapshot before each round starts, used for convergence detection and oscillation detection
    matrix_history: List[List[List[float]]] = []
    patience_counter = 0   # Counter for consecutive rounds satisfying the convergence condition

    # Previous-round state signals, used for reward computation in this round
    # Round 0 has no historical signal: prev_validated=0 triggers the matrix std estimation mode;
    # prev_delta=0 and prev_top1_change_ratio=1 prevent convergence factors from suppressing the initial reward
    prev_delta             = 0.0
    prev_top1_change_ratio = 1.0
    prev_validated         = 0    # Number of validated assignments from the previous round; drives adaptive reward scale

    for t in range(max_iter):
        # Save the matrix snapshot before this round starts, used for delta computation and oscillation detection after this round ends
        prev = copy.deepcopy(current)
        matrix_history.append(prev)

        # ── 1. Compute dynamic parameters for this round ──────────────────────
        sharpness = _compute_sharpness(current)
        top_k     = _compute_dynamic_k(t, sharpness, k_min, k_max, max_iter)

        # Two-phase adaptive reward (FD / IND / CFD share the same reward scale):
        #   Round 0: scale(S) * decay^t, auto-estimated from matrix std
        #   Round 1+: scale(validated) * stability * exploration
        #              convergence signals replace iteration decay, avoiding
        #              the multiplicative amplification of three factors decaying in sync
        reward_kwargs = dict(
            sim_matrix=current,
            prev_validated=prev_validated,
            t=t,
            decay_factor=decay_factor,
            prev_delta=prev_delta,
            prev_top1_change_ratio=prev_top1_change_ratio,
            k_stability=k_stability,
            k_exploration=k_exploration,
            target_delta=target_delta,
            reward_min=reward_min,
            reward_max=reward_max,
            matrix_scale=matrix_scale,
        )
        depy_reward  = _compute_reward(**reward_kwargs) # fd and ind share the same reward score
        

        if verbose:
            scale = _estimate_reward_scale(
                sim_matrix=current,
                prev_validated=prev_validated,
                target_delta=target_delta,
                reward_min=reward_min,
                reward_max=reward_max,
                matrix_scale=matrix_scale,
            )
            print(f"\n[Iter {t}] sharpness={sharpness:.4f}  top_k={top_k}"
                  f"  reward_scale={scale:.5f}"
                  f"  dependenct_reward={depy_reward:.5f}"
                  f"  (prev_delta={prev_delta:.4f}"
                  f"  prev_top1_ratio={prev_top1_change_ratio:.2f}"
                  f"  prev_validated={prev_validated})")

        # ── 2. Online update: validate each dependency and update current immediately ──
        #
        # [Core of online update]: both sim_matrix and updated point to current,
        # meaning each dependency reads the latest matrix when determining its
        # candidate set, and writes back to the same object immediately upon
        # validation. Thus the candidate set for dependency i+1 already
        # reflects the reward from dependency i, in contrast to the original
        # approach of "using a fixed similarity_matrix for all candidates".
        #
        # [Role of original]: prev is passed as the round-start snapshot to
        # _apply_reward for computing headroom (ceiling - current_value),
        # ensuring that cumulative rewards on the same cell within a single
        # round do not exceed score_ceiling.
        #
        # [Dependency processing order]: FD -> IND, semantically from strong
        # constraints to weak constraints, so that strong-constraint reward
        # signals influence the candidate set first, then weak constraints
        # further refine.
        #
        total_validated = 0

        for fd in fds:
            v = _validate_fd(
                fd, current, current,   # both sim_matrix and updated point to current (online update)
                target_relation, n_source, top_k, depy_reward,
                original=prev, score_ceiling=score_ceiling,
            )
            total_validated += v

        for ind in inds:
            v = _validate_ind(
                ind, current, current,
                target_relation, n_source, top_k, depy_reward,
                original=prev, score_ceiling=score_ceiling,
            )
            total_validated += v

        # CFD is disabled by default (cfd_file=None); structure retained for future extension
        # for cfd in cfds:
        #     v = _validate_cfd(
        #         cfd, current, current,
        #         target_relation, n_source, top_k, depy_reward,
        #         original=prev, score_ceiling=score_ceiling,
        #     )
        #     total_validated += v

        # ── 3. Convergence detection ──────────────────────────────────────────
        delta    = _frobenius_relative_delta(current, prev)
        top1_chg = _count_top1_changes(current, prev)

        # Update state signals for next round's _compute_reward
        prev_delta             = delta
        prev_top1_change_ratio = top1_chg / max(n_source, 1)
        prev_validated         = total_validated  # Drives next round's adaptive reward scale

        if verbose:
            print(f"         validated={total_validated}  "
                  f"matrix_delta={delta:.6f}  top1_changes={top1_chg}")

        # Oscillation detection takes priority over convergence check: if oscillation is detected, take the mean of the last 4 rounds and force exit
        if _detect_oscillation(matrix_history, window=4):
            if verbose:
                print(f"[Iter {t}] Oscillation detected; taking mean of last 4 rounds and exiting")
            current = _average_matrices(matrix_history[-4:])
            break

        # Multi-signal convergence: trigger convergence signal only when matrix change is negligible AND best matches no longer change
        converged = (delta < matrix_eps) and (top1_chg == 0)
        if converged:
            patience_counter += 1
            if verbose:
                print(f"         Convergence signal ({patience_counter}/{patience})")
            if patience_counter >= patience:
                if verbose:
                    print(f"[Iter {t}] Reached patience={patience}; early convergence exit")
                break
        else:
            # Reset counter if convergence condition is not met, preventing intermittent satisfaction from being mistaken for convergence
            patience_counter = 0

    return current


# ══════════════════════════════════════════════════════════════════════════════
# Original Single-Round _refinement (retained for comparison and fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _refinement(
    similarity_matrix: List[List[float]],
    target_instances: List[List[Any]],
    fd_file: Optional[str] = None,
    ind_file: Optional[str] = None,
    cfd_file: Optional[str] = None,
    top_n: int = 3,
    rewards: Optional[Dict[str, float]] = None,
) -> List[List[float]]:
    """
    Original single-round reward; retained for comparison experiments.
    Behavior is fully consistent with v1.

    Note: score_ceiling is set to float("inf"), i.e., no reward cap,
    which matches the original behavior.
    """
    if rewards is None:
        rewards = {"fd": 0.01, "ind": 0.01, "cfd": 0.01}

    n_source = len(similarity_matrix[0]) if similarity_matrix else 0
    # Build target Relation only once; keys are target attribute global index strings
    target_relation = build_target_relation(target_instances, n_source)
    # Deep copy the matrix; do not modify the original input
    updated = copy.deepcopy(similarity_matrix)

    fds:  List[FD]  = parse_fds(fd_file)   if fd_file  else []
    inds: List[IND] = parse_inds(ind_file) if ind_file else []
    # cfds: List[CFD] = parse_cfds(cfd_file) if cfd_file else []

    for fd in fds:
        _validate_fd(fd, similarity_matrix, updated,
                     target_relation, n_source, top_n,
                     rewards.get("fd", 0.01),
                     original=similarity_matrix, score_ceiling=float("inf"))
    for ind in inds:
        _validate_ind(ind, similarity_matrix, updated,
                      target_relation, n_source, top_n,
                      rewards.get("ind", 0.01),
                      original=similarity_matrix, score_ceiling=float("inf"))
    # for cfd in cfds:
    #     _validate_cfd(cfd, similarity_matrix, updated,
    #                   target_relation, n_source, top_n,
    #                   rewards.get("cfd", 0.01),
    #                   original=similarity_matrix, score_ceiling=float("inf"))
    # return updated


# ══════════════════════════════════════════════════════════════════════════════
# Public API Entry Points (new iterative version + original retained)
# ══════════════════════════════════════════════════════════════════════════════

def refinement(
    similarity_matrix: List[List[float]],
    target_path: str,
    data: str,
    top_n: int = 3,
    rewards: Optional[Dict[str, float]] = None,
) -> List[List[float]]:
    """Original single-round refinement entry point (retained; behavior fully consistent with previous version)."""
    with open(target_path, "r") as f:
        target_instances = list(csv.reader(f))

    fd_path  = f"../data/{data}/source/fds.txt"
    ind_path = f"../data/{data}/source/inds.txt"

    return _refinement(
        similarity_matrix=similarity_matrix,
        target_instances=target_instances,
        fd_file=fd_path,
        ind_file=ind_path,
        cfd_file=None,    # CFD validation disabled by default
        top_n=top_n,
        rewards=rewards if rewards else {"fd": 0.01, "ind": 0.01, "cfd": 0.01},
    )


def refinement_iterative(
    similarity_matrix: List[List[float]],
    target_instances: List[List[Any]],
    data: str,
    # Iteration control
    max_iter: int = 5,
    patience: int = 2,
    matrix_eps: float = 1e-4,
    # top_k range
    k_min: int = 1,
    k_max: int = 3,
    # Reward (no fixed base_reward; scale is auto-estimated)
    decay_factor: float = 0.6,
    score_ceiling: float = 1.5,
    k_stability: float = 10.0,
    k_exploration: float = 3.0,
    target_delta: float = 0.015,
    reward_min: float = 0.001,
    reward_max: float = 0.05,
    matrix_scale: float = 0.07,
    dep_mode: str = "all",   # 'all' (V0/V2), 'fd' (V3), 'ind' (V4)
    verbose: bool = False,
) -> List[List[float]]:
    """
    PDMVR iterative refinement public entry point.

    Reward scale requires no manual setting:
        Round 0 is auto-estimated from the matrix standard deviation
            (matrix_scale controls the strength);
        Round 1+ is inversely proportional to the validated count from
            the previous round; automatically amplified when dependencies
            are sparse.
    Recommended parameters: max_iter=5, patience=2, k_min=1, k_max=3,
        decay_factor=0.4, target_delta=0.015, matrix_scale=0.07
    """

    fd_path  = f"../data/{data}/source/fds.txt"
    ind_path = f"../data/{data}/source/inds.txt"

    return _refinement_iterative(
        similarity_matrix=similarity_matrix,
        target_instances=target_instances,
        fd_file=fd_path,
        ind_file=ind_path,
        cfd_file=None,    # CFD validation disabled by default
        max_iter=max_iter,
        patience=patience,
        matrix_eps=matrix_eps,
        k_min=k_min,
        k_max=k_max,
        decay_factor=decay_factor,
        score_ceiling=score_ceiling,
        k_stability=k_stability,
        k_exploration=k_exploration,
        target_delta=target_delta,
        reward_min=reward_min,
        reward_max=reward_max,
        matrix_scale=matrix_scale,
        dep_mode=dep_mode,
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════════════════════
# __main__: Run both original and iterative versions, save results for comparison
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir    = Path("../data/Movielens")
    sim_path    = data_dir / "res/sim_matrix.json"
    target_path = str(data_dir / "target/target.csv")

    similarity_matrix = json.load(open(sim_path, "r"))

    # ── Original single-round (unchanged, as baseline) ────────────────────────
    static_reward = {"fd": 0.01, "ind": 0.01, "cfd": 0.01}
    updated_static = refinement(
        similarity_matrix=similarity_matrix,
        target_path=target_path,
        data="Movielens",
        top_n=3,
        rewards=static_reward,
    )
    static_out = data_dir / "res/updated_similarity_static_0.01_0.01.json"
    with open(static_out, "w") as f:
        json.dump(updated_static, f, indent=2)
    print(f"Static single-round result saved -> {static_out}")

    # ── PDMVR iterative refinement: parameter tuning configurations ────────────
    # Format: (max_iter, patience, k_min, k_max, decay, ceiling, comment)
    # base_reward has been removed; reward scale is auto-estimated from matrix and validated count
    # Group header rows: ("GROUP", title_text), skipped during execution
    iter_configs = [

        ("GROUP", "Group 1: decay variation, others fixed (controls initial-round decay speed)"),
        (5, 2, 1, 3, 0.4, 1.5, "decay=0.4"),
        (5, 2, 1, 3, 0.6, 1.5, "decay=0.6"),
        (5, 2, 1, 3, 0.8, 1.5, "decay=0.8"),
        (5, 2, 1, 3, 1.0, 1.5, "decay=1.0(no initial decay)"),

        ("GROUP", "Group 2: k_max variation, others fixed (controls candidate set width)"),
        (5, 2, 1, 2, 0.4, 1.5, "k_max=2"),
        (5, 2, 1, 3, 0.4, 1.5, "k_max=3(recommended)"),
        (5, 2, 1, 5, 0.4, 1.5, "k_max=5"),

        ("GROUP", "Group 3: patience variation, others fixed"),
        (5, 1, 1, 3, 0.4, 1.5, "patience=1"),
        (5, 2, 1, 3, 0.4, 1.5, "patience=2"),
        (5, 3, 1, 3, 0.4, 1.5, "patience=3"),

        ("GROUP", "Group 4: score_ceiling variation, others fixed"),
        (5, 2, 1, 3, 0.4, 1.1, "ceiling=1.1"),
        (5, 2, 1, 3, 0.4, 1.5, "ceiling=1.5"),
        (5, 2, 1, 3, 0.4, 2.0, "ceiling=2.0"),

        ("GROUP", "Group 5: max_iter variation, others fixed"),
        (3, 2, 1, 3, 0.4, 1.5, "max_iter=3"),
        (5, 2, 1, 3, 0.4, 1.5, "max_iter=5"),
        (8, 2, 1, 3, 0.4, 1.5, "max_iter=8"),
    ]

    def _make_tag(max_iter, patience, k_min, k_max, decay, ceiling) -> str:
        """Generate a unique filename tag containing all parameters to avoid filename conflicts across groups."""
        return (
            f"iter{max_iter}"
            f"_p{patience}"
            f"_k{k_min}{k_max}"
            f"_d{int(decay * 10)}"
            f"_c{int(ceiling * 10)}"
        )

    for cfg in iter_configs:

        # Group header row: skip directly, do not run experiment
        if cfg[0] == "GROUP":
            continue

        max_iter, patience, k_min, k_max, decay, ceiling, comment = cfg
        tag = _make_tag(max_iter, patience, k_min, k_max, decay, ceiling)

        updated_iter = refinement_iterative(
            similarity_matrix=similarity_matrix,
            target_path=target_path,
            data="Movielens",
            max_iter=max_iter,
            patience=patience,
            k_min=k_min,
            k_max=k_max,
            decay_factor=decay,
            score_ceiling=ceiling,
            verbose=True,
        )

        out_path = data_dir / f"res/updated_sim_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(updated_iter, f, indent=2)
        print(f"  Iterative result saved -> {out_path}")