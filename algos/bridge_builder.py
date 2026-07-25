"""
column_mapping.py
=================
Map each column of the target schema to candidate columns of the source schema
based on a four-signal hybrid approach.

Problem Background
------------------
- Source schema: Relational database provided as a CSV directory; a single table
  may contain up to millions of rows. Each table has complete column names and
  data, and column Profiles have been pre-built.
- Target schema: Only a few instances (e.g., 3-5 rows), no column names
  (represented by positional indices 0, 1, 2, ...).

Core Idea
---------
For each target column, find the corresponding source column candidates
(column mapping) through four hybrid signals.

Four Column Mapping Signals
---------------------------
  S1 Value-domain Subset   (weight=0.40): Whether target column values appear
                                           in the source column's value domain
  S2 Type Compatibility    (weight=0.20): Whether the semantic data types of
                                           the two columns are compatible
  S3 Distribution Similarity (weight=0.20): Range overlap for numeric columns;
                                             length and charset similarity for
                                             string columns
  S4 Semantic Similarity   (weight=0.20): Edit-distance-based; captures encoding
                                           differences such as M/F vs male/female

Important Note: Type Inference for Target Values
-------------------------------------------------
All values in target instances are treated as raw strings (str) read from CSV.
Type inference uses the exact same string-pattern matching logic as
_detect_dtype() in column_profiler.py, rather than relying on Python's
native isinstance() type checks.

Performance Strategy
--------------------
  - Column Profile precomputation (column_profiler.py): All signal computations
    are O(1) Profile lookups
  - Profile disk cache: Loaded directly when CSVs are unchanged, avoiding
    redundant scans
"""

from __future__ import annotations
from sys import exception

from util import get_topk_colidx
import json
import os
import re
import time
from pathlib import Path
from typing import Any, List

# column_profiler provides column Profile loading/building capabilities
from column_profiler import load_or_build_profiles

# Obtain the absolute path of the directory containing this .py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# Four column mapping signals (all Profile-based, O(1) queries)
# ─────────────────────────────────────────────────────────────

# Set of numeric types, used for type compatibility checks
_NUMERIC_TYPES = {"int", "float"}

# Default signal weights: value-domain subset has the highest weight (strongest exact-match signal)
DEFAULT_WEIGHTS = {
    "subset":   0.40,   # Signal 1: Value-domain subset
    "type":     0.20,   # Signal 2: Type compatibility
    "dist":     0.20,   # Signal 3: Distribution similarity
    "semantic": 0.20,   # Signal 4: Semantic similarity
}


# Regex pattern for recognizing numeric strings (consistent with column_profiler.py)
# Supports integers, decimals, and scientific notation, with an optional leading minus sign.
NUM_PATTERN = re.compile(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$')

# Regex to distinguish pure integers from floats (no decimal point or scientific notation).
INT_PATTERN = re.compile(r'^-?\d+$')


# ─────────────────────────────────────────────────────────────
# Target column type inference (aligned with column_profiler._detect_dtype)
# ─────────────────────────────────────────────────────────────

def _detect_target_dtype(vals: list[str]) -> str:
    """
    Infer the semantic data type from the raw string values of a target column.

    Uses the exact same string-pattern matching logic as
    column_profiler._detect_dtype() to ensure that the inferred target type
    is comparable to the dtype field in the source Profile.

    All input values are treated as strings (even if the caller has already
    parsed them as int/float, they are first converted back via str() before
    inference). Null values (empty strings, null, none, etc.) are skipped.

    Inference Rules (by priority)
    -----------------------------
    1. All values match the pure integer pattern (-?\d+)          -> 'int'
    2. All values match the numeric pattern (decimals/sci. notation) -> 'float'
    3. All values belong to the boolean vocabulary (true/false/yes/no, etc.) -> 'bool'
    4. Otherwise                                                  -> 'str'

    If multiple types coexist (e.g., some numeric and some string), returns 'mixed'.

    Parameters
    ----------
    vals : All values of the target column (str type, or any type convertible via str())

    Returns
    -------
    'int' | 'float' | 'bool' | 'str' | 'mixed'
    """
    # Null value vocabulary (consistent with column_profiler._ColAgg.feed)
    _NULL_WORDS = {"", "null", "none", "na", "nan", "n/a"}
    # Boolean value vocabulary (consistent with column_profiler._detect_dtype)
    _BOOL_VALS  = {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}

    # Convert all values to strings uniformly, filtering out null values
    samples = [str(v).strip() for v in vals
               if v is not None and str(v).strip().lower() not in _NULL_WORDS]

    if not samples:
        return "str"


    all_int   = True    # Whether all values are pure integers
    all_num   = True    # Whether all values are numeric
    all_bool  = True    # Whether all values are boolean
    has_str   = False   # Whether any plain string (non-numeric, non-boolean) appears

    for v in samples:
        vl = v.strip()           # Regex matching must use the original string (not lowered)
        vl_lower = vl.lower()

        # 1. Check for pure integer (using INT_PATTERN)
        if not INT_PATTERN.match(vl):
            all_int = False

        # 2. Check for general numeric
        if not NUM_PATTERN.match(vl):
            all_num = False

        # 3. Check for boolean
        if vl_lower not in _BOOL_VALS:
            all_bool = False

        # 4. Flag whether a plain string appears (for mixed-type detection)
        is_num = NUM_PATTERN.match(vl) is not None
        is_bool = vl_lower in _BOOL_VALS
        if not is_num and not is_bool:
            has_str = True

    # ===================== Rule 5: Mixed type =====================
    # Numeric/boolean values coexisting with plain strings -> mixed
    if (all_num or all_bool) and has_str:
        return "mixed"

    # Return by priority
    if all_int:
        return "int"
    if all_num:
        return "float"
    if all_bool:
        return "bool"

    return "str"

# ── Signal 1: Value-domain Subset ─────────────────────────────

def signal_value_subset(
    target_vals: list[Any],
    prof: dict,
) -> float:
    """
    Check whether the value domain of the target column is a subset of the
    source column's value domain.

    This is the most direct matching signal: if every value in the target
    column appears in the source column, the two columns are highly likely
    to be semantically identical (or the target is a sub-table of the source).

    Two Modes
    ---------
    Exact mode (prof["exact_unique"] == True):
      The source has <= UNIQUE_VALS_CAP unique values, and unique_vals is the
      complete set.
      Score = matched count / target unique count (1.0 for a strict subset).

    Approximate mode (prof["exact_unique"] == False):
      The source is a high-cardinality column; unique_vals is only a random
      sample, so exact membership checks are not possible.
      For unmatched values, the HLL-estimated cardinality is used to assess
      the probability of "possible inclusion":
        p = HLL cardinality / total rows (= expected value-domain coverage)
      Approximate score = hit ratio + miss ratio * p

    Parameters
    ----------
    target_vals : All values of the target column (compared after uniform str conversion)
    prof        : The Profile dictionary for this source column

    Returns
    -------
    Score in [0, 1]
    """
    t_set    = {str(v) for v in target_vals if v is not None}
    s_unique = set(prof.get("unique_vals", []))
    exact    = prof.get("exact_unique", False)

    if not t_set:
        return 0.0

    matched = len(t_set & s_unique)

    if exact:
        return matched / len(t_set)
    else:
        unmatched    = len(t_set) - matched
        hll_card     = prof.get("hll_cardinality") or len(s_unique)
        n_rows       = prof.get("n_rows", 1)
        p_hit_random = min(1.0, hll_card / max(n_rows, 1))
        approx_score = matched / len(t_set) + (unmatched / len(t_set)) * p_hit_random
        return min(1.0, approx_score)


# ── Signal 2: Type Compatibility ──────────────────────────────

def signal_type_compat(
    target_vals: list[Any],
    prof: dict,
) -> float:
    """
    Evaluate the data type compatibility between a target column and a source column.

    Type Inference Strategy
    -----------------------
    The target type is inferred from string content via _detect_target_dtype(),
    which is fully consistent with _detect_dtype() in column_profiler.py.
    The source type is read directly from the Profile's dtype field.

    Type Compatibility Matrix
    -------------------------
    Same type            -> 1.0  (exact match)
    int <-> float        -> 0.8  (numeric compatibility, possibly precision difference)
    bool -> int/float    -> 0.5  (booleans are typically encoded as 0/1)
    mixed <-> any        -> 0.2  (uncertain)
    str <-> int/float    -> 0.3  (possible encoding difference, e.g. "male" <-> 1)
    Otherwise            -> 0.0

    Cardinality Similarity Bonus
    ----------------------------
    If target cardinality / source cardinality is in [0.05, 5.0], an extra 0.1
    is added. This rule prevents mapping a low-cardinality enumeration column
    (e.g., gender: 2 values) to a high-cardinality ID column. The ratio range
    is intentionally generous (5x) because the target has only a few rows,
    so sampled cardinality is naturally biased low.

    Parameters
    ----------
    target_vals : Target column value list (str or other types; uniformly inferred via string patterns)
    prof        : Source column Profile

    Returns
    -------
    Score in [0, 1] (clamped via min)
    """
    # Convert target values uniformly to strings, then infer semantic type via string patterns
    str_vals = [str(v) for v in target_vals if v is not None]
    t_type   = _detect_target_dtype(str_vals)

    # Source type comes from the Profile (inferred during CSV scan)
    s_type = prof.get("dtype", "str")

    # Type compatibility matrix
    if t_type == s_type:
        type_score = 1.0
    elif {t_type, s_type} <= _NUMERIC_TYPES:
        type_score = 0.8
    elif t_type == "bool" and s_type in _NUMERIC_TYPES:
        type_score = 0.5
    elif "mixed" in (t_type, s_type):
        type_score = 0.2
    elif t_type != s_type and (t_type == "str" or s_type == "str"):
        type_score = 0.3   # str <-> numeric, possibly different encoding
    else:
        type_score = 0.0

    # Cardinality similarity bonus (prevents confusing ID columns with enumeration columns)
    t_card = len({str(v) for v in target_vals if v is not None})
    s_card = prof.get("n_unique", 0)
    if s_card > 0:
        ratio      = t_card / s_card
        card_bonus = 0.1 if 0.05 <= ratio <= 5.0 else 0.0
    else:
        card_bonus = 0.0

    return min(1.0, type_score + card_bonus)


# ── Signal 3: Distribution Similarity (O(1)) ─────────────────

def signal_distribution(
    target_vals: list[Any],
    prof: dict,
) -> float:
    """
    Evaluate the numeric/string distribution similarity between a target column
    and a source column.

    The source column type is determined by the Profile's dtype field (numeric
    columns follow the numeric branch; all others follow the string branch).
    Target values are first inferred as a semantic type via string patterns to
    determine whether they can participate in numeric comparisons.

    Numeric Column Logic
    --------------------
    Compute the range overlap between target [min, max] and source [min, max]:
      range_score = overlap_length / target_range  in [0, 1]
    Also compute the proximity of the target median to the source [p25, p75]:
      quantile_score = 1 - dist_to_IQR / IQR  in [0, 1]
    Final = 0.6 * range_score + 0.4 * quantile_score

    String Column Logic
    -------------------
    (1) Average length similarity: 1 - |t_avg_len - s_avg_len| / max(both)
    (2) Character-set consistency: same char_class -> 1.0; alpha <-> alnum -> 0.5;
        otherwise 0.2
    Final = 0.5 * length score + 0.5 * charset score

    Parameters
    ----------
    target_vals : Target column value list (all values treated as strings)
    prof        : Source column Profile

    Returns
    -------
    Score in [0, 1]
    """
    s_type = prof.get("dtype", "str")

    # ── Numeric distribution branch ──
    if s_type in _NUMERIC_TYPES:
        # Attempt to convert string values to numeric; penalize those that cannot be converted
        t_num: list[float] = []
        for v in target_vals:
            if v is None:
                continue
            sv = str(v).strip()
            if NUM_PATTERN.match(sv):
                t_num.append(float(sv))
        if not t_num:
            return 0.1   # Target values cannot be parsed as numeric; mild penalty

        t_min, t_max = min(t_num), max(t_num)
        s_min = prof.get("num_min")
        s_max = prof.get("num_max")
        if s_min is None or s_max is None:
            return 0.0

        # Range overlap ratio
        overlap_lo = max(t_min, s_min)
        overlap_hi = min(t_max, s_max)
        if overlap_hi < overlap_lo:
            return 0.0   # Ranges do not overlap at all

        t_range     = max(t_max - t_min, 1e-9)
        range_score = min(1.0, (overlap_hi - overlap_lo) / t_range)

        # Quantile proximity: does the target median fall within the source IQR?
        t_med          = sorted(t_num)[len(t_num) // 2]
        s_p25          = prof.get("num_p25", s_min)
        s_p75          = prof.get("num_p75", s_max)
        iqr            = max(s_p75 - s_p25, 1e-9)
        dist_to_iqr    = max(0.0, min(abs(t_med - s_p25), abs(t_med - s_p75)) - iqr)
        quantile_score = max(0.0, 1.0 - dist_to_iqr / (iqr + 1e-9))

        return 0.6 * range_score + 0.4 * quantile_score

    # ── String distribution branch ──
    t_strs = [str(v) for v in target_vals if v is not None]
    if not t_strs:
        return 0.0

    # Average length similarity
    t_avg_len = sum(len(s) for s in t_strs) / len(t_strs)
    s_avg_len = prof.get("str_avg_len") or 0.0
    max_len   = max(t_avg_len, s_avg_len, 1e-9)
    len_score = 1.0 - abs(t_avg_len - s_avg_len) / max_len

    # Character-set type consistency (consistent with column_profiler._char_class logic)
    def _cclass(strs: list[str]) -> str:
        all_alpha = all(s.replace(" ", "").replace("-", "").isalpha() for s in strs if s)
        all_digit = all(s.isdigit() for s in strs if s)
        all_alnum = all(s.replace(" ", "").isalnum() for s in strs if s)
        if all_alpha: return "alpha"
        if all_digit: return "digit"
        if all_alnum: return "alnum"
        return "mixed"

    s_cc      = prof.get("str_char_class", "mixed")
    t_cc      = _cclass(t_strs)
    cls_score = 1.0 if t_cc == s_cc else (0.5 if {t_cc, s_cc} <= {"alpha", "alnum"} else 0.2)

    return 0.5 * len_score + 0.5 * cls_score


# ── Signal 4: Semantic Similarity ─────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings (case-insensitive).

    Dynamic programming implementation, O(|a| * |b|) time, O(|b|) space
    (rolling array optimization). Edit operations: insertion, deletion,
    substitution, each with cost 1.

    Parameters
    ----------
    a, b : The two strings to compare

    Returns
    -------
    Minimum number of edit operations (integer)
    """
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la

    prev = list(range(lb + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb)
            ))
        prev = curr
    return prev[lb]


def _norm_edit_sim(a: str, b: str) -> float:
    """
    Normalized edit similarity = 1 - edit_distance / max(len(a), len(b)).
    Result in [0, 1]; identical strings score 1.0; completely different strings
    score close to 0.0.
    """
    return 1.0 - _edit_distance(a, b) / max(len(a), len(b), 1)


def signal_semantic(
    target_vals: list[Any],
    prof: dict,
) -> float:
    """
    Evaluate semantic similarity based on value-level edit distance, capturing
    encoding inconsistencies.

    Use Case
    --------
    When source and target use different encodings for the same semantics,
    the value-domain subset signal (S1) fails:
      - Gender: source uses "M/F", target uses "male/female"
        -> subset=0, semantic~0.6
      - Boolean: source uses "0/1", target uses "false/true"
        -> subset=0, semantic~0.2
      - Code: source uses "CN", target uses "China"
        -> subset=0, semantic~0.2

    Core Algorithm
    --------------
    For each unique target value tv:
      best_sim(tv) = max over all sv in source_unique_vals of norm_edit_sim(tv, sv)
    Final score = mean of best_sim over all target unique values

    Parameters
    ----------
    target_vals : All values of the target column
    prof        : Source column Profile (unique_vals field used for comparison)

    Returns
    -------
    Score in [0, 1]
    """
    t_unique = list({str(v) for v in target_vals if v is not None})
    s_unique = prof.get("unique_vals", [])
    if not t_unique or not s_unique:
        return 0.0

    total = 0.0
    for tv in t_unique:
        best = max(_norm_edit_sim(tv, sv) for sv in s_unique)
        total += best
    return total / len(t_unique)


# ─────────────────────────────────────────────────────────────
# Four-signal fusion
# ─────────────────────────────────────────────────────────────

def score_column_mapping(
    target_vals: list[Any],
    prof: dict,
    weights: dict[str, float] | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Compute the four-signal fused matching score for a single
    (target column, source column) pair.

    Fusion Formula
    --------------
    total = w_subset * S1 + w_type * S2 + w_dist * S3 + w_semantic * S4

    Default Weights (DEFAULT_WEIGHTS)
    ---------------------------------
    subset=0.40, type=0.20, dist=0.20, semantic=0.20

    Weight Tuning Suggestions
    -------------------------
    - If data quality is high and encodings are consistent: increase subset
      weight (0.5-0.6), decrease semantic
    - If there are many encoding inconsistencies (M/F <-> male/female):
      increase semantic (0.3-0.4)
    - If numeric column distributions differ significantly: increase dist (0.3)

    Parameters
    ----------
    target_vals : Target column value list (all values treated as strings; type inferred via string patterns)
    prof        : Source column Profile
    weights     : Custom weight dictionary; None to use DEFAULT_WEIGHTS
    verbose     : If True, print detailed four-signal score breakdown

    Returns
    -------
    {
      "total":    float,   # Weighted total score in [0, 1]
      "subset":   float,   # S1 score
      "type":     float,   # S2 score
      "dist":     float,   # S3 score
      "semantic": float,   # S4 score
    }
    """
    w     = weights or DEFAULT_WEIGHTS
    s1    = signal_value_subset(target_vals, prof)
    s2    = signal_type_compat(target_vals, prof)
    s3    = signal_distribution(target_vals, prof)
    s4    = signal_semantic(target_vals, prof)
    total = w["subset"] * s1 + w["type"] * s2 + w["dist"] * s3 + w["semantic"] * s4

    if verbose:
        print(f"    subset={s1:.3f}  type={s2:.3f}  dist={s3:.3f}  semantic={s4:.3f}  → {total:.3f}")

    return {"total": total, "subset": s1, "type": s2, "dist": s3, "semantic": s4}


def find_candidate_column_mappings(
    instances: list[list[Any]],
    table_profiles: dict[str, dict],
    weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> dict[int, list[tuple[str, dict[str, float]]]]:
    """
    For each target column, compute its four-signal fused score against all source
    columns of a given table, and return the full results.

    Return Structure
    ----------------
    {
      target_col_idx: [
        (source_col_name, score_dict),  # Sorted by total descending, includes all columns of this table
        ...
      ]
    }

    Note: This function does not apply top_k truncation; truncation is performed
    by the caller (run()) after cross-table merging.

    Parameters
    ----------
    instances      : Target instance list; each row is a value list (all values treated as strings)
    table_profiles : {col_name: profile_dict} for the current source table
    weights        : Signal weights
    verbose        : Whether to print detailed scores for each column pair

    Returns
    -------
    See "Return Structure" above
    """
    if not instances:
        return {}
    n_cols     = len(instances[0])
    candidates: dict[int, list[tuple[str, dict]]] = {}

    for t_idx in range(n_cols):
        t_vals = [row[t_idx] for row in instances]
        scored: list[tuple[str, dict]] = []

        for s_col, prof in table_profiles.items():
            if verbose:
                print(f"  T[{t_idx}] ↔ {s_col}:")
            sc = score_column_mapping(t_vals, prof, weights, verbose=verbose)
            scored.append((s_col, sc))

        # Sort by total score descending (no truncation; deferred to cross-table top_k selection)
        scored.sort(key=lambda x: x[1]["total"], reverse=True)
        candidates[t_idx] = scored

    return candidates


# ─────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────

def print_candidate_mappings(
    candidates: dict[int, list[tuple[str, dict[str, float]]]],
    instances: list[list[Any]],
) -> None:
    """
    Print candidate source columns and their score details for each target
    column in a human-readable format.

    Parameters
    ----------
    candidates : Return value from find_candidate_column_mappings
    instances  : Target instance list (used to print sample values)
    """
    print("\n" + "=" * 65)
    print("Target Column Candidate Mappings (by confidence score, descending)")
    print("=" * 65)
    n_cols = len(instances[0]) if instances else 0
    for t_idx in range(n_cols):
        sample = [row[t_idx] for row in instances]
        cands  = candidates.get(t_idx, [])
        print(f"\n  T[{t_idx}]  Sample values: {sample}")
        if cands:
            for s_col, sc in cands:
                detail = (
                    f"subset={sc.get('subset', 0):.2f}  "
                    f"type={sc.get('type', 0):.2f}  "
                    f"dist={sc.get('dist', 0):.2f}  "
                    f"sem={sc.get('semantic', 0):.2f}  "
                    f"-> total={sc.get('total', 0):.2f}"
                )
                print(f"    -> '{s_col}'  [{detail}]")
        else:
            print("    -> (no candidate columns meeting threshold)")


# ─────────────────────────────────────────────────────────────
# Public main entry point
# ─────────────────────────────────────────────────────────────

def build_bridge(
    data_dir: str | Path,
    target_instances: list[list[Any]],
    source_number: int,
    cache_dir: str | Path | None = None,
    weights: dict[str, float] | None = None,
    top_k: int = 3,
    verbose_signals: bool = False,
    bridge_edge_path: str = None,
    force_rebuild_profiles: bool = True,
    random_bridge: bool = False,
) -> dict:
    """
    Unified entry point for the complete two-step column mapping pipeline.

    Steps
    -----
    Step 1: Load or build source column Profiles (with disk cache)
    Step 2: Compute candidate source columns for each target column via four signals

    Important: Values in target_instances are all treated as strings.
    If the caller has performed type conversion (e.g., converting "123" to int),
    the function will automatically restore them to strings before type inference.

    Parameters
    ----------
    data_dir                : Data folder directory path
    target_instances        : Target instance list, e.g. [["1", "Alice", "F"], ...]
    source_number           : Number of source columns (used for target column index offset;
                              assumes source column indices start at 0, target column indices
                              start at source_number)
    cache_dir               : Profile cache directory; None to use data_dir/source/.profile_cache/
    weights                 : Custom four-signal weights; None to use defaults
    top_k                   : Number of candidates to retain per target column; default 3
    verbose_signals         : If True, print detailed signal scores for each column pair
    bridge_edge_path        : Default location is inside data_dir/target/
    force_rebuild_profiles  : If True, ignore disk cache and force a full CSV rescan
    random_bridge           : Ablation variant V2 switch. When True, skips four-signal
                              scoring + global top_k selection, and instead randomly
                              uniformly samples top_k source attributes as bridge edges
                              for each target column (controlled by random.seed(42) at
                              entry point, so results are reproducible). Default False
                              (V0/V1 behavior).

    Returns
    -------
    {
      "candidates" : {table_name: {target_col_idx: [(source_col, score_dict), ...]}},
      "profiles"   : {table_name: {col_name: profile_dict}},
    }
    """
    # Set random seed to ensure reproducible random sampling results in Profiles
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)


    data_dir = Path(data_dir)

    # Step 1: Load or build source column Profiles
    print("=" * 65)
    print("Column Profile Loading / Building")
    print("=" * 65)
    t0 = time.time()
    all_profiles = load_or_build_profiles(
        Path(data_dir / "source"), cache_dir=cache_dir,
        force_rebuild=force_rebuild_profiles,
    )
    print(f"  Profile ready  ({time.time() - t0:.2f}s)")

    # Step 2: Compute scores for each table (no truncation; for cross-table merging)
    all_candidates: dict[str, dict] = {}
    for table_name, table_profiles in all_profiles.items():
        print(f"\n{'=' * 60}")
        print(f"Computing [Table: {table_name}]  Column Mapping Scores")
        print("=" * 60)

        candidates = find_candidate_column_mappings(
            target_instances, table_profiles, weights, verbose_signals,
        )
        all_candidates[table_name] = candidates


        # Print single-table summary (full; for debugging)
        # print("\n-- Single-table candidate summary --")
        # for t_idx, cands in candidates.items():
        #     names = [f"{c}({s['total']:.2f})" for c, s in cands]
        #     print(f"  T[{t_idx}] -> {names or '(none)'}")

    # For testing
    # with open(Path(data_dir) / "test" / "all_candidates.json", "w", encoding="utf-8") as f:
    #     json.dump(all_candidates, f, indent=2)

    # Step 3: Cross-table merging; for each target column, select global top_k, generate bridge edges
    print(f"\n{'=' * 60}")
    print(f"Bridge Edge Summary (across all source tables, global top_k={top_k})")
    print(f"{'=' * 60}\n")

    target_num = len(target_instances[0]) if target_instances else 0

    all_scores = [[0 for _ in range(source_number)] for _ in range(target_num)]
    # Stores scores of target columns vs all source columns; rows = target columns, cols = source column scores

    for _, candidates in all_candidates.items():
        for t_idx, cands in candidates.items():
            for s_col, sc in cands:
                s_idx = int(s_col)  
                t_idx = int(t_idx)
                all_scores[t_idx][s_idx] = sc["total"]

    # For testing: save the full score matrix for debugging analysis
    # with open(Path(data_dir) / "test" / "all_scores.json", "w", encoding="utf-8") as f:
    #     json.dump(all_scores, f, indent=2)

    # For each target column, select source columns as bridge edges
    if random_bridge:
        # ── V2 (Random Bridge Edges) ──
        # Without any signal scoring, randomly uniformly sample top_k source attributes
        # for each target column.
        # build_bridge entry already sets random.seed(42), so sampling is reproducible;
        # candidate count matches the full method (=top_k).
        # Note: all_candidates (true four-signal scores) are still returned as usual,
        #       so build_target_edge's mapped dependency generation is completely
        #       unaffected (consistent with the ablation design).
        k = min(top_k, source_number)
        bdg = {t_idx: random.sample(range(source_number), k) for t_idx in range(target_num)}
    else:
        # ── V0/V1 (default): Select global top_k scoring ──
        bdg = get_topk_colidx(all_scores, top_k)
    bridge_edges = {k + source_number: v for k, v in bdg.items()}

    # Print final bridge edge summary
    for t_idx, s_idxs in bridge_edges.items():
        names = [f"col{s_idx}({all_scores[int(t_idx)-source_number][s_idx]:.2f})" for s_idx in s_idxs]
        print(f"  T[{t_idx}] -> {names or '(none)'}")


    # Save to file
    if bridge_edge_path:
        with open(bridge_edge_path, "w", encoding="utf-8") as f:
            json.dump(bridge_edges, f, indent=2)
    else:
        with open(Path(data_dir) / "target" / "bridge_edges.json", "w", encoding="utf-8") as f:
            json.dump(bridge_edges, f, indent=2)


    return {"all_candidates": all_candidates, "profiles": all_profiles}





# The DEFAULT_WEIGHTS global variable can be adjusted according to experimental requirements; it holds the weight for each signal