"""
column_mapping.py
=================
Map each column of the target schema to candidate columns in the source schema
based on four hybrid signals.

Problem Background
------------------
- Source schema: A relational database provided as a directory of CSV files,
  with individual tables containing up to millions of rows.
  Each table has complete column names and data, with column profiles
  pre-built.
- Target schema: Only a few instances available (e.g., 3-5 rows), no column
  names (represented by positional indices 0, 1, 2, ...).

Core Approach
--------------
For each column in the target, identify the corresponding source column
candidates (column mapping) through four hybrid signals.

Four Column Mapping Signals
----------------------------
  S1 Value Domain Subset   (weight=0.40): Whether target column values appear
      in the source column's value domain
  S2 Type Compatibility    (weight=0.20): Whether the semantic data types of
      the two columns are compatible
  S3 Distribution Similarity (weight=0.20): Range overlap for numeric columns;
      length and charset similarity for string columns
  S4 Semantic Similarity   (weight=0.20): Based on edit distance, capturing
      encoding differences such as M/F <-> male/female

Important Note: Type Inference for Target Values
-------------------------------------------------
All values in target instances are treated as raw strings (str) read from CSV.
Type inference uses the exact same string pattern matching logic as
_detect_dtype() in column_profiler.py, rather than relying on Python's
native isinstance() type checking.

Performance Strategy
--------------------
  - Column profile precomputation (column_profiler.py): All signal computations
    are done via O(1) profile lookups
  - Profile disk caching: When CSV files are unchanged, profiles are loaded
    directly without re-scanning
"""

from __future__ import annotations
import os
from typing import Any, List, FrozenSet
from util import get_topk_colidx, get_topk_colidx, parse_cfds, parse_fds, parse_inds, build_target_relation
from dependency_validator import CFD, FD, IND, Relation
# dependency_validator provides dependency data structures and validation functions
from dependency_validator import (
    FD, CFD, IND,
    Relation,
    validate_fd, validate_cfd, validate_ind,
)

# column_profiler provides column profile loading/building capabilities
from column_profiler import load_or_build_profiles


def _remap_attrs(attrs: list[str], assignment: dict[int, int]) -> list[str]:
    """
    Replace source attribute indices in the string list with the corresponding
    target global index strings according to the assignment mapping.
    Identical to _remap_attrs in dependency_reranker.py.
    """
    return [str(assignment[int(a)]) for a in attrs]
 

def _enumerate_assignments(
    src_indices: list[int],
    candidate_map: dict[int, list[int]],
) -> list[dict[int, int]]:
    """
    Enumerate all Cartesian product assignments for the given source attribute
    index list based on the candidate sets.
    Assignment format: {src_idx: tgt_global_idx}
    Returns an empty list if any src_idx has no candidates (the dependency
    cannot be mapped).
    """
    import itertools
    candidates_per_src = []
    for src_idx in src_indices:
        if src_idx not in candidate_map or not candidate_map[src_idx]:
            return []
        candidates_per_src.append(candidate_map[src_idx])
    return [
        dict(zip(src_indices, combo))
        for combo in itertools.product(*candidates_per_src)
    ]
 
 
def _collect_mapped_fds(
    fds: list[FD],
    candidate_map: dict[int, list[int]],
    target_relation: Relation,
) -> list[FD]:
    """
    Enumerate candidate assignments for the FD, validate, and collect results
    (using global index strings, deduplicated).
    Validation logic is identical to _validate_fd in dependency_reranker.py.
    """
    seen: set[tuple] = set()
    result: list[FD] = []
    for fd in fds:
        involved = list(dict.fromkeys(int(a) for a in fd.lhs + fd.rhs))
        for assignment in _enumerate_assignments(involved, candidate_map):
            mapped_lhs = _remap_attrs(fd.lhs, assignment)
            mapped_rhs = _remap_attrs(fd.rhs, assignment)
            # Duplicate column filter: if A,B -> C maps such that A=B, the assignment is invalid, skip
            if len(set(mapped_lhs)) < len(mapped_lhs):
                continue
            # Triviality filter (A->A,C)
            if set(mapped_rhs) & set(mapped_lhs):
                continue
            mapped_fd = FD(lhs=mapped_lhs, rhs=mapped_rhs)
            if validate_fd(target_relation, mapped_fd).satisfied:
                key = (tuple(mapped_lhs), tuple(mapped_rhs))
                if key not in seen:
                    seen.add(key)
                    result.append(FD(lhs=mapped_lhs, rhs=mapped_rhs))
    return remove_redundant_fds(result)

 
 
def _collect_mapped_inds(
    inds: list[IND],
    candidate_map: dict[int, list[int]],
    target_relation: Relation,
) -> list[IND]:
    """
    Enumerate candidate assignments for the IND (LHS/RHS enumerated independently),
    validate, and collect results (deduplicated).
    Validation logic is identical to _validate_ind in dependency_reranker.py.
    """
    seen: set[tuple] = set()
    result: list[IND] = []
    for ind in inds:
        lhs_src = [int(a) for a in ind.lhs_attrs]
        rhs_src = [int(a) for a in ind.rhs_attrs]
        for lhs_assign in _enumerate_assignments(lhs_src, candidate_map):
            for rhs_assign in _enumerate_assignments(rhs_src, candidate_map):
                mapped_lhs = _remap_attrs(ind.lhs_attrs, lhs_assign)
                mapped_rhs = _remap_attrs(ind.rhs_attrs, rhs_assign)
                # Triviality filter
                if set(mapped_lhs) & set(mapped_rhs):
                    continue
                mapped_ind = IND(lhs_attrs=mapped_lhs, rhs_attrs=mapped_rhs)
                val = validate_ind(mapped_ind,
                                   lhs_relation=target_relation,
                                   rhs_relation=target_relation)
                if val.satisfied:
                    key = (tuple(mapped_lhs), tuple(mapped_rhs))
                    if key not in seen:
                        seen.add(key)
                        result.append(IND(lhs_attrs=mapped_lhs, rhs_attrs=mapped_rhs))
    return result
 
 
def _collect_mapped_cfds(
    cfds: list[CFD],
    candidate_map: dict[int, list[int]],
    target_relation: Relation,
) -> list[CFD]:
    """
    Enumerate candidate assignments for the CFD, validate, and collect results
    (deduplicated).
    Vacuous satisfaction filtering is identical to _validate_cfd in
    dependency_reranker.py.
    """
    seen: set[tuple] = set()
    result: list[CFD] = []
    for cfd in cfds:
        involved = list(dict.fromkeys(int(a) for a in cfd.lhs + cfd.rhs))
        for assignment in _enumerate_assignments(involved, candidate_map):
            mapped_lhs = _remap_attrs(cfd.lhs, assignment)
            mapped_rhs = _remap_attrs(cfd.rhs, assignment)
            # Duplicate column filter: if A,B -> C maps such that A=B, the assignment is invalid, skip
            if len(set(mapped_lhs)) < len(mapped_lhs):
                continue
            # Triviality filter
            if set(mapped_rhs) & set(mapped_lhs):
                continue
            mapped_pattern = {
                str(assignment[int(k)]): v
                for k, v in cfd.pattern.items()
            }
            mapped_cfd = CFD(lhs=mapped_lhs, rhs=mapped_rhs, pattern=mapped_pattern)
            val = validate_cfd(target_relation, mapped_cfd)
            # Vacuous satisfaction filter
            vacuous = any("vacuously" in v for v in val.violations)
            if val.satisfied and not vacuous:
                key = (tuple(mapped_lhs), tuple(mapped_rhs),
                       tuple(sorted(mapped_pattern.items())))
                if key not in seen:
                    seen.add(key)
                    result.append(CFD(lhs=mapped_lhs, rhs=mapped_rhs,
                                      pattern=mapped_pattern))
    return result


def remove_redundant_fds(fds: List[FD]) -> List[FD]:
    """
    Remove FDs whose LHS is a strict superset of another FD's LHS
    that has the same or covering RHS.

    An FD B → Y is redundant if there exists A → Y such that A ⊂ B (proper subset).
    In other words: if we can derive the same RHS from fewer attributes, the
    larger-LHS FD adds no information.
    """
    result = []
    for i, fd in enumerate(fds):
        lhs_i: FrozenSet[str] = frozenset(fd.lhs)
        rhs_i: FrozenSet[str] = frozenset(fd.rhs)
        dominated = False

        for j, other in enumerate(fds):
            if i == j:
                continue
            lhs_j: FrozenSet[str] = frozenset(other.lhs)
            rhs_j: FrozenSet[str] = frozenset(other.rhs)

            # other dominates fd if:
            #   1. other's LHS is a *strict subset* of fd's LHS  (fewer attributes suffice)
            #   2. other's RHS covers fd's RHS                    (same or more attributes determined)
            if lhs_j < lhs_i and rhs_i <= rhs_j:
                dominated = True
                break

        if not dominated:
            result.append(fd)

    return result

 
# ─────────────────────────────────────────────────────────────
# Dependency output (same format as dependency_reranker.py input files)
# Attribute names are all global index strings
# ─────────────────────────────────────────────────────────────
 
def _write_fds(fds: list[FD], out_path: str) -> None:
    """Write FD file. Format: 0,1 -> 2 (global indices)"""
    with open(out_path, "w", encoding="utf-8") as f:
        for fd in fds:
            f.write(f"{','.join(fd.lhs)} -> {','.join(fd.rhs)}\n")
    print(f"[FD]  Written {len(fds)} entries -> {out_path}")
 
 
def _write_inds(inds: list[IND], out_path: str) -> None:
    """Write IND file. Format: 0[=1 (global indices)"""
    with open(out_path, "w", encoding="utf-8") as f:
        for ind in inds:
            f.write(f"{','.join(ind.lhs_attrs)}[={','.join(ind.rhs_attrs)}\n")
    print(f"[IND] Written {len(inds)} entries -> {out_path}")
 
 
def _write_cfds(cfds: list[CFD], out_path: str) -> None:
    """Write CFD file. Format: (0, 1=1996) => 2 (global indices)"""
    with open(out_path, "w", encoding="utf-8") as f:
        for cfd in cfds:
            lhs_items = [
                f"{a}={cfd.pattern[a]}" if a in cfd.pattern else a
                for a in cfd.lhs
            ]
            rhs_items = [
                f"{a}={cfd.pattern[a]}" if a in cfd.pattern else a
                for a in cfd.rhs
            ]
            f.write(f"({', '.join(lhs_items)}) => {', '.join(rhs_items)}\n")
    print(f"[CFD] Written {len(cfds)} entries -> {out_path}")
 
 
# ─────────────────────────────────────────────────────────────
# Public main entry point: build_target_edge
# ─────────────────────────────────────────────────────────────
 
def build_target_edge(
    all_candidates: dict,
    target_instances: list[list[Any]],
    n_source: int,
    top_k: int = 3,
    fd_file: str | None = None,
    ind_file: str | None = None,
    cfd_file: str | None = None,
    out_fd_path: str = "target_fds.txt",
    out_ind_path: str = "target_inds.txt",
    out_cfd_path: str = "target_cfds.txt",
) -> dict:
    """
    Using the candidate mapping set and source-side dependencies from
    build_bridge(), map dependencies to the target schema, validate them
    against target instances, and write the validated dependencies to text
    files (in the same format as dependency_reranker.py input).

    Parameters
    ----------
    all_candidates   : Return value of build_bridge(),
                       containing two keys: "all_candidates" and "profiles".
                       all_candidates format:
                       {table_name: {target_col_idx: [(source_col, score_dict), ...]}}
    target_instances : Target instance list; each row is a list of values,
                       with column order corresponding to t_global_idx.
    n_source         : Total number of source schema attributes
                       (same as build_bridge's source_number).
    fd_file          : Source FD file path (optional; None to skip FD mapping).
    ind_file         : Source IND file path (optional; None to skip IND mapping).
    cfd_file         : Source CFD file path (optional; None to skip CFD mapping).
    out_fd_path      : Output FD file path (default: target_fds.txt).
    out_ind_path     : Output IND file path (default: target_inds.txt).
    out_cfd_path     : Output CFD file path (default: target_cfds.txt).

    Returns
    -------
    {
      "fds":  List[FD],   # Validated mapped FDs (attribute names are global index strings)
      "inds": List[IND],  # Validated mapped INDs
      "cfds": List[CFD],  # Validated mapped CFDs
    }

    Output file formats (attribute names are global index strings, same as
    dependency_reranker.py input format)
    ----
    FD  file:  5,6 -> 7
    IND file:  5[=6
    CFD file:  (5, 6=1996) => 7
    """
 
    print("=" * 65)
    print("build_target_edge: source dependency -> target dependency mapping and validation")
    print("=" * 65)
 
    # -- Construct target Relation (done once, reusing dependency_reranker.py logic)
    target_relation = build_target_relation(target_instances, n_source)
    target_number = len(target_instances[0]) if target_instances else 0
 
    # -- Use all_candidates to map each source column to its corresponding
    #    target global index candidate list
    all_candidates = all_candidates["all_candidates"]
    all_scores = [[0 for _ in range(target_number)] for _ in range(n_source)]
    # Stores scores for each source column against all target columns;
    # rows represent source columns, columns represent scores for different target columns

    for _, candidates in all_candidates.items():
        for t_idx, cands in candidates.items():
            for s_col, sc in cands:
                s_idx = int(s_col)  
                t_idx = int(t_idx)
                all_scores[s_idx][t_idx] = sc["total"]
    
    # Select global top_k target columns for each source column
    cand_map = get_topk_colidx(all_scores, top_k)
    candidate_map = {k : [int(x) + n_source for x in v] for k, v in cand_map.items()}
  
    print(candidate_map)
    
    # -- Read source dependencies
    fds  = parse_fds(fd_file)   if os.path.exists(fd_file)  else []
    inds = parse_inds(ind_file) if os.path.exists(ind_file) else []
    print(f"Read source dependencies: {len(fds)} FD, {len(inds)} IND")
 
    # -- Enumerate, validate, collect
    mapped_fds  = _collect_mapped_fds(fds, candidate_map, target_relation)
    mapped_inds = _collect_mapped_inds(inds, candidate_map, target_relation)
    print(f"Validation passed: {len(mapped_fds)} FD, {len(mapped_inds)} IND")
 
    # -- Write output files
    _write_fds(mapped_fds, out_fd_path) if mapped_fds else None
    _write_inds(mapped_inds, out_ind_path) if mapped_inds else None
 
    return {"fds": mapped_fds, "inds": mapped_inds}