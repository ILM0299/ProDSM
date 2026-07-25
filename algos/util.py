from typing import Any, Dict, List, Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/mnt/project")

from dependency_validator import (
    FD, CFD, IND,
    Relation
)



#══════════════════════════════════════════════════════════════════════════════
# Parsing functions: read dependencies from file paths; attribute names are
# all source attribute index strings
# ═════════════════════════════════════════════════════════════════════════════

def parse_fds(fd_file: str) -> List[FD]:
    """
    Read a list of FDs from file.

    Format (one per line, # for comments):
        0,1 -> 2
        0 -> 2,3
    Attribute names are source attribute index strings. Both LHS and RHS
    may contain multiple attributes (comma-separated).
    """
    fds: List[FD] = []
    with open(fd_file, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                raise ValueError(f"FD format error (missing ->): {line!r}")
            lhs_part, rhs_part = line.split("->", 1)
            lhs = [a.strip() for a in lhs_part.split(",") if a.strip()]
            rhs = [a.strip() for a in rhs_part.split(",") if a.strip()]
            if not lhs or not rhs:
                raise ValueError(f"FD LHS/RHS cannot be empty: {line!r}")
            fds.append(FD(lhs, rhs))
    return fds


def parse_inds(ind_file: str) -> List[IND]:
    """
    Read a list of INDs from file.

    Format (one per line, # for comments):
        0[=1
        0,1[=2,3
    '[=' separates the dependent side (left) from the referenced side (right),
    matched by position.
    """
    inds: List[IND] = []
    with open(ind_file, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "[=" not in line:
                raise ValueError(f"IND format error (missing [=): {line!r}")
            lhs_part, rhs_part = line.split("[=", 1)
            lhs = [a.strip() for a in lhs_part.split(",") if a.strip()]
            rhs = [a.strip() for a in rhs_part.split(",") if a.strip()]
            if not lhs or not rhs:
                raise ValueError(f"IND LHS/RHS cannot be empty: {line!r}")
            inds.append(IND(lhs, rhs))
    return inds


def parse_cfds(cfd_file: str) -> List[CFD]:
    """
    Read a list of CFDs from file.

    Format (one per line, # for comments):
        (0, 1=1996) => 2
        (0=F, 1=1236) => 2

    Inside parentheses, comma-separated items:
      - Items with '=': pattern constants, formatted as
        attribute_index=constant_value
      - Items without '=': ordinary wildcard LHS attribute indices

    Parsing example:
        (0, 1=1996) => 2
          -> lhs=['0','1'], rhs=['2'], pattern={'1':'1996'}
    """
    cfds: List[CFD] = []
    with open(cfd_file, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" not in line:
                raise ValueError(f"CFD format error (missing =>): {line!r}")
            lhs_block, rhs_part = line.split("=>", 1)
            lhs_block = lhs_block.strip()
            if not (lhs_block.startswith("(") and lhs_block.endswith(")")):
                raise ValueError(f"CFD LHS must be enclosed in parentheses: {line!r}")
            lhs_block = lhs_block[1:-1]

            lhs_attrs: List[str] = []
            pattern: Dict[str, str] = {}
            for item in lhs_block.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    attr, val = item.split("=", 1)
                    attr, val = attr.strip(), val.strip()
                    lhs_attrs.append(attr)
                    pattern[attr] = val       # constant value stored as string
                else:
                    lhs_attrs.append(item)    # wildcard attribute, not added to pattern

            rhs_attrs: List[str] = []
            for item in rhs_part.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    attr, val = item.split("=", 1)
                    attr, val = attr.strip(), val.strip()
                    rhs_attrs.append(attr)
                    pattern[attr] = val      # RHS constant values also stored in pattern
                else:
                    rhs_attrs.append(item)

            if not lhs_attrs or not rhs_attrs:
                raise ValueError(f"CFD LHS/RHS cannot be empty: {line!r}")
            cfds.append(CFD(lhs_attrs, rhs_attrs, pattern))
    return cfds


# ══════════════════════════════════════════════════════════════════════════════
# Construct target Relation (done once)
# ══════════════════════════════════════════════════════════════════════════════

def build_target_relation(
    target_instances: List[List[Any]],
    n_source: int,
) -> Relation:
    """
    Convert target instances into a Relation (list[dict]).

    Keys use the target attribute global index strings directly:
        column k -> key str(n_source + k)

    This Relation is constructed only once for the entire reranking process;
    all validations reuse it.
    """
    relation: Relation = []
    for row in target_instances:
        rec = {str(n_source + k): val for k, val in enumerate(row)}
        relation.append(rec)
    return relation

def get_topk_colidx(
    all_scores: List[List[float]], 
    k: int = 5  # customizable k value
) -> Dict[int, List[int]]:
    """
    Retrieve the top-k highest-scoring col_idx for each row_idx.
    :param all_scores: 2D list, all_scores[row_idx][col_idx] = score
    :param k: number of top scores to select
    :return: {row_idx: [col_idx1, col_idx2, ...], ...}
    """
    topk_result = {}
    
    # Iterate over each target index t_idx
    for row_idx, score_list in enumerate(all_scores):
        # Skip empty score lists
        if not score_list:
            topk_result[row_idx] = []
            continue
        
        # Generate (col_idx, score) pairs
        colidx_score_pairs = [
            (col_idx, score) for col_idx, score in enumerate(score_list)
        ]
        
        # Sort by score in descending order (highest first)
        colidx_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top k col_idx values
        topk_colidx = [col_idx for col_idx, _ in colidx_score_pairs[:k]]
        
        # Store into results
        topk_result[row_idx] = topk_colidx
    
    return topk_result