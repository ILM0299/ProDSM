# coding: utf-8
"""
make_cfd_subsets_by_rule.py
===========================
V5 CFD dose-response ablation: sample by **CFD rules** (advisor: "random subset of the discovered CFDs").

Difference from the old make_cfd_subsets.py
-------------------------------------------
The old version sampled by "instance nodes v" (incorrect: the x-axis semantics is evidence
quantity, not rule count).
This version samples by **each line in cfd_raw.txt = one CFD rule**, N = total rule count
(Movielens=251).

Fixed semantics (all confirmed with you)
-----------------------------------------
1. Each line in cfd_raw.txt = one rule; the full line string serves as the rule ID, without
   parsing special values internally (to avoid quote/space errors).
2. Column numbers in rules = global attribute indices, consistent with source headers;
   **no col[3:] slicing**.
3. All columns of each rule belong to the same table => column numbers uniquely determine
   the owning table and that table's instance offset.
4. Edge creation: instances satisfying a rule are connected to all LHS columns + the RHS
   column (including RHS with condition values, following cfd_trans.py).
5. Sampling: seed=42 shuffles rules once; the first floor(f*N) rules form the **nested**
   subset for fraction f (0.25 subset 0.5 subset 0.75 subset 1.0).
6. Deduplication: within the **kept rule set**, take the union of (attr, global_inst) and
   deduplicate. f=1.0 is equivalent to global dedup => should be line-by-line equal to the
   existing cfds.txt.
7. Global instance ID = single-table row index (from 0) + that table's offset; offsets
   accumulate according to your custom table order.

Each dataset requires a TABLES configuration (see DATASETS at the bottom):
    (csv filename, global attribute column range [lo,hi] inclusive, that table's global instance offset)

Usage
-----
    cd algos
    python make_cfd_subsets_by_rule.py
Output: data/<DS>/source/cfds_f{100,075,050,025}.txt
Self-consistency check (local, no need to upload large files):
    sort data/<DS>/source/cfds_f100.txt | uniq > /tmp/a
    sort data/<DS>/source/cfds.txt      | uniq > /tmp/b
    diff /tmp/a /tmp/b && echo "f=1.0 consistent with existing cfds.txt"
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
FRACTIONS   = [1.00, 0.75, 0.50, 0.25]
FRAC_TAG    = {1.00: "100", 0.75: "075", 0.50: "050", 0.25: "025", 0.00: "000"}


# -- Rule parsing (following cfd_trans.parse_cfd splitting, but column numbers as plain integers, no slicing) --
def parse_cfd(cfd_str):
    """
    Return (lhs_columns:list[int], lhs_conditions:dict[int->str],
            rhs_column:int, rhs_condition:(int,str)|None)
    Column numbers are always int. Special values (with spaces/quotes) are preserved as-is as string values.
    """
    lhs_part, rhs_part = cfd_str.split('=>')
    lhs_part = lhs_part.strip()[1:-1]   # remove outer parentheses of LHS
    rhs_part = rhs_part.strip()

    lhs_conditions = {}
    lhs_columns = []
    for elem in [e.strip() for e in lhs_part.split(',')]:
        if elem == "":
            continue
        if '=' in elem:
            col, val = elem.split('=', 1)
            col = int(col.strip())
            lhs_conditions[col] = val.strip()
            lhs_columns.append(col)
        else:
            lhs_columns.append(int(elem.strip()))

    rhs_condition = None
    if '=' in rhs_part:
        col, val = rhs_part.split('=', 1)
        rhs_column = int(col.strip())
        rhs_condition = (rhs_column, val.strip())
    else:
        rhs_column = int(rhs_part.strip())
    return lhs_columns, lhs_conditions, rhs_column, rhs_condition


def load_rules(cfd_raw_path):
    """Read cfd_raw.txt, one rule per line. Following the filtering criteria from cfd_merge.is_valid_cfd."""
    rules = []
    with open(cfd_raw_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            t = s.strip()
            if not t or not t.startswith("(") or t.count("=") < 2 or "=>" not in t:
                continue
            lhs_p, rhs_p = t.split("=>", 1)
            if not lhs_p.strip() or not rhs_p.strip():
                continue
            rules.append(s)   # preserve original line (with original spacing) as rule ID
    return rules


class TableIndex:
    """A single table: caches data rows, provides (global column number, value) condition matching, and outputs global instance IDs."""
    def __init__(self, csv_path, col_lo, col_hi, inst_offset):
        self.col_lo = col_lo
        self.col_hi = col_hi
        self.inst_offset = inst_offset
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            self.header = next(r)            # header is global column number strings, e.g. ['9','10','11','12']
            for row in r:
                self.rows.append(row)
        # global column number -> local position within this table
        self.gcol_to_local = {}
        for local_idx, h in enumerate(self.header):
            self.gcol_to_local[int(h.strip())] = local_idx

    def owns(self, gcol):
        return self.col_lo <= gcol <= self.col_hi

    def matching_global_instances(self, lhs_conditions, rhs_condition):
        """Return list of global instance IDs satisfying the conditions (single-table row index + offset)."""
        conds = list(lhs_conditions.items())
        if rhs_condition is not None:
            conds.append(rhs_condition)
        # Pre-parse into (local column position, expected value)
        local_conds = []
        for gcol, val in conds:
            if gcol not in self.gcol_to_local:
                # Rule column not in this table (should not happen since each rule is single-table); treat as no match
                return []
            local_conds.append((self.gcol_to_local[gcol], val))

        out = []
        for row_idx, row in enumerate(self.rows):
            ok = True
            for lpos, val in local_conds:
                if lpos >= len(row) or row[lpos] != val:
                    ok = False
                    break
            if ok:
                out.append(row_idx + self.inst_offset)
        return out


def build_table_router(source_dir, tables_cfg):
    """tables_cfg: [(csv_name, (lo,hi), offset)] -> [TableIndex]"""
    tabs = []
    for csv_name, (lo, hi), offset in tables_cfg:
        p = Path(source_dir) / csv_name
        if not p.exists():
            raise FileNotFoundError(f"Source table not found: {p}")
        tabs.append(TableIndex(str(p), lo, hi, offset))
    return tabs


def rule_to_edges(rule_str, tables):
    """
    One rule -> the set of (attr_global, global_inst) edges it generates.
    All LHS columns + RHS column are connected to every satisfying instance (following cfd_trans.generate_output).
    """
    lhs_cols, lhs_conds, rhs_col, rhs_cond = parse_cfd(rule_str)
    all_attr_cols = list(lhs_cols) + [rhs_col]

    # Determine the owning table from any column of the rule (each rule is single-table)
    home = None
    for t in tables:
        if t.owns(all_attr_cols[0]):
            home = t
            break
    if home is None:
        # Fallback: check which table's header contains the column number
        for t in tables:
            if all_attr_cols[0] in t.gcol_to_local:
                home = t
                break
    if home is None:
        return set()

    insts = home.matching_global_instances(lhs_conds, rhs_cond)
    edges = set()
    for a in all_attr_cols:
        for v in insts:
            edges.add((a, v))
    return edges


def nested_rule_subsets(n_rules, fractions, seed=RANDOM_SEED):
    """Return {fraction: set of kept rule indices}, nested."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rules)
    out = {}
    for f in fractions:
        k = int(np.floor(f * n_rules + 1e-9))
        out[f] = set(perm[:k].tolist())
    return out


def process_dataset(data_dir, dataset, tables_cfg, fractions=FRACTIONS, also_zero=False):
    src = Path(data_dir) / dataset / "source"
    cfd_raw = src / "cfd_raw.txt"
    if not cfd_raw.exists():
        print(f"  [skip] {dataset}: {cfd_raw} not found")
        return None

    rules = load_rules(cfd_raw)
    N = len(rules)
    print(f"\n  {dataset}: CFD rule count N = {N}")

    tables = build_table_router(src, tables_cfg)

    # Precompute edge set per rule (computed once, reused for all fractions)
    rule_edges = [rule_to_edges(r, tables) for r in rules]
    total_edges = set().union(*rule_edges) if rule_edges else set()
    print(f"    Distinct (attr,inst) edge count after expanding all rules = {len(total_edges)}  "
          f"(should equal the deduplicated line count of existing cfds.txt)")

    subsets = nested_rule_subsets(N, fractions)
    # Nesting check
    ordered = sorted(fractions)
    for a, b in zip(ordered, ordered[1:]):
        assert subsets[a] <= subsets[b], f"Nesting violated: f={a} is not a subset of f={b}"

    for f in fractions:
        kept = subsets[f]
        edges = set()
        for ridx in kept:
            edges |= rule_edges[ridx]
        tag = FRAC_TAG[f]
        out_path = src / f"cfds_f{tag}.txt"
        with open(out_path, "w", encoding="utf-8") as fout:
            for u, v in sorted(edges):
                fout.write(f"{u} {v}\n")
        print(f"      f={f:.2f}  rules kept {len(kept):>4}/{N}  edges {len(edges):>8}  -> {out_path.name}")

    if also_zero:
        out_path = src / "cfds_f000.txt"
        open(out_path, "w").close()
        print(f"      f=0.00  rules kept 0/{N}  edges 0  -> {out_path.name}  [V5 degenerate point, N/A]")
    return N


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = "../data"

    # ============================================================
    # Table configuration per dataset: (csv filename, global attribute column range [lo,hi] inclusive, that table's global instance offset)
    # Offsets accumulate according to your custom table order: first table offset=0, second=first table's row count, and so on.
    # Must match the actual table order and row counts used when generating cfds.txt, otherwise f=1.0 will not match.
    # The Movielens config below is a placeholder example (column ranges based on your pasted rules: movies~0-4, users~5-8, ratings~9-12);
    # please replace offsets with actual row counts, and confirm column ranges and table order.
    # ============================================================
    DATASETS = {
        "Movielens": [
            # (csv,           col range,  inst offset)  <-- fill in your custom order
            ("movies.csv",   (0, 3),    0),
            ("users.csv",    (4, 8),    None),   # None = auto-accumulate from preceding table row counts (see auto-fill below)
            ("ratings.csv",  (9, 12),   None),
        ],
        "MusicRecordings": [ ("MusicRecording.csv", (0, 5), 0)],
        "TPCH": [
            ("region.csv",   (0, 2),    0),
            ("nation.csv",   (3, 6),    None),
            ("supplier.csv",   (7, 13),    None),
            ("customer.csv",   (14, 21),    None),
            ("part.csv",   (22, 30),    None),
            ("partsupp.csv",   (31, 35),    None),
            ("orders.csv",   (36, 44),    None),
            ("lineitem.csv",   (45, 60),    None),
        ],
    }

    base = Path(DATA_DIR)
    for ds, tables_cfg in DATASETS.items():
        src = base / ds / "source"
        if not src.exists():
            print(f"[skip] {ds}: no {src}")
            continue

        # Auto-accumulate offsets: fill None offsets by accumulating row counts from preceding tables (in config order)
        filled = []
        running = 0
        for csv_name, colrange, offset in tables_cfg:
            p = src / csv_name
            if not p.exists():
                print(f"[skip] {ds}: {p} not found")
                filled = None
                break
            with open(p, encoding="utf-8") as f:
                nrows = sum(1 for _ in f) - 1   # subtract header row
            use_off = running if offset is None else offset
            filled.append((csv_name, colrange, use_off))
            running = use_off + nrows
            print(f"[cfg] {ds}/{csv_name}: cols{colrange} offset={use_off} rows={nrows}")
        if filled is None:
            continue

        process_dataset(DATA_DIR, ds, filled, also_zero=True)

    print("\n[done] Rule-sampled cfds_f*.txt generated. Please run the f=1.0 self-consistency check first (see header comments).")