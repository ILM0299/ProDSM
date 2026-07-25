# coding: utf-8
"""
gen_cfds_from_rules.py
======================
Use the rule expansion logic to regenerate each dataset's [main experiment]
source/cfds.txt (replacing the manually created old version).

Relationship with make_cfd_subsets_by_rule.py
----------------------------------------------
Identical expansion logic, except this script writes the [full] cfds.txt
(= f=1.0, all rules expanded), for the main experiment DepySM_batch_more.py
embedding step to read. The f=1.0 results are line-by-line identical.

Fixed semantics (confirmed with you)
-------------------------------------
1. Each line in cfd_raw.txt = one CFD rule; the full line serves as the rule
   identifier, without parsing special values internally.
2. Rule column numbers = global attribute indices, consistent with source
   headers; no col[3:] slicing is performed.
3. All columns of each rule belong to the same table => column numbers
   uniquely determine the owning table and that table's instance offset.
4. Edge creation: instances satisfying a rule are connected to all LHS
   columns + the RHS column of that rule (including RHS with condition values).
5. Deduplication: the union of (attr, global_inst) across all expanded rules
   is deduplicated (= the expected content of the old cfds.txt).
6. Global instance ID = single-table row index (from 0) + that table's offset;
   offsets accumulate according to your custom table order.

Usage
-----
    cd algos
    python gen_cfds_from_rules.py
This backs up the old source/cfds.txt as cfds.txt.manual.bak, then writes
the new cfds.txt. It also prints the distinct edge count (= cfds.txt line
count) and the maximum instance ID (should be < n_instance).
"""

import csv
import os
import shutil
import sys
from pathlib import Path


# -- Rule parsing (column numbers as plain integers, no slicing) --
def parse_cfd(cfd_str):
    lhs_part, rhs_part = cfd_str.split('=>')
    lhs_part = lhs_part.strip()[1:-1]
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
            rules.append(s)
    return rules


class TableIndex:
    def __init__(self, csv_path, col_lo, col_hi, inst_offset):
        self.col_lo = col_lo
        self.col_hi = col_hi
        self.inst_offset = inst_offset
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            self.header = next(r)
            for row in r:
                self.rows.append(row)
        self.gcol_to_local = {}
        for local_idx, h in enumerate(self.header):
            self.gcol_to_local[int(h.strip())] = local_idx
        self.nrows = len(self.rows)

    def owns(self, gcol):
        return self.col_lo <= gcol <= self.col_hi

    def matching_global_instances(self, lhs_conditions, rhs_condition):
        conds = list(lhs_conditions.items())
        if rhs_condition is not None:
            conds.append(rhs_condition)
        local_conds = []
        for gcol, val in conds:
            if gcol not in self.gcol_to_local:
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
    tabs = []
    for csv_name, (lo, hi), offset in tables_cfg:
        p = Path(source_dir) / csv_name
        if not p.exists():
            raise FileNotFoundError(f"Source table not found: {p}")
        tabs.append(TableIndex(str(p), lo, hi, offset))
    return tabs


def rule_to_edges(rule_str, tables):
    lhs_cols, lhs_conds, rhs_col, rhs_cond = parse_cfd(rule_str)
    all_attr_cols = list(lhs_cols) + [rhs_col]
    home = None
    for t in tables:
        if t.owns(all_attr_cols[0]):
            home = t
            break
    if home is None:
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


def fill_offsets(source_dir, tables_cfg):
    """Auto-accumulate offsets where offset=None based on preceding table row counts;
    return the filled configuration + total instance count (n_instance)."""
    filled = []
    running = 0
    for csv_name, colrange, offset in tables_cfg:
        p = Path(source_dir) / csv_name
        with open(p, encoding="utf-8") as f:
            nrows = sum(1 for _ in f) - 1
        use_off = running if offset is None else offset
        filled.append((csv_name, colrange, use_off))
        running = max(running, use_off + nrows)
        print(f"    [cfg] {csv_name}: cols{colrange} offset={use_off} rows={nrows}")
    return filled, running


def generate_cfds(data_dir, dataset, tables_cfg, backup=True):
    src = Path(data_dir) / dataset / "source"
    cfd_raw = src / "cfd_raw.txt"
    cfds_out = src / "cfds.txt"
    if not cfd_raw.exists():
        print(f"  [skip] {dataset}: {cfd_raw} not found")
        return None

    print(f"\n  === {dataset} ===")
    filled_cfg, n_instance = fill_offsets(src, tables_cfg)
    rules = load_rules(cfd_raw)
    print(f"    CFD rule count N = {len(rules)}, inferred n_instance (total instances) = {n_instance}")

    tables = build_table_router(src, filled_cfg)
    edges = set()
    for r in rules:
        edges |= rule_to_edges(r, tables)

    max_v = max((v for _u, v in edges), default=-1)
    print(f"    After expansion: distinct (attr,inst) edge count = {len(edges)}, max instance ID = {max_v} "
          f"(should be < n_instance={n_instance})")
    if max_v >= n_instance:
        print(f"    [warn] Max instance ID {max_v} >= n_instance {n_instance}! "
              f"Offset or table order may be incorrect, please check tables_cfg.")

    # Back up the old manually created cfds.txt
    if backup and cfds_out.exists():
        bak = src / "cfds.txt.manual.bak"
        if not bak.exists():
            shutil.copy2(cfds_out, bak)
            print(f"    Old cfds.txt backed up -> {bak.name}")
        else:
            print(f"    Backup already exists ({bak.name}), not overwriting backup")

    with open(cfds_out, "w", encoding="utf-8") as f:
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")
    print(f"    New cfds.txt written ({len(edges)} lines)")
    return n_instance


if __name__ == "__main__":
    DATA_DIR = "../data"

    # ============================================================
    # Table configuration per dataset: (csv filename, global attribute column range [lo,hi] inclusive, instance offset)
    # Column range = first to last column number in that table's header (inclusive on both ends).
    # offset None = auto-accumulate from preceding table row counts; the table [order] must match your original global numbering order.
    # Please fill in according to the actual table structure of your three datasets.
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
    if not base.exists():
        print(f"[FATAL] {os.path.abspath(DATA_DIR)} does not exist; please run from algos/.")
        sys.exit(1)

    n_inst_map = {}
    for ds, cfg in DATASETS.items():
        if not (base / ds / "source").exists():
            print(f"[skip] {ds}: no source directory")
            continue
        ni = generate_cfds(DATA_DIR, ds, cfg)
        if ni is not None:
            n_inst_map[ds] = ni

    print("\n[done] cfds.txt regeneration complete. Inferred n_instance per dataset:")
    for ds, ni in n_inst_map.items():
        print(f"    {ds}: n_instance = {ni}")
    print("Next, re-run the main experiment with the updated DepySM_batch_more.py (it will read the new cfds.txt).")