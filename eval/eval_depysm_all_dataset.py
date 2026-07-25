# coding: utf-8
"""
eval_depysm_all_dataset.py
==========================
Following eval_depysm_all_inst.py, evaluate all three datasets (MusicRecordings / Movielens / TPCH) at once.

Only differences from the original
-----------------------------------
1. An outer dataset loop is added.
2. n_source is no longer hardcoded to 6, but auto-inferred for each dataset from the sum of
   column counts in source/*.csv (three datasets have different column counts: 6 / 13 / 61;
   hardcoding would cause incorrect metrics for the latter two datasets).
3. target subdirectories are auto-discovered (target* directories under target/ containing target.csv).
Metric definitions, filename conventions, and print format are verbatim identical to the original.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np


# ----------------------------------------------------------
# Metric helpers (verbatim from original)
# ----------------------------------------------------------

def precision_topk(sim_matrix, gt_has_match, gt_indices, k):
    topk_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    correct = sum(
        1 for i in range(sim_matrix.shape[0])
        if gt_has_match[i] and any(idx in gt_indices[i] for idx in topk_indices[i])
    )
    return correct, float(correct) / float(gt_has_match.sum())


def compute_mrr(sim_matrix, gt_has_match, gt_indices):
    rr_sum, valid_count = 0.0, 0
    for i in range(sim_matrix.shape[0]):
        if not gt_has_match[i]:
            continue
        for rank, col in enumerate(np.argsort(sim_matrix[i])[::-1], start=1):
            if col in gt_indices[i]:
                rr_sum += 1.0 / rank
                break
        valid_count += 1
    return rr_sum / valid_count if valid_count > 0 else 0.0


def compute_recall_at_gt(sim_matrix, gt_has_match, gt_indices):
    G = {
        (i, col)
        for i in range(sim_matrix.shape[0])
        if gt_has_match[i]
        for col in gt_indices[i]
    }
    k = len(G)
    if k == 0:
        return 0.0
    rows, cols = np.unravel_index(np.argsort(sim_matrix, axis=None)[::-1], sim_matrix.shape)
    top_k_pairs = set(zip(rows[:k].tolist(), cols[:k].tolist()))
    return len(G & top_k_pairs) / k


def gt_json_to_matrix(gt_json, n_source_cols):
    gt_matrix = np.zeros((len(gt_json), n_source_cols), dtype=int)
    for i, (_, src_indices) in enumerate(gt_json.items()):
        for src_idx in src_indices:
            gt_matrix[i, src_idx] = 1
    return gt_matrix


# ----------------------------------------------------------
# Per-target evaluation (verbatim from original)
# ----------------------------------------------------------

TOPK_REPORT = [1, 3, 5]

def evaluate_one_target(res_path, config, n_source, inst_suffix):
    sim_path     = f"{res_path}/{inst_suffix}/sim_matrix.json"
    gt_path      = f"{res_path}/ground_true.json"
    updated_path = f"{res_path}/{inst_suffix}/updated_sim_matrix_iter5_p2_k13_d4_c15.json"

    for p in [sim_path, gt_path, updated_path]:
        if not os.path.exists(p):
            print(f"  [skip] missing: {p}")
            return None

    gt_json     = json.load(open(gt_path))
    ground_true = gt_json_to_matrix(gt_json, n_source)
    m           = ground_true.shape[0]
    gt_has_match = ground_true.max(axis=1) == 1
    gt_indices  = [set(np.where(ground_true[i] == 1)[0]) for i in range(m)]

    sim     = np.array(json.load(open(sim_path)))
    updated = np.array(json.load(open(updated_path)))

    def _metrics(mat):
        p = {k: precision_topk(mat, gt_has_match, gt_indices, k)[1] for k in TOPK_REPORT}
        return {
            **{f"p@{k}": p[k] for k in TOPK_REPORT},
            "mrr":    compute_mrr(mat, gt_has_match, gt_indices),
            "recall": compute_recall_at_gt(mat, gt_has_match, gt_indices),
        }

    return {"emb": _metrics(sim), "refine": _metrics(updated)}


# ----------------------------------------------------------
# Printing helpers (verbatim from original)
# ----------------------------------------------------------

METRIC_KEYS = [f"p@{k}" for k in TOPK_REPORT] + ["mrr", "recall"]
METRIC_LABELS = {f"p@{k}": f"P@{k}" for k in TOPK_REPORT}
METRIC_LABELS.update({"mrr": "MRR", "recall": "Recall@GT"})

COL_W = 8
CELL_W = 17   # width per "Emb/Ref" numeric cell

def _metric_header(label_w):
    """One header row: list 5 metric names (each cell below shows Emb/Ref values)."""
    cells = "  ".join(f"{METRIC_LABELS[m] + ' (Emb/Ref)':>{CELL_W}}" for m in METRIC_KEYS)
    return f"  {'':<{label_w}}  {cells}"

def _row(label, emb, refine, label_w):
    """One data row: each cell is 'emb/ref', aligned with header, no repeated metric names."""
    cells = "  ".join(f"{emb[m]:.4f}/{refine[m]:.4f}".rjust(CELL_W) for m in METRIC_KEYS)
    return f"  {label:<{label_w}}  {cells}"

def print_summary_table(all_results, target_configs, n_target_instances_list, ds_name):
    subdirs  = [c["subdir"] for c in target_configs]
    col_w    = 8
    lbl_w    = 6

    for section in ("emb", "refine"):
        tag = "Emb (no refinement)" if section == "emb" else "Refine (with dependency rewards)"
        print(f"\n{'=' * 80}")
        print(f"  [{ds_name}] {tag}")
        print(f"{'=' * 80}")

        hdr = f"  {'n_inst':<{lbl_w}}  " + "  ".join(f"{s:>{col_w}}" for s in subdirs) + f"  {'Avg':>{col_w}}"
        print(hdr)
        print("-" * len(hdr))

        for metric in METRIC_KEYS:
            print(f"  [{METRIC_LABELS[metric]}]")
            for n_inst in n_target_instances_list:
                inst_scores = all_results.get(n_inst, {})
                vals = []
                for sd in subdirs:
                    r = inst_scores.get(sd)
                    vals.append(r[section][metric] if r else None)
                valid = [v for v in vals if v is not None]
                avg   = np.mean(valid) if valid else None

                def fmt(v):
                    return f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}}"

                row = f"  {n_inst:<{lbl_w}}  " + "  ".join(fmt(v) for v in vals)
                row += f"  {fmt(avg)}"
                print(row)


# ----------------------------------------------------------
# Multi-dataset auto-inference (consistent with DepySM_batch_more.py)
# ----------------------------------------------------------

def auto_n_source(source_dir):
    """n_source = sum of column counts across all CSV files in the source directory."""
    total = 0
    for p in sorted(Path(source_dir).glob("*.csv")):
        with open(p, encoding="utf-8") as f:
            total += len(next(csv.reader(f)))
    return total


def auto_target_subdirs(res_dir):
    """Discover target* subdirectories under res/ that contain ground_true.json."""
    base = Path(res_dir)
    if not base.exists():
        return []
    subs = sorted((p for p in base.iterdir() if p.is_dir() and (p / "ground_true.json").exists()),
                  key=lambda p: (len(p.name), p.name))
    return [{"subdir": p.name} for p in subs]


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_ROOT = "/home/ouyang/Code/DepySM/data"   # or change to relative path "../data"
    n_target_instances_list = [2, 3, 5, 8, 10]   # must match DepySM_batch_more.py (no 1, has 2)

    # Per dataset: n_source left as None for auto-inference (recommended); target_configs left as None for auto-discovery
    DATASETS = [
        {"name": "MusicRecordings", "n_source": 6,    "target_configs": None},
        {"name": "Movielens",       "n_source": None, "target_configs": None},
        {"name": "TPCH",            "n_source": None, "target_configs": None},
    ]

    # grand[ds][n_inst][subdir] = {"emb":..,"refine":..}
    grand = {}

    for ds in DATASETS:
        ds_name   = ds["name"]
        data_dir  = f"{DATA_ROOT}/{ds_name}"
        base_path = f"{data_dir}/res"
        src_dir   = f"{data_dir}/source"

        if not os.path.isdir(base_path):
            print(f"\n[skip] {ds_name}: cannot find {base_path}")
            continue

        n_source = ds["n_source"] if ds["n_source"] is not None else auto_n_source(src_dir)
        target_configs = ds["target_configs"] if ds["target_configs"] is not None else auto_target_subdirs(base_path)
        if not target_configs:
            print(f"\n[skip] {ds_name}: no target* directory with ground_true.json found under {base_path}")
            continue

        print(f"\n\n{'#' * 80}")
        print(f"#  Dataset {ds_name}   (n_source={n_source},  "
              f"targets={[c['subdir'] for c in target_configs]})")
        print(f"{'#' * 80}")

        all_results = {}
        for n_inst in n_target_instances_list:
            inst_suffix = f"inst{n_inst}"
            print(f"\n{'*' * 60}")
            print(f"  [{ds_name}] n_target_instances = {n_inst}")
            print(f"{'*' * 60}")

            inst_scores = {}
            for config in target_configs:
                res_path = f"{base_path}/{config['subdir']}"
                print(f"\n  --- [{ds_name}] {config['subdir']} [{inst_suffix}] ---")
                result = evaluate_one_target(res_path, config, n_source, inst_suffix)
                if result is None:
                    continue
                inst_scores[config["subdir"]] = result

                label_w = 8
                print(_metric_header(label_w))
                print(_row(config["subdir"], result["emb"], result["refine"], label_w))

            all_results[n_inst] = inst_scores

            valid = [c for c in target_configs if c["subdir"] in inst_scores]
            if not valid:
                continue
            avg_emb    = {m: np.mean([inst_scores[c["subdir"]]["emb"][m]    for c in valid]) for m in METRIC_KEYS}
            avg_refine = {m: np.mean([inst_scores[c["subdir"]]["refine"][m] for c in valid]) for m in METRIC_KEYS}
            print(f"\n  >>> [{ds_name}] inst{n_inst} average across {len(valid)} targets (Emb/Ref):")
            print(_metric_header(8))
            print(_row("Avg", avg_emb, avg_refine, 8))

        print_summary_table(all_results, target_configs, n_target_instances_list, ds_name)
        grand[ds_name] = all_results
        print(f"\n~ Dataset {ds_name} evaluation complete\n")

    # -- Compute cross-target averages per (dataset, n_inst) (both Emb and Ref) --
    # overview[ds][n_inst] = {"emb": {...}, "refine": {...}, "n_targets": int}
    overview = {}
    for ds_name, all_results in grand.items():
        overview[ds_name] = {}
        for n_inst in n_target_instances_list:
            inst_scores = all_results.get(n_inst, {})
            valid = list(inst_scores.keys())
            if not valid:
                continue
            overview[ds_name][n_inst] = {
                "emb":    {m: float(np.mean([inst_scores[sd]["emb"][m]    for sd in valid])) for m in METRIC_KEYS},
                "refine": {m: float(np.mean([inst_scores[sd]["refine"][m] for sd in valid])) for m in METRIC_KEYS},
                "n_targets": len(valid),
            }

    # -- Cross-dataset overview: Avg per dataset per n_inst, both Emb and Ref displayed --
    print(f"\n\n{'#' * 90}")
    print(f"#  Cross-Dataset Overview (averaged across targets, each cell Emb/Ref)")
    print(f"{'#' * 90}")
    hdr = (f"  {'dataset':<16}  {'n_inst':>6}  "
           + "  ".join(f"{METRIC_LABELS[m] + ' (Emb/Ref)':>{CELL_W}}" for m in METRIC_KEYS))
    print(hdr); print("-" * len(hdr))
    for ds_name in overview:
        for n_inst in n_target_instances_list:
            cell = overview[ds_name].get(n_inst)
            if not cell:
                continue
            e, r = cell["emb"], cell["refine"]
            vals = "  ".join(f"{e[m]:.4f}/{r[m]:.4f}".rjust(CELL_W) for m in METRIC_KEYS)
            print(f"  {ds_name:<16}  {n_inst:>6}  {vals}")

    # -- Per-dataset "Final Avg" summary (default n_inst=5, corresponding to one column in the paper table) --
    PAPER_N_INST = 5   # instance count for paper main table; change as needed
    print(f"\n\n{'#' * 90}")
    print(f"#  Paper table: Final Avg per dataset at n_inst={PAPER_N_INST} (one row per dataset)")
    print(f"{'#' * 90}")
    hdr2 = (f"  {'dataset':<16}  {'variant':<7}  "
            + "  ".join(f"{METRIC_LABELS[m]:>9}" for m in METRIC_KEYS))
    print(hdr2); print("-" * len(hdr2))
    for ds_name in overview:
        cell = overview[ds_name].get(PAPER_N_INST)
        if not cell:
            print(f"  {ds_name:<16}  (no n_inst={PAPER_N_INST} results)")
            continue
        for variant, key in [("Emb", "emb"), ("Ref", "refine")]:
            mvals = cell[key]
            print(f"  {ds_name:<16}  {variant:<7}  "
                  + "  ".join(f"{mvals[m]:>9.4f}" for m in METRIC_KEYS))

    # -- Export CSV / JSON (for easy table filling) --
    out_dir = "eval_out"
    os.makedirs(out_dir, exist_ok=True)

    # JSON: complete per-target results + cross-target overview
    json_payload = {
        "metric_keys": METRIC_KEYS,
        "per_target": grand,          # grand[ds][n_inst][subdir] = {"emb":..,"refine":..}
        "overview":   overview,       # overview[ds][n_inst] = {"emb":..,"refine":..,"n_targets":..}
    }
    json_path = os.path.join(out_dir, "eval_all_dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # CSV: long format, one row = (dataset, n_inst, variant, all metrics), directly readable by Excel/pandas
    csv_path = os.path.join(out_dir, "eval_all_dataset.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_inst", "variant"] + [METRIC_LABELS[m] for m in METRIC_KEYS] + ["n_targets"])
        for ds_name in overview:
            for n_inst in n_target_instances_list:
                cell = overview[ds_name].get(n_inst)
                if not cell:
                    continue
                for variant, key in [("Emb", "emb"), ("Ref", "refine")]:
                    w.writerow([ds_name, n_inst, variant]
                               + [f"{cell[key][m]:.4f}" for m in METRIC_KEYS]
                               + [cell["n_targets"]])

    print(f"\n[export] Full results -> {json_path}")
    print(f"[export] Overview long table -> {csv_path}  (columns: dataset,n_inst,variant,P@1,P@3,P@5,MRR,Recall@GT,n_targets)")