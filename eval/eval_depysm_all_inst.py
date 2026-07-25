# coding: utf-8
import json
import os

import numpy as np


# ──────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# Per-target evaluation
# ──────────────────────────────────────────────────────────

TOPK_REPORT = [1, 3, 5]   # only these k values are printed / stored

def evaluate_one_target(res_path, config, n_source, inst_suffix):
    """
    Load results for one target + inst_suffix and compute metrics.
    Returns a dict with keys: emb / refine, each containing p@1,3,5 / mrr / recall.
    Returns None if any required file is missing.
    """
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


# ──────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────

METRIC_KEYS = [f"p@{k}" for k in TOPK_REPORT] + ["mrr", "recall"]
METRIC_LABELS = {f"p@{k}": f"P@{k}" for k in TOPK_REPORT}
METRIC_LABELS.update({"mrr": "MRR", "recall": "Recall@GT"})

COL_W = 8   # width per metric value column

def _row(label, emb, refine, label_w):
    vals = "  ".join(
        f"{METRIC_LABELS[m]:>{COL_W}} {emb[m]:.4f}/{refine[m]:.4f}"
        for m in METRIC_KEYS
    )
    return f"  {label:<{label_w}}  {vals}"

def print_target_block(subdir, result):
    label_w = max(len(subdir), 8)
    header_parts = "  ".join(f"{METRIC_LABELS[m]:>{COL_W}} {'Emb/Ref':>11}" for m in METRIC_KEYS)
    print(f"\n  {'':>{label_w}}  {header_parts}")
    print(_row(subdir, result["emb"], result["refine"], label_w))

def print_summary_table(all_results, target_configs, n_target_instances_list):
    """
    Two-section summary table (Emb / Refine) across all n_inst values.
    Rows = n_inst, Columns = each target + Avg.
    One table per metric.
    """
    subdirs  = [c["subdir"] for c in target_configs]
    col_w    = 8
    lbl_w    = 6   # "n_inst" label width

    for section in ("emb", "refine"):
        tag = "Emb (no refinement)" if section == "emb" else "Refine (with dependency rewards)"
        print(f"\n{'=' * 80}")
        print(f"  {tag}")
        print(f"{'=' * 80}")

        # column header
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


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_path = "/home/ouyang/Code/DepySM/data/MusicRecordings/res"
    target_configs = [
        {"subdir": "target1"},   # SongName + Singer + Album (SongName key)
        {"subdir": "target2"},   # SongName + Singer + Album + Style (SongName key)
        {"subdir": "target3"},   # SongName + Duration + Style (SongName key, no Album->Singer FD)
        {"subdir": "target4"},   # All columns (SongName + URL dual key)
        {"subdir": "target5"},   # URL + Album + Singer (URL backup key)
        {"subdir": "target6"},   # Singer + Album + Style (no unique key)
    ]

    n_source = 6
    n_target_instances_list = [1, 3, 5, 8, 10]

    # all_results[n_inst][subdir] = {"emb": {...}, "refine": {...}}
    all_results = {}

    for n_inst in n_target_instances_list:
        inst_suffix = f"inst{n_inst}"
        print(f"\n{'*' * 60}")
        print(f"  n_target_instances = {n_inst}")
        print(f"{'*' * 60}")

        inst_scores = {}
        for config in target_configs:
            res_path = f"{base_path}/{config['subdir']}"
            print(f"\n  --- {config['subdir']} [{inst_suffix}] ---")
            result = evaluate_one_target(res_path, config, n_source, inst_suffix)
            if result is None:
                continue
            inst_scores[config["subdir"]] = result

            # Per-target compact output: Emb/Refine side-by-side
            label_w = 8
            header  = "  ".join(f"{METRIC_LABELS[m]:>{COL_W}} {'Emb / Ref':>11}" for m in METRIC_KEYS)
            print(f"  {'':>{label_w}}  {header}")
            print(_row(config["subdir"], result["emb"], result["refine"], label_w))

        all_results[n_inst] = inst_scores

        # Per-n_inst average
        valid = [c for c in target_configs if c["subdir"] in inst_scores]
        if not valid:
            continue
        avg_emb    = {m: np.mean([inst_scores[c["subdir"]]["emb"][m]    for c in valid]) for m in METRIC_KEYS}
        avg_refine = {m: np.mean([inst_scores[c["subdir"]]["refine"][m] for c in valid]) for m in METRIC_KEYS}
        print(f"\n  {'--- Average across all targets ---':}")
        header = "  ".join(f"{METRIC_LABELS[m]:>{COL_W}} {'Emb / Ref':>11}" for m in METRIC_KEYS)
        print(f"  {'':>8}  {header}")
        print(_row("Avg", avg_emb, avg_refine, 8))

    # Final cross-n_inst summary tables
    print_summary_table(all_results, target_configs, n_target_instances_list)