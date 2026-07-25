# coding: utf-8
"""
eval_phi_only.py
================
phi-only control experiment: use the four-way compatibility score phi of bridge edges
directly as the final similarity matrix, bypassing graph / embedding / ADVR,
and report P@1/P@3/P@5/MRR/Recall@GT.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from column_profiler import load_or_build_profiles
from bridge_builder import score_column_mapping, DEFAULT_WEIGHTS


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


TOPK_REPORT  = [1, 3, 5]
METRIC_KEYS  = [f"p@{k}" for k in TOPK_REPORT] + ["mrr", "recall"]
METRIC_LABELS = {f"p@{k}": f"P@{k}" for k in TOPK_REPORT}
METRIC_LABELS.update({"mrr": "MRR", "recall": "Recall@GT"})


def load_target_instances(target_csv, n_target_instances, random_seed=42):
    rows = []
    with open(target_csv, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(list(row))

    if n_target_instances is not None and n_target_instances < len(rows):
        rng = np.random.default_rng(seed=random_seed)
        idx = sorted(rng.choice(len(rows), size=n_target_instances, replace=False).tolist())
        rows = [rows[i] for i in idx]
    return rows


def compute_phi_matrices(data_dir, target_instances, force_rebuild_profiles=False):
    import random
    random.seed(42)
    np.random.seed(42)

    all_profiles = load_or_build_profiles(
        Path(data_dir) / "source", force_rebuild=force_rebuild_profiles
    )

    m = len(target_instances[0])
    n_src = 1 + max(int(c) for tp in all_profiles.values() for c in tp.keys())

    keys = ["total", "subset", "type", "dist", "semantic"]
    mats = {k: np.zeros((m, n_src), dtype=float) for k in keys}

    for t_idx in range(m):
        t_vals = [row[t_idx] for row in target_instances]
        for _table, table_profiles in all_profiles.items():
            for s_col, prof in table_profiles.items():
                sc = score_column_mapping(t_vals, prof, DEFAULT_WEIGHTS, verbose=False)
                s_idx = int(s_col)
                for k in keys:
                    mats[k][t_idx, s_idx] = sc[k]
    return mats, n_src


def signal_only_matrix(mats, signal):
    return mats[signal]


def metrics_deterministic(mat, gt_has_match, gt_indices):
    out = {f"p@{k}": precision_topk(mat, gt_has_match, gt_indices, k)[1] for k in TOPK_REPORT}
    out["mrr"]    = compute_mrr(mat, gt_has_match, gt_indices)
    out["recall"] = compute_recall_at_gt(mat, gt_has_match, gt_indices)
    return out


def metrics_tiebreak(mat, gt_has_match, gt_indices, n_perm=200, seed=42, jitter=1e-9):
    rng = np.random.default_rng(seed)
    acc = {f"p@{k}": 0.0 for k in TOPK_REPORT}
    acc["mrr"] = 0.0
    acc["recall"] = 0.0
    for _ in range(n_perm):
        j = mat + rng.uniform(0.0, jitter, size=mat.shape)
        for k in TOPK_REPORT:
            acc[f"p@{k}"] += precision_topk(j, gt_has_match, gt_indices, k)[1]
        acc["mrr"]    += compute_mrr(j, gt_has_match, gt_indices)
        acc["recall"] += compute_recall_at_gt(j, gt_has_match, gt_indices)
    return {k: v / n_perm for k, v in acc.items()}


def analytic_random(gt_has_match, gt_indices, n_src, m_rows):
    rows = [i for i in range(len(gt_has_match)) if gt_has_match[i]]
    out = {}

    for k in TOPK_REPORT:
        vals = []
        for i in rows:
            g = len(gt_indices[i]); N = n_src
            p = 1.0 - math.comb(N - g, k) / math.comb(N, k)
            vals.append(p)
        out[f"p@{k}"] = float(np.mean(vals)) if vals else 0.0

    mrr_vals = []
    for i in rows:
        g = len(gt_indices[i]); N = n_src
        e = 0.0
        denom = math.comb(N, g)
        for r in range(1, N - g + 2):
            e += (1.0 / r) * math.comb(N - r, g - 1) / denom
        mrr_vals.append(e)
    out["mrr"] = float(np.mean(mrr_vals)) if mrr_vals else 0.0

    G = sum(len(gt_indices[i]) for i in rows)
    out["recall"] = G / (m_rows * n_src) if m_rows * n_src > 0 else 0.0
    return out


def evaluate_one(data_dir, dataset, target_subdir, n_inst,
                 force_rebuild_profiles=False, random_seed=42,
                 n_perm=200, write_cache=True):
    target_csv = f"{data_dir}/{dataset}/target/{target_subdir}/target.csv"
    gt_path    = f"{data_dir}/{dataset}/res/{target_subdir}/ground_true.json"
    inst_suffix = f"inst{n_inst}"
    cache_dir   = f"{data_dir}/{dataset}/target/{target_subdir}/{inst_suffix}"
    phi_cache   = f"{cache_dir}/phi_matrix.json"
    sig_cache   = f"{cache_dir}/phi_signals.json"

    if not os.path.exists(target_csv):
        print(f"  [skip] missing target.csv: {target_csv}")
        return None
    if not os.path.exists(gt_path):
        print(f"  [skip] missing ground_true.json: {gt_path}")
        return None

    target_instances = load_target_instances(target_csv, n_inst, random_seed)
    if not target_instances:
        print(f"  [skip] empty target instances: {target_csv}")
        return None

    if os.path.exists(sig_cache):
        sig = json.load(open(sig_cache))
        mats = {k: np.array(v, dtype=float) for k, v in sig.items()}
        n_src = mats["total"].shape[1]
    else:
        mats, n_src = compute_phi_matrices(
            f"{data_dir}/{dataset}", target_instances, force_rebuild_profiles
        )
        if write_cache:
            os.makedirs(cache_dir, exist_ok=True)
            json.dump(mats["total"].tolist(), open(phi_cache, "w"))
            json.dump({k: v.tolist() for k, v in mats.items()}, open(sig_cache, "w"))

    gt_json     = json.load(open(gt_path))
    ground_true = gt_json_to_matrix(gt_json, n_src)
    m           = ground_true.shape[0]
    gt_has_match = ground_true.max(axis=1) == 1
    gt_indices  = [set(np.where(ground_true[i] == 1)[0]) for i in range(m)]

    if m != mats["total"].shape[0]:
        print(f"  [warn] GT rows ({m}) != phi rows ({mats['total'].shape[0]}) "
              f"for {dataset}/{target_subdir} -- check that target column count aligns with ground_true")

    result = {
        "phi":     metrics_tiebreak(mats["total"], gt_has_match, gt_indices, n_perm, random_seed),
        "phi_det": metrics_deterministic(mats["total"], gt_has_match, gt_indices),
        "phi_s1":  metrics_tiebreak(mats["subset"], gt_has_match, gt_indices, n_perm, random_seed),
        "phi_s2":  metrics_tiebreak(mats["type"],   gt_has_match, gt_indices, n_perm, random_seed),
        "phi_s3":  metrics_tiebreak(mats["dist"],   gt_has_match, gt_indices, n_perm, random_seed),
        "phi_s4":  metrics_tiebreak(mats["semantic"], gt_has_match, gt_indices, n_perm, random_seed),
        "random":  analytic_random(gt_has_match, gt_indices, n_src, m),
    }
    return result


def _avg(dicts):
    if not dicts:
        return {k: float("nan") for k in METRIC_KEYS}
    return {k: float(np.mean([d[k] for d in dicts])) for k in METRIC_KEYS}


def print_variant_row(label, mdict):
    cells = "  ".join(f"{METRIC_LABELS[k]} {mdict[k]:.4f}" for k in METRIC_KEYS)
    print(f"    {label:<16} {cells}")


def latex_phi_only_table(per_dataset_at5):
    order = [("random", "Random (analytic)"),
             ("phi_s1", "$\\phi$-only (S1)"),
             ("phi",    "$\\phi$-only (full)")]
    lines = []
    lines.append("% === tab:phi_only (phi-only companion table, n_inst=5) ===")
    lines.append("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4.5pt}")
    lines.append("\\caption{Input-signal control at $n_{\\text{inst}}=5$: $\\phi$-only "
                 "(the bridge-edge signal, encoding-independent) vs.\\ the random lower bound, "
                 "the S1-only variant, and the full pipeline (Emb/Ref). "
                 "Read the staircase Random $<\\phi$-only(S1)$<\\phi$-only$<$Emb$<$Ref.}")
    lines.append("\\label{tab:phi_only}")
    lines.append("\\begin{tabular}{ll ccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Method & P@1 & P@3 & P@5 & MRR & R@GT \\\\")
    lines.append("\\midrule")
    for ds, res in per_dataset_at5.items():
        first = True
        for key, name in order:
            if key not in res:
                continue
            md = res[key]
            head = f"\\multirow{{3}}{{*}}{{{ds}}}" if first else ""
            first = False
            lines.append(f" {head} & {name} & "
                         f"{md['p@1']:.3f} & {md['p@3']:.3f} & {md['p@5']:.3f} & "
                         f"{md['mrr']:.3f} & {md['recall']:.3f} \\\\")
        lines.append(" & Emb & --- & --- & --- & --- & --- \\\\")
        lines.append(" & Ref & --- & --- & --- & --- & --- \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}\\end{table}")
    return "\n".join(lines)


def latex_prisma_phi_rows(per_dataset_at5):
    lines = ["% === Insert phi-only rows into tab:prisma_dataset (n_inst=5) ==="]
    for ds, res in per_dataset_at5.items():
        md = res["phi"]
        lines.append(f"% {ds}:")
        lines.append(f" & $\\phi$-only & {md['p@1']:.3f} & {md['p@3']:.3f} & "
                     f"{md['p@5']:.3f} & {md['mrr']:.3f} & {md['recall']:.3f} \\\\")
    return "\n".join(lines)


def discover_datasets(data_dir):
    base = Path(data_dir)
    found = {}
    if not base.exists():
        return found
    preferred = ["MusicRecordings", "Movielens", "MovieLens", "TPCH", "TPC-H", "PTCH"]
    names = sorted(p.name for p in base.iterdir() if p.is_dir())
    ordered = [n for n in preferred if n in names] + [n for n in names if n not in preferred]
    for ds in ordered:
        src = base / ds / "source"
        tgt = base / ds / "target"
        if not (src.is_dir() and tgt.is_dir()):
            continue
        targets = sorted(
            (p.name for p in tgt.iterdir()
             if p.is_dir() and (p / "target.csv").exists()),
            key=lambda s: (len(s), s),
        )
        if targets:
            found[ds] = targets
    return found


if __name__ == "__main__":
    DATA_DIR = "../data"

    N_INST_LIST = [5]
    RANDOM_SEED = 42
    N_PERM      = 200
    FORCE_REBUILD_PROFILES = False

    abs_data = os.path.abspath(DATA_DIR)
    print(f"[init] cwd            = {os.getcwd()}", flush=True)
    print(f"[init] DATA_DIR       = {DATA_DIR}  ->  {abs_data}", flush=True)
    print(f"[init] DATA_DIR exists= {os.path.isdir(abs_data)}", flush=True)

    DATASETS = discover_datasets(DATA_DIR)
    if not DATASETS:
        print("\n[FATAL] No datasets with source/ and target/ subdirectories found under ../data.", flush=True)
        sys.exit(1)

    print(f"[init] discovered datasets:", flush=True)
    for ds, tg in DATASETS.items():
        print(f"         {ds:<16} targets={tg}", flush=True)

    all_results = {}
    per_dataset_at5 = {}

    for dataset, targets in DATASETS.items():
        print(f"\n{'#' * 70}\n#  Dataset = {dataset}\n{'#' * 70}")
        all_results[dataset] = {}

        for n_inst in N_INST_LIST:
            print(f"\n{'=' * 60}\n  n_inst = {n_inst}\n{'=' * 60}")
            per_target = {}
            for tgt in targets:
                res = evaluate_one(
                    DATA_DIR, dataset, tgt, n_inst,
                    force_rebuild_profiles=FORCE_REBUILD_PROFILES,
                    random_seed=RANDOM_SEED, n_perm=N_PERM,
                )
                if res is None:
                    continue
                per_target[tgt] = res
                print(f"  --- {tgt} ---")
                print_variant_row("phi-only",       res["phi"])
                print_variant_row("  (det. argsort)", res["phi_det"])
                print_variant_row("phi-only(S1)",   res["phi_s1"])
                print_variant_row("random",         res["random"])

            all_results[dataset][n_inst] = per_target

            if per_target:
                print(f"\n  --- {dataset} / n_inst={n_inst}  Avg over {len(per_target)} targets ---")
                for key, lbl in [("phi", "phi-only"), ("phi_det", "phi-only(det)"),
                                 ("phi_s1", "phi-only(S1)"), ("phi_s2", "phi-only(S2)"),
                                 ("phi_s3", "phi-only(S3)"), ("phi_s4", "phi-only(S4)"),
                                 ("random", "random")]:
                    avg = _avg([r[key] for r in per_target.values()])
                    print_variant_row(lbl, avg)

                if n_inst == 5:
                    per_dataset_at5[dataset] = {
                        key: _avg([r[key] for r in per_target.values()])
                        for key in ["phi", "phi_s1", "phi_s2", "phi_s3", "phi_s4", "random", "phi_det"]
                    }

    os.makedirs("phi_only_out", exist_ok=True)
    json.dump(
        {ds: {ni: {tg: r for tg, r in pt.items()}
              for ni, pt in by_ni.items()}
         for ds, by_ni in all_results.items()},
        open("phi_only_out/phi_only_results.json", "w"), indent=2,
    )

    if per_dataset_at5:
        tex1 = latex_phi_only_table(per_dataset_at5)
        tex2 = latex_prisma_phi_rows(per_dataset_at5)
        open("phi_only_out/tab_phi_only.tex", "w").write(tex1)
        open("phi_only_out/tab_prisma_phi_rows.tex", "w").write(tex2)
        print("\n\n" + "=" * 70)
        print("LaTeX: tab:phi_only companion table")
        print("=" * 70)
        print(tex1)
        print("\n" + "=" * 70)
        print("LaTeX: phi-only rows for tab:prisma_dataset")
        print("=" * 70)
        print(tex2)

    print("\n[done] Results and LaTeX written to algos/phi_only_out/")