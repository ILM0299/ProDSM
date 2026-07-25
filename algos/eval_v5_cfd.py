# coding: utf-8
"""
eval_v5_cfd.py
==============
Aggregate V5 (CFD-mediator) dose-response ablation results, producing:
  1) Compact table requested by advisor: {V1, V5-Emb} and {V0, V5-Ref} x 3 datasets x 5 metrics
  2) Full dose-response table: fraction x {Emb, Ref} x 5 metrics (per dataset, averaged across targets)
  3) Three-panel dose-response plot (one panel per dataset, Ref full pipeline, 5 metric curves vs CFD fraction)
  4) Paste-ready LaTeX

Conventions (aligned with advisor's design):
  V0 = Ref with all CFDs (fraction 1.00) -- already available from main experiments
  V1 = Emb with all CFDs (fraction 1.00) -- already available from main experiments
  V5-Emb / V5-Ref = low-CFD variants; fraction 0.00 is the degenerate point (features all empty),
                     reported as 'degenerate'.
  Metric definitions are verbatim identical to eval_depysm_all_inst.py.
"""

import json
import os
from pathlib import Path

import numpy as np

# Plotting works in headless environments (no display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -- Metric helpers (verbatim copy from eval_depysm_all_inst.py) -------------------
def precision_topk(sim, gh, gi, k):
    topk = np.argsort(sim, axis=1)[:, -k:]
    correct = sum(1 for i in range(sim.shape[0]) if gh[i] and any(idx in gi[i] for idx in topk[i]))
    return float(correct) / float(gh.sum())

def compute_mrr(sim, gh, gi):
    s, c = 0.0, 0
    for i in range(sim.shape[0]):
        if not gh[i]:
            continue
        for rank, col in enumerate(np.argsort(sim[i])[::-1], 1):
            if col in gi[i]:
                s += 1.0 / rank; break
        c += 1
    return s / c if c else 0.0

def compute_recall_at_gt(sim, gh, gi):
    G = {(i, col) for i in range(sim.shape[0]) if gh[i] for col in gi[i]}
    k = len(G)
    if k == 0:
        return 0.0
    rows, cols = np.unravel_index(np.argsort(sim, axis=None)[::-1], sim.shape)
    return len(G & set(zip(rows[:k].tolist(), cols[:k].tolist()))) / k

def gt_to_matrix(gt_json, n_src):
    gm = np.zeros((len(gt_json), n_src), dtype=int)
    for i, (_, idxs) in enumerate(gt_json.items()):
        for j in idxs:
            gm[i, j] = 1
    return gm


TOPK = [1, 3, 5]
MKEYS = [f"p@{k}" for k in TOPK] + ["mrr", "recall"]
MLABEL = {"p@1": "P@1", "p@3": "P@3", "p@5": "P@5", "mrr": "MRR", "recall": "R@GT"}
FRAC_TAG = {1.00: "100", 0.75: "075", 0.50: "050", 0.25: "025"}
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
N_INST = 5
UPD_NAME = "updated_sim_matrix_iter5_p2_k13_d4_c15.json"


def metrics(sim_mat, gh, gi):
    out = {f"p@{k}": precision_topk(sim_mat, gh, gi, k) for k in TOPK}
    out["mrr"] = compute_mrr(sim_mat, gh, gi)
    out["recall"] = compute_recall_at_gt(sim_mat, gh, gi)
    return out


def eval_one(data_root, target_subdir, frac):
    """Return {'emb':{...}, 'ref':{...}} or None."""
    tag = FRAC_TAG[frac]
    res = f"{data_root}/res/{target_subdir}/inst{N_INST}_cfd{tag}"
    gt_path  = f"{data_root}/res/{target_subdir}/ground_true.json"
    sim_path = f"{res}/sim_matrix.json"
    upd_path = f"{res}/{UPD_NAME}"
    for p in (gt_path, sim_path, upd_path):
        if not os.path.exists(p):
            return None

    gt_json = json.load(open(gt_path))
    sim = np.array(json.load(open(sim_path)))
    upd = np.array(json.load(open(upd_path)))
    n_src = sim.shape[1]
    gm = gt_to_matrix(gt_json, n_src)
    gh = gm.max(axis=1) == 1
    gi = [set(np.where(gm[i] == 1)[0]) for i in range(gm.shape[0])]
    return {"emb": metrics(sim, gh, gi), "ref": metrics(upd, gh, gi)}


def avg(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in MKEYS} if dicts else None


def plot_dose_response(per_ds_frac, out_pdf):
    """Three-panel plot (one per dataset), Ref full pipeline, 5 metric curves vs CFD fraction."""
    dsets = list(per_ds_frac.keys())
    fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 4), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        for mk in MKEYS:
            xs, ys = [], []
            for f in FRACTIONS:
                cell = per_ds_frac[ds].get(f)
                if cell and cell.get("_avg_ref"):
                    xs.append(f); ys.append(cell["_avg_ref"][mk])
            if xs:
                ax.plot(xs, ys, marker="o", label=MLABEL[mk])
        ax.set_title(ds)
        ax.set_xlabel("CFD fraction")
        ax.set_ylabel("metric (Ref)")
        ax.set_xticks(FRACTIONS)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            # Place legend below the panel, outside the plot area, to avoid obscuring near-horizontal curves
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                      ncol=len(MKEYS), frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.suptitle("V5 CFD-mediator dose–response (n_inst=5, seed=42, nested subsets)")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[fig] -> {out_pdf}")


def latex_compact(per_ds_frac):
    """Compact table: {V1, V5-Emb} and {V0, V5-Ref} x dataset x metric. V5 uses the lowest viable fraction=0.25."""
    L = ["% === V5 CFD ablation compact table (n_inst=5) ===",
         "\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
         "\\caption{CFD-mediator ablation. V1/V0 = Emb/Ref with all CFDs (fraction 1.0); "
         "V5-Emb/V5-Ref = with the lowest CFD dose (fraction 0.25, nested). "
         "Fraction 0 (no CFD) degenerates the PANE embedding and is reported as N/A.}",
         "\\label{tab:cfd_ablation}",
         "\\begin{tabular}{ll ccccc}", "\\toprule",
         "Dataset & Variant & P@1 & P@3 & P@5 & MRR & R@GT \\\\", "\\midrule"]
    for ds, by_f in per_ds_frac.items():
        v1 = by_f.get(1.00, {}).get("_avg_emb")
        v0 = by_f.get(1.00, {}).get("_avg_ref")
        ve = by_f.get(0.25, {}).get("_avg_emb")
        vr = by_f.get(0.25, {}).get("_avg_ref")
        def row(name, m):
            if not m:
                return f" & {name} & N/A & N/A & N/A & N/A & N/A \\\\"
            return (f" & {name} & {m['p@1']:.3f} & {m['p@3']:.3f} & {m['p@5']:.3f} & "
                    f"{m['mrr']:.3f} & {m['recall']:.3f} \\\\")
        L.append(f"\\multirow{{4}}{{*}}{{{ds}}}")
        L.append(row("V1 Emb (all CFD)", v1))
        L.append(row("V5-Emb (f=0.25)", ve))
        L.append(row("V0 Ref (all CFD)", v0))
        L.append(row("V5-Ref (f=0.25)", vr))
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}\\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    DATA_DIR = "../data"
    DATASETS = {
        "MusicRecordings": [f"target{i}" for i in range(1, 7)],
        "Movielens":       [f"target{i}" for i in range(1, 7)],
        "TPCH":            [f"target{i}" for i in range(1, 9)],
    }

    per_ds_frac = {}   # per_ds_frac[ds][frac] = {"_avg_emb":..., "_avg_ref":..., per-target...}
    for ds, targets in DATASETS.items():
        data_root = f"{DATA_DIR}/{ds}"
        if not os.path.isdir(data_root):
            continue
        print(f"\n{'='*64}\n  {ds}\n{'='*64}")
        per_ds_frac[ds] = {}
        for frac in FRACTIONS:
            embs, refs = [], []
            for tgt in targets:
                r = eval_one(data_root, tgt, frac)
                if r is None:
                    continue
                embs.append(r["emb"]); refs.append(r["ref"])
            cell = {"_avg_emb": avg(embs), "_avg_ref": avg(refs)}
            per_ds_frac[ds][frac] = cell
            if cell["_avg_ref"]:
                e, rf = cell["_avg_emb"], cell["_avg_ref"]
                print(f"  f={frac:.2f}  Emb[" +
                      " ".join(f"{MLABEL[k]} {e[k]:.3f}" for k in MKEYS) + "]")
                print(f"           Ref[" +
                      " ".join(f"{MLABEL[k]} {rf[k]:.3f}" for k in MKEYS) + "]")
            else:
                print(f"  f={frac:.2f}  [no results, check whether res/<tgt>/inst5_cfd{FRAC_TAG[frac]}/ has been generated]")

    os.makedirs("v5_cfd_out", exist_ok=True)
    json.dump(per_ds_frac, open("v5_cfd_out/v5_cfd_results.json", "w"),
              ensure_ascii=False, indent=2)

    if any(any(c.get("_avg_ref") for c in by_f.values()) for by_f in per_ds_frac.values()):
        plot_dose_response(per_ds_frac, "v5_cfd_out/dose_response.pdf")
        tex = latex_compact(per_ds_frac)
        open("v5_cfd_out/tab_cfd_ablation.tex", "w").write(tex)
        print("\n" + "=" * 64 + "\nLaTeX compact table:\n" + "=" * 64)
        print(tex)

    print("\n[done] Results/plot/LaTeX written to algos/v5_cfd_out/")