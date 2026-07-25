import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def evaluate_top1(S, gt_pairs):
    """Top-1 prediction: for each tgt_j, select the most similar src_i"""
    gt_set = set(gt_pairs)
    predicted = set()
    for tgt_j in range(S.shape[0]):
        src_i = np.argmax(S[tgt_j])
        predicted.add((tgt_j, src_i))

    return compute_prf(predicted, gt_set)


def evaluate_hungarian(S, gt_pairs, threshold=None):
    """Hungarian matching + optional threshold filtering"""
    gt_set = set(gt_pairs)
    M = S.T  # (n_src, n_tgt)

    row_ind, col_ind = linear_sum_assignment(-M)

    predicted = set()
    for src_i, tgt_j in zip(row_ind, col_ind):
        if threshold is None or M[src_i, tgt_j] >= threshold:
            predicted.add((tgt_j, src_i))

    return compute_prf(predicted, gt_set)


def compute_prf(predicted, gt_set):
    TP = len(predicted & gt_set)
    precision = TP / len(predicted) if predicted else 0
    recall    = TP / len(gt_set)    if gt_set    else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    return {"TP": TP, "Predicted": len(predicted), "GT": len(gt_set),
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "F1":        round(f1, 4)}


def hungarian_pr_curve(S, gt_pairs, thresholds=None):
    """Sweep threshold, return P/R/F1 curve + best result"""
    if thresholds is None:
        thresholds = np.arange(0.0, 1.11, 0.01)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        r = evaluate_hungarian(S, gt_pairs, threshold=t)
        precisions.append(r["Precision"])
        recalls.append(r["Recall"])
        f1s.append(r["F1"])

    # Optimal threshold (oracle)
    best_idx = np.argmax(f1s)
    best_result = {
        "threshold": round(thresholds[best_idx], 2),
        "Precision": precisions[best_idx],
        "Recall":    recalls[best_idx],
        "F1":        f1s[best_idx],
    }

    return thresholds, precisions, recalls, f1s, best_result


def plot_pr_curve(thresholds, precisions, recalls, f1s, best_result, top1_result, tgt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left plot: P-R curve ---
    ax1.plot(recalls, precisions, color="steelblue", linewidth=2, label="Hungarian")
    ax1.scatter(                                      # Best F1 point
        best_result["Recall"], best_result["Precision"],
        color="red", zorder=5, s=80,
        label=f"Best F1={best_result['F1']:.3f} (t={best_result['threshold']})"
    )
    ax1.scatter(                                      # Top-1 point
        top1_result["Recall"], top1_result["Precision"],
        color="orange", marker="^", zorder=5, s=80,
        label=f"Top-1 F1={top1_result['F1']:.3f}"
    )
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    # --- Right plot: F1 vs Threshold ---
    ax2.plot(thresholds, f1s, color="steelblue", linewidth=2, label="Hungarian F1")
    ax2.axvline(best_result["threshold"], color="red",
                linestyle="--", label=f"Best threshold={best_result['threshold']}")
    ax2.axhline(top1_result["F1"], color="orange",
                linestyle="--", label=f"Top-1 F1={top1_result['F1']:.3f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("F1")
    ax2.set_title("F1 vs Threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"pr_curve_{tgt}.png", dpi=150)
    plt.show()

def main(S, gt_pairs, tgt):
    # 1. Top-1
    top1_result = evaluate_top1(S, gt_pairs)
    print("Top-1:", top1_result)

    # 2. Hungarian: sweep threshold + best result
    thresholds, precisions, recalls, f1s, best_result = hungarian_pr_curve(S, gt_pairs)
    print("Hungarian (oracle threshold):", best_result)

    # 3. Plot
    plot_pr_curve(thresholds, precisions, recalls, f1s, best_result, top1_result, tgt)


# -- Main pipeline --
data_name = "Movielens"

for i in range(1,4):
    print(f"\n=== Evaluating Target {i} ===")
    sim_matrix_path = f"../data/{data_name}/res/target{i}/sim_matrix.json"
    updated_sim_matrix_path = f"../data/{data_name}/res/target{i}/updated_sim_matrix_iter5_p2_k13_d4_c15.json"
    gt_path = f"../data/{data_name}/res/target{i}/ground_true.csv"

    with open(sim_matrix_path, "r") as f:
        sim_matrix = json.load(f)
    with open(updated_sim_matrix_path, "r") as f:
        updated_sim_matrix = json.load(f)
    S = np.array(sim_matrix)  # shape (n_tgt, n_src)
    UP_S = np.array(updated_sim_matrix)
    
    gt_array = np.loadtxt(gt_path, delimiter=",")
    gt_pairs = set()
    for j in range(gt_array.shape[0]):
        for k in range(gt_array.shape[1]):
            if gt_array[j, k] == 1:
                gt_pairs.add((j, k))
                break  # only take first matching src_i per row
    
    main(S, gt_pairs, i)
    main(UP_S, gt_pairs, f"up_{i}")




