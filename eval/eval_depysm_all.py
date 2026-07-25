# coding: utf-8
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def precision_topk(sim_matrix, gt_has_match, gt_indices, k):
    m = sim_matrix.shape[0]
    topk_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    correct = sum(
        1 for i in range(m)
        if gt_has_match[i] and any(idx in gt_indices[i] for idx in topk_indices[i])
    )
    return correct, float(correct) / float(gt_has_match.sum())

def precision_threshold(sim_matrix, gt_has_match, gt_indices, t):
    m = len(sim_matrix)
    candi_dict = {
        i: list(np.where(sim_matrix[i] > t)[0])
        for i in range(m)
    }
    correct = sum(
        1 for i in range(m)
        if gt_has_match[i] and len(gt_indices[i] & set(candi_dict[i])) > 0
    )
    precision = float(correct) / float(gt_has_match.sum())
    return correct, precision


def compute_mrr(sim_matrix, gt_has_match, gt_indices):
    """
    MRR approach: take only the first position in each row's ranking list that hits ground truth.
    Only computed for rows where gt_has_match is True.
    """
    m = sim_matrix.shape[0]
    rr_sum = 0.0
    valid_count = 0

    for i in range(m):
        if not gt_has_match[i]:
            continue
        # Sort column indices by similarity from high to low
        ranked_cols = np.argsort(sim_matrix[i])[::-1]
        # Find the first position that hits ground truth (1-indexed)
        for rank, col in enumerate(ranked_cols, start=1):
            if col in gt_indices[i]:
                rr_sum += 1.0 / rank
                break
        valid_count += 1

    mrr = rr_sum / valid_count if valid_count > 0 else 0.0
    return mrr


def compute_recall_at_gt(sim_matrix, gt_has_match, gt_indices):
    """
    Recall@GT:
    Build a global ranking list (all (row, col) pairs sorted by similarity from high to low),
    take the top k = |G|, and compute how many ground truth pairs are covered.
    |G| = total number of ground truth pairs across all rows.
    """
    m, n = sim_matrix.shape

    # Build global ground truth set: (row index, column index) pairs
    G = set()
    for i in range(m):
        if gt_has_match[i]:
            for col in gt_indices[i]:
                G.add((i, col))
    k = len(G)

    if k == 0:
        return 0.0

    # Flatten sim_matrix, get all (row, col, similarity) triples and sort by similarity descending
    rows, cols = np.unravel_index(
        np.argsort(sim_matrix, axis=None)[::-1],
        sim_matrix.shape
    )

    # Take top k (row, col) pairs
    top_k_pairs = set(zip(rows[:k].tolist(), cols[:k].tolist()))

    # Compute intersection
    hits = len(G & top_k_pairs)
    recall_at_gt = hits / k
    return recall_at_gt


def gt_json_to_matrix(gt_json, n_source_cols, n_target_cols):
    gt_matrix = np.zeros((n_target_cols, n_source_cols), dtype=int)
    for i, (tgt_col, src_indices) in enumerate(gt_json.items()):
        for src_idx in src_indices:
            gt_matrix[i, src_idx] = 1
    return gt_matrix


if __name__ == "__main__":
    base_path = "/home/ouyang/Code/DepySM/data/TPCH/res"
    target_configs = [
        {"number": 5, "subdir": "target1"},  
        {"number": 5, "subdir": "target2"},   
        {"number": 6, "subdir": "target3"},  
        {"number": 6, "subdir": "target4"},  
        {"number": 5, "subdir": "target5"},  
        {"number": 7, "subdir": "target6"},  
        {"number": 4, "subdir": "target7"},  
        {"number": 5, "subdir": "target8"},      
    ]
    n_source = 61

    scores = {}

    for config in target_configs:
        res_path = base_path + "/" + config["subdir"]
        sim_matrix_path = res_path + "/sim_matrix.json"
        ground_true_path = res_path + "/ground_true.json"
        updated_sim_matrix_path = res_path + "/updated_sim_matrix_iter5_p2_k13_d4_c15.json"

        print()
        print("+" * 56)
        print(f"Evaluating {config['subdir']}")
        print("+" * 56)

        gt_json = json.load(open(ground_true_path, "r"))
        ground_true = gt_json_to_matrix(gt_json, n_source_cols=n_source, n_target_cols=config["number"])
        m = ground_true.shape[0]
        gt_has_match = (ground_true.max(axis=1) == 1)
        gt_indices = [
            set(np.where(ground_true[i] == 1)[0])
            for i in range(m)
        ]

        with open(sim_matrix_path, 'r') as f:
            sim_matrix = json.load(f)
        sim_matrix = np.array(sim_matrix)
        with open(updated_sim_matrix_path, 'r') as f:
            updated_sim_matrix = json.load(f)
        updated_sim_matrix = np.array(updated_sim_matrix)

        # ==============================================================
        # Original similarity matrix evaluation
        # ==============================================================
        print("\n" + "=" * 56)
        print("Evaluating original similarity matrix (without dependency rewards):")
        print("=" * 56)

        emb_scores = []
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision = precision_topk(sim_matrix, gt_has_match, gt_indices, k)
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))
            emb_scores.append(precision)

        emb_mrr = compute_mrr(sim_matrix, gt_has_match, gt_indices)
        emb_recall_gt = compute_recall_at_gt(sim_matrix, gt_has_match, gt_indices)
        print(f"MRR:        {emb_mrr:.4f}")
        print(f"Recall@GT:  {emb_recall_gt:.4f}")

        # ==============================================================
        # Updated similarity matrix evaluation
        # ==============================================================
        print("\n" + "=" * 56)
        print("Evaluating updated similarity matrices (with dependency rewards):")
        print("=" * 56)

        refine_scores = []
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision = precision_topk(updated_sim_matrix, gt_has_match, gt_indices, k)
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))
            refine_scores.append(precision)

        refine_mrr = compute_mrr(updated_sim_matrix, gt_has_match, gt_indices)
        refine_recall_gt = compute_recall_at_gt(updated_sim_matrix, gt_has_match, gt_indices)
        print(f"MRR:        {refine_mrr:.4f}")
        print(f"Recall@GT:  {refine_recall_gt:.4f}")

        scores[config["subdir"]] = {
            "emb_scores": emb_scores,
            "refine_scores": refine_scores,
            "emb_mrr": emb_mrr,
            "emb_recall_gt": emb_recall_gt,
            "refine_mrr": refine_mrr,
            "refine_recall_gt": refine_recall_gt,
        }

    # ==============================================================
    # Average metrics across all targets
    # ==============================================================
    print("+" * 56)
    print("Average metrics across all targets:")
    print("+" * 56)

    for k in range(10):
        avg_emb = np.mean([scores[c["subdir"]]["emb_scores"][k] for c in target_configs])
        avg_refine = np.mean([scores[c["subdir"]]["refine_scores"][k] for c in target_configs])
        print("Top-{} | Avg Emb Precision: {:.4f} | Avg Refine Precision: {:.4f}".format(k + 1, avg_emb, avg_refine))

    avg_emb_mrr = np.mean([scores[c["subdir"]]["emb_mrr"] for c in target_configs])
    avg_refine_mrr = np.mean([scores[c["subdir"]]["refine_mrr"] for c in target_configs])
    avg_emb_recall = np.mean([scores[c["subdir"]]["emb_recall_gt"] for c in target_configs])
    avg_refine_recall = np.mean([scores[c["subdir"]]["refine_recall_gt"] for c in target_configs])

    print(f"Avg MRR       | Emb: {avg_emb_mrr:.4f} | Refine: {avg_refine_mrr:.4f}")
    print(f"Avg Recall@GT | Emb: {avg_emb_recall:.4f} | Refine: {avg_refine_recall:.4f}")