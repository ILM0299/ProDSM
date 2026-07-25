# coding: utf-8
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Evaluate matching results by top-k: for each row, select the k columns with highest similarity; if at least one column index is in the ground truth column index set for that row, the row is considered correctly matched.
def precision_topk(sim_matrix, gt_has_match, gt_indices, k):
    m = sim_matrix.shape[0]
    topk_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    correct = sum(
        1 for i in range(m)
        if gt_has_match[i] and any(idx in gt_indices[i] for idx in topk_indices[i])
    )
    return correct, float(correct) / float(gt_has_match.sum())

# Evaluate candidate matching results by similarity threshold: for each row, select column indices with similarity greater than t as candidates; if at least one candidate column index is in the ground truth column index set for that row, the row is considered correctly matched.
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


def gt_json_to_matrix(gt_json, n_source_cols,n_target_cols):
    """
    gt_json:
        readable ground truth dict


    n_source_cols:
        total source columns (global index space)
    """

    

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
        {"number": 8, "subdir": "target3"},  
        {"number": 7, "subdir": "target4"},  
        {"number": 6, "subdir": "target5"},  
        {"number": 7, "subdir": "target6"},  
        {"number": 6, "subdir": "target7"},  
        {"number": 6, "subdir": "target8"},      
    ]
    n_source = 61

    

    scores = {}



    for config in target_configs:
        res_path = base_path + "/" +config["subdir"]
        sim_matrix_path = res_path + "/sim_matrix.json"
        ground_true_path = res_path + "/ground_true.json"
        updated_sim_matrix_path = res_path + "/updated_sim_matrix_iter5_p2_k13_d4_c15.json"


        print()
        print("+"*56)
        print(f"Evaluating {config["subdir"]}")
        print("+"*56)

        # Ground truth info: column indices where value is 1 for each row (may be multiple)
        # Read ground truth
        gt_json = json.load(open(ground_true_path, "r"))
        ground_true = gt_json_to_matrix(gt_json, n_source_cols=n_source, n_target_cols=config["number"])
        # ground_true = np.loadtxt(ground_true_path, delimiter=",")
        m = ground_true.shape[0]
        gt_has_match = (ground_true.max(axis=1) == 1)          # whether the row has a true match
        gt_indices = [
            set(np.where(ground_true[i] == 1)[0])               # all ground truth column indices per row
            for i in range(m)
        ]

        # Load original similarity matrix and post-reward similarity matrix
        with open(sim_matrix_path, 'r') as f:
            sim_matrix = json.load(f)
        sim_matrix = np.array(sim_matrix)
        with open(updated_sim_matrix_path, 'r') as f:
            updated_sim_matrix = json.load(f)
        updated_sim_matrix = np.array(updated_sim_matrix)

        # ==============================================================================
        # Evaluation results without dependency reward mechanism (sim_matrix):
        # ==============================================================================

        # -------------Evaluate matching results by top-k---------------
        print("\n" + "=" * 56)
        print("Evaluating original similarity matrix (without dependency rewards):")
        print("=" * 56)

        # print("similarity matrix: ")
        # print(sim_matrix)
        # print()

        emb_scores = []
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision = precision_topk(sim_matrix, gt_has_match, gt_indices, k)
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))
            emb_scores.append(precision)

        # ==============================================================================
        # Evaluation results with dependency reward mechanism (sim_matrix):
        # ==============================================================================
        print("\n" + "=" * 56)
        print("Evaluating updated similarity matrices (with dependency rewards):")
        print("=" * 56)

        # print("\nupdated similarity matrix: ")
        # print(updated_sim_matrix)
        # print()

        # -------------Evaluate matching results by top-k---------------
        refine_scores = []
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision = precision_topk(updated_sim_matrix, gt_has_match, gt_indices, k)
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))
            refine_scores.append(precision)


        scores[config["subdir"]] = {
            "emb_scores": emb_scores,
            "refine_scores": refine_scores
        }
    

    print("+"*56)
    print(f"Average Precision@k across all targets:")
    print("+"*56)
    for k in range(10):
        avg_emb_precision = np.mean([scores[config["subdir"]]["emb_scores"][k] for config in target_configs])
        avg_refine_precision = np.mean([scores[config["subdir"]]["refine_scores"][k] for config in target_configs])
        print("Top-{} | Average Emb Precision: {:.4f} | Average Refine Precision: {:.4f}".format(k+1, avg_emb_precision, avg_refine_precision))