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

if __name__ == "__main__":
    # Parameters to change for each run
    rewards  = [
        {"fd": 0.10, "ind": 0.05, "cfd": 0.15},
        {"fd": 0.03, "ind": 0.03, "cfd": 0.05},
        {"fd": 0.01, "ind": 0.01, "cfd": 0.01},
        {"fd": 0.01, "ind": 0.01, "cfd": 0.02},
        {"fd": 0.03, "ind": 0.03, "cfd": 0.03},
        {"fd": 0.05, "ind": 0.05, "cfd": 0.05},
        {"fd": 0.10, "ind": 0.10, "cfd": 0.10}
        
    ]
    sim_matrix_path = "/home/ouyang/Code/DepySM/data/Movielens/res/sim_matrix.json"
    ground_true_path = "/home/ouyang/Code/DepySM/data/Movielens/res/ground_true.csv"

    # Read ground truth
    ground_true = np.loadtxt(ground_true_path, delimiter=",")

    # Ground truth info: column indices where value is 1 for each row (may be multiple)
    m = ground_true.shape[0]
    gt_has_match = (ground_true.max(axis=1) == 1)          # whether the row has a true match
    gt_indices = [
        set(np.where(ground_true[i] == 1)[0])               # all ground truth column indices per row
        for i in range(m)
    ]

    # Load original similarity matrix
    with open(sim_matrix_path, 'r') as f:
        sim_matrix = json.load(f)
    sim_matrix = np.array(sim_matrix)

    # ==============================================================================
    # Evaluation results without dependency reward mechanism (sim_matrix):
    # ==============================================================================

    # -------------Evaluate matching results by top-k---------------
    print("\n" + "=" * 80)
    print("Evaluating original similarity matrix (without dependency rewards):")
    print("=" * 80)

    print("similarity matrix: ")
    print(sim_matrix)
    print()

    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        correct, precision = precision_topk(sim_matrix, gt_has_match, gt_indices, k)
        print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))

    # ==============================================================================
    # Evaluation results with dependency reward mechanism (sim_matrix):
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Evaluating updated similarity matrices (with dependency rewards):")
    print("=" * 80)

    for reward in rewards:
        updated_sim_matrix_path = f"/home/ouyang/Code/DepySM/data/Movielens/res/updated_similarity_{reward['fd']}_{reward['ind']}_{reward['cfd']}.json"
        with open(updated_sim_matrix_path, 'r') as f:
            updated_sim_matrix = json.load(f)
        updated_sim_matrix = np.array(updated_sim_matrix)

        print("\n" + "-" * 15 + reward.__str__() + "-" * 15)
        print("\nupdated similarity matrix: ")
        print(updated_sim_matrix)
        print()

        # -------------Evaluate matching results by top-k---------------
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision = precision_topk(updated_sim_matrix, gt_has_match, gt_indices, k)
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))

   