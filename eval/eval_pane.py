import json
import numpy as np

"""
This evaluates the accuracy of SM after PANE embedding.
Uses top-k or threshold to generate bridge edges; uses bidirectional or unidirectional representation for IND.
Similarity matrices are: (bi = bidirectional IND representation, uni = unidirectional IND representation)
sim_matrix_top_bi.json
sim_matrix_top_uni.json
sim_matrix_threshold_bi.json
sim_matrix_threshold_uni.json

"""


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

    # Load original similarity matrix and post-reward similarity matrix
    with open(sim_matrix_path, 'r') as f:
        sim_matrix = json.load(f)
    sim_matrix = np.array(sim_matrix)


    # ==============================================================================
    # Evaluation results using only PANE (sim_matrix):
    # ==============================================================================

    print("similarity matrix: ")
    print(sim_matrix)
    print()

    # -------------Evaluate matching results by top-k---------------
    print("\n" + "=" * 70)
    print("Evaluating original similarity matrix (without dependency rewards):")
    print("=" * 70)

    for k in [1, 2, 3, 4, 5, 6]:
        correct, precision = precision_topk(sim_matrix, gt_has_match, gt_indices, k)
        print(f"Accuracy@{k}= {precision:.4f} | Correct: {correct} / {int(gt_has_match.sum())}")
    