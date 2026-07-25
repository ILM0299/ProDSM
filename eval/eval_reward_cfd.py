import json
import numpy as np

"""
This evaluates the accuracy of SM after dependency validation and reward.
Different reward parameters are used to evaluate the accuracy of the reranked similarity matrix.


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


if __name__ == "__main__":
    sim_matrix_path = "/home/ouyang/Code/DepySM/data/Movielens/res/sim_matrix_top_uni.json"
    ground_true_path = "/home/ouyang/Code/DepySM/data/Movielens/res/ground_true.csv"
    rewards  = [
        {"fd": 0.01, "ind": 0.01, "cfd": 0.01},
        {"fd": 0.01, "ind": 0.01, "cfd": 0.015},
        {"fd": 0.01, "ind": 0.01, "cfd": 0.02},
        {"fd": 0.01, "ind": 0.01, "cfd": 0.03},
        
    ]   # reward parameters for dependency reranking

    # Read ground truth
    ground_true = np.loadtxt(ground_true_path, delimiter=",")

    # Ground truth info: column indices where value is 1 for each row (may be multiple)
    m = ground_true.shape[0]
    gt_has_match = (ground_true.max(axis=1) == 1)          # whether the row has a true match
    gt_indices = [
        set(np.where(ground_true[i] == 1)[0])               # all ground truth column indices per row
        for i in range(m)
    ]
    
    for reward in rewards:
        updated_path = f"/home/ouyang/Code/DepySM/data/Movielens/res/updated_sim_matrix_{reward['fd']}_{reward['ind']}.json" 
        with open(updated_path, 'r') as f:
            updated_sim_matrix = json.load(f)
        updated_sim_matrix = np.array(updated_sim_matrix)

        print("\n" + "=" * 70)
        print("Evaluating updated similarity matrix with reward fd={} and ind={}:".format(reward['fd'], reward['ind']))
        print("=" * 70)

        for k in [1, 2, 3, 4, 5, 6]:
            correct, precision = precision_topk(updated_sim_matrix, gt_has_match, gt_indices, k)
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))
