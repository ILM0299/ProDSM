import json
from pathlib import Path
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


def _make_tag(max_iter, patience, k_min, k_max, decay, ceiling) -> str:
        """Generate a unique filename tag containing all parameters to avoid naming conflicts between different config groups."""
        return (
            f"iter{max_iter}"
            f"_p{patience}"
            f"_k{k_min}{k_max}"
            f"_d{int(decay * 10)}"
            f"_c{int(ceiling * 10)}"
        )


if __name__ == "__main__":
    sim_matrix_path         = "/home/ouyang/Code/DepySM/data/Movielens/res/sim_matrix.json"
    ground_true_path        = "/home/ouyang/Code/DepySM/data/Movielens/res/ground_true.csv"
    data_dir                = Path("/home/ouyang/Code/DepySM/data/Movielens")
    static_updated_sim_path = "/home/ouyang/Code/DepySM/data/Movielens/res/updated_similarity_static_0.01_0.01.json"

    # -- PDMVR iterative refinement: parameter configuration --
    # Format: (max_iter, patience, k_min, k_max, decay, ceiling, comment)
    # base_reward removed; reward magnitude auto-computed from matrix and validated count
    # Group header rows: ("GROUP", title_text), skipped during evaluation
    iter_configs = [

        ("GROUP", "Group 1: decay variation, others fixed (controls initial round decay speed)"),
        (5, 2, 1, 3, 0.4, 1.5, "decay=0.4"),
        (5, 2, 1, 3, 0.6, 1.5, "decay=0.6"),
        (5, 2, 1, 3, 0.8, 1.5, "decay=0.8"),
        (5, 2, 1, 3, 1.0, 1.5, "decay=1.0 (no initial decay)"),

        ("GROUP", "Group 2: k_max variation, others fixed (controls candidate set width)"),
        (5, 2, 1, 2, 0.4, 1.5, "k_max=2"),
        (5, 2, 1, 3, 0.4, 1.5, "k_max=3 (recommended)"),
        (5, 2, 1, 5, 0.4, 1.5, "k_max=5"),

        ("GROUP", "Group 3: patience variation, others fixed"),
        (5, 1, 1, 3, 0.4, 1.5, "patience=1"),
        (5, 2, 1, 3, 0.4, 1.5, "patience=2"),
        (5, 3, 1, 3, 0.4, 1.5, "patience=3"),

        ("GROUP", "Group 4: score_ceiling variation, others fixed"),
        (5, 2, 1, 3, 0.4, 1.1, "ceiling=1.1"),
        (5, 2, 1, 3, 0.4, 1.5, "ceiling=1.5"),
        (5, 2, 1, 3, 0.4, 2.0, "ceiling=2.0"),

        ("GROUP", "Group 5: max_iter variation, others fixed"),
        (3, 2, 1, 3, 0.4, 1.5, "max_iter=3"),
        (5, 2, 1, 3, 0.4, 1.5, "max_iter=5"),
        (8, 2, 1, 3, 0.4, 1.5, "max_iter=8"),
    ]

    # Read ground truth
    ground_true  = np.loadtxt(ground_true_path, delimiter=",")
    m            = ground_true.shape[0]
    gt_has_match = (ground_true.max(axis=1) == 1)
    gt_indices   = [set(np.where(ground_true[i] == 1)[0]) for i in range(m)]

    # -- Evaluate PANE embedding effectiveness --
    print("\n" + "=" * 56)
    print("Evaluating PANE similarity matrix")
    print("=" * 56)
    with open(sim_matrix_path, "r", encoding="utf-8") as f:
        sim_matrix = np.array(json.load(f))
    for k in [1, 2, 3, 4, 5, 6]:
        correct, precision = precision_topk(sim_matrix, gt_has_match, gt_indices, k)
        print(f"Accuracy@{k}: {precision:.4f} | Correct: {correct} / {int(gt_has_match.sum())}")

    # -- Evaluate effectiveness after static dependency reward (fd=0.01, ind=0.01) --
    print("\n" + "=" * 56)
    print("Evaluating static rerank (fd=0.01, ind=0.01)")
    print("=" * 56)
    with open(static_updated_sim_path, "r", encoding="utf-8") as f:
        static_updated_sim_matrix = np.array(json.load(f))
    for k in [1, 2, 3, 4, 5, 6]:
        correct, precision = precision_topk(static_updated_sim_matrix, gt_has_match, gt_indices, k)
        print(f"Accuracy@{k}: {precision:.4f} | Correct: {correct} / {int(gt_has_match.sum())}")

    # -- Evaluate effectiveness after dynamic iterative dependency reward --
    for cfg in iter_configs:

        # Group header row: only print separator, skip evaluation
        if cfg[0] == "GROUP":
            print("\n" + "=" * 56)
            print(f"  ▶  {cfg[1]}")
            print("=" * 56)
            continue

        max_iter, patience, k_min, k_max, decay, ceiling, comment = cfg
        tag = _make_tag(max_iter, patience, k_min, k_max, decay, ceiling)

        updated_sim_path = data_dir / f"res/updated_similarity_{tag}.json"

        print(f"\n  [{comment}]")
        print(f"  max_iter={max_iter}  patience={patience}"
              f"  k=[{k_min},{k_max}]"
              f"  decay={decay}  ceiling={ceiling}")
        print(f"  file: updated_similarity_{tag}.json")
        print("  " + "-" * 38)

        with open(updated_sim_path, "r", encoding="utf-8") as f:
            updated_sim_matrix = np.array(json.load(f))

        for k in [1, 2, 3, 4, 5, 6]:
            correct, precision = precision_topk(updated_sim_matrix, gt_has_match, gt_indices, k)
            print(f"  Accuracy@{k}: {precision:.4f} | Correct: {correct} / {int(gt_has_match.sum())}")