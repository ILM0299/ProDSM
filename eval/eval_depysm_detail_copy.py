# coding: utf-8
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def precision_topk(sim_matrix, gt_has_match, gt_indices, k, col_names=None):
    m = sim_matrix.shape[0]
    topk_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    correct = 0
    errors = []
    hits = []  # new: record correct match information
    for i in range(m):
        if gt_has_match[i]:
            matched = [idx for idx in topk_indices[i] if idx in gt_indices[i]]
            if matched:
                correct += 1
                col_name = col_names[i] if col_names is not None else str(i)
                hits.append({
                    "target_col": col_name,
                    "predicted_top{}_src_idx".format(k): topk_indices[i].tolist(),
                    "ground_truth_src_idx": sorted(gt_indices[i]),
                    "matched_src_idx": matched  # actually matched indices
                })
            else:
                col_name = col_names[i] if col_names is not None else str(i)
                errors.append({
                    "target_col": col_name,
                    "predicted_top{}_src_idx".format(k): topk_indices[i].tolist(),
                    "ground_truth_src_idx": sorted(gt_indices[i])
                })
    return correct, float(correct) / float(gt_has_match.sum()), errors, hits


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


def gt_json_to_matrix(gt_json, n_source_cols, n_target_cols):
    gt_matrix = np.zeros((n_target_cols, n_source_cols), dtype=int)
    for i, (tgt_col, src_indices) in enumerate(gt_json.items()):
        for src_idx in src_indices:
            gt_matrix[i, src_idx] = 1
    return gt_matrix


def print_hits(hits, label):
    if not hits:
        print(f"  [{label}] No correct matches.")
    else:
        print(f"  [{label}] Correct match details:")
        for h in hits:
            tgt = h["target_col"]
            pred_key = [k for k in h if k.startswith("predicted")][0]
            pred = h[pred_key]
            gt = h["ground_truth_src_idx"]
            matched = h["matched_src_idx"]
            print(f"    Target col: '{tgt}' | {pred_key}: {pred} | ground_truth: {gt} | matched: {matched}")

def print_errors(errors, label):
    if not errors:
        print(f"  [{label}] No errors at this k.")
    else:
        print(f"  [{label}] Mismatch details:")
        for e in errors:
            tgt = e["target_col"]
            pred_key = [k for k in e if k.startswith("predicted")][0]
            pred = e[pred_key]
            gt = e["ground_truth_src_idx"]
            print(f"    Target col: '{tgt}' | {pred_key}: {pred} | ground_truth: {gt}")



def analyze_hit_transitions(
        orig_sim_matrix,
        updated_sim_matrix,
        gt_indices,
        target_col_names,
        k):

    m = orig_sim_matrix.shape[0]

    improved = []
    dropped = []

    for i in range(m):

        gt_set = gt_indices[i]
        tgt_name = target_col_names[i]

        # Top-k
        orig_topk = np.argsort(orig_sim_matrix[i])[::-1][:k]
        updated_topk = np.argsort(updated_sim_matrix[i])[::-1][:k]

        # Whether hit occurred
        orig_hit = any(gt in orig_topk for gt in gt_set)
        updated_hit = any(gt in updated_topk for gt in gt_set)

        record = {
            "target_col": tgt_name,
            "ground_truth": sorted(gt_set),
            "orig_topk": orig_topk.tolist(),
            "updated_topk": updated_topk.tolist()
        }

        # False -> True
        if (not orig_hit) and updated_hit:

            matched = [
                gt for gt in gt_set
                if gt in updated_topk
            ]

            record["newly_hit_gt"] = matched

            improved.append(record)

        # True -> False
        elif orig_hit and (not updated_hit):

            lost = [
                gt for gt in gt_set
                if gt in orig_topk
            ]

            record["lost_gt"] = lost

            dropped.append(record)

    # Output
    print("\n" + "#" * 70)
    print(f"Hit transition analysis @Top-{k}")
    print("#" * 70)

    # Improved
    print("\n[Improved Cases] False -> True")
    if not improved:
        print("  None")
    else:
        for r in improved:
            print("--------------------------------------------------")
            print(f"Target: {r['target_col']}")
            print(f"GT: {r['ground_truth']}")
            print(f"Newly hit GT: {r['newly_hit_gt']}")
            print(f"Original top-{k}: {r['orig_topk']}")
            print(f"Updated  top-{k}: {r['updated_topk']}")

    # Degraded
    print("\n[Dropped Cases] True -> False")
    if not dropped:
        print("  None")
    else:
        for r in dropped:
            print("--------------------------------------------------")
            print(f"Target: {r['target_col']}")
            print(f"GT: {r['ground_truth']}")
            print(f"Lost GT: {r['lost_gt']}")
            print(f"Original top-{k}: {r['orig_topk']}")
            print(f"Updated  top-{k}: {r['updated_topk']}")


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
        res_path = base_path + "/" + config["subdir"]
        sim_matrix_path = res_path + "/sim_matrix.json"
        ground_true_path = res_path + "/ground_true.json"
        updated_sim_matrix_path = res_path + "/updated_sim_matrix_iter5_p2_k13_d4_c15.json"

        print()
        print("+" * 56)
        print(f"Evaluating {config['subdir']}")
        print("+" * 56)

        gt_json = json.load(open(ground_true_path, "r"))
        # Extract target column names from gt_json (preserving order)
        target_col_names = list(gt_json.keys())

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

        # ==============================================================================
        # Evaluation results without dependency reward mechanism (sim_matrix):
        # ==============================================================================
        print("\n" + "=" * 56)
        print("Evaluating original similarity matrix (without dependency rewards):")
        print("=" * 56)

        emb_scores = []
        # Only print error details at top-1, top-3, top-5, top-10 to avoid excessive output; remove if statement to print for every k
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision, errors, hits = precision_topk(   # also receive hits
                sim_matrix, gt_has_match, gt_indices, k, col_names=target_col_names
            )
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(
                k, correct, int(gt_has_match.sum()), precision))
            if k in [1, 3, 5]:  # selectively print error and hit details for some k values
                print_hits(hits, label=f"orig top-{k}")    # new
                print_errors(errors, label=f"orig top-{k}")
            emb_scores.append(precision)

        # ==============================================================================
        # Evaluation results with dependency reward mechanism (updated_sim_matrix):
        # ==============================================================================
        print("\n" + "=" * 56)
        print("Evaluating updated similarity matrices (with dependency rewards):")
        print("=" * 56)

        refine_scores = []
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            correct, precision, errors, hits = precision_topk(   # also receive hits
                updated_sim_matrix, gt_has_match, gt_indices, k, col_names=target_col_names
            )
            print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(
                k, correct, int(gt_has_match.sum()), precision))
            # if k in [1, 3, 5, 10]:
            #     print_hits(hits, label=f"refine top-{k}")    # new
            #     print_errors(errors, label=f"refine top-{k}")
            refine_scores.append(precision)
            # if k in [1, 4]:  # perform hit transition analysis at top-5
            #     analyze_hit_transitions(
            #         sim_matrix,
            #         updated_sim_matrix,
            #         gt_indices,
            #         target_col_names,
            #         k=k
            #     )

        scores[config["subdir"]] = {
            "emb_scores": emb_scores,
            "refine_scores": refine_scores
        }

    print("+" * 56)
    print(f"Average Precision@k across all targets:")
    print("+" * 56)
    for k in [0, 2, 4, 9]: # selectively print average precision for some k values to avoid excessive output
        avg_emb_precision = np.mean([scores[config["subdir"]]["emb_scores"][k] for config in target_configs])
        avg_refine_precision = np.mean([scores[config["subdir"]]["refine_scores"][k] for config in target_configs])
        print("Top-{} | Average Emb Precision: {:.4f} | Average Refine Precision: {:.4f}".format(
            k + 1, avg_emb_precision, avg_refine_precision))