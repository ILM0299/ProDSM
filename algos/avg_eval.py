# ===== main.py (Experiment + Evaluation Combined Version) =====
# coding: utf-8
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dependency_refinement import refinement_iterative
from edge_builder import build_target_edge
from bridge_builder import build_bridge


# ================================================================
# Evaluation Functions
# ================================================================

def precision_topk(sim_matrix, gt_has_match, gt_indices, k):
    m = sim_matrix.shape[0]
    topk_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    correct = sum(
        1 for i in range(m)
        if gt_has_match[i] and any(idx in gt_indices[i] for idx in topk_indices[i])
    )
    return correct, float(correct) / float(gt_has_match.sum())


def evaluate_all_targets(target_configs, base_path, updated_sim_filename, k_list):
    """
    Evaluate the current result file (which has been overwritten by this experiment run).
    Returns dict: {subdir: {"emb": [...], "refine": [...]}}
    """
    run_scores = {}
    for config in target_configs:
        subdir = config["subdir"]
        res_path = Path(base_path) / subdir
        sim_matrix_path = res_path / "sim_matrix.json"
        updated_sim_matrix_path = res_path / updated_sim_filename
        ground_true_path = res_path / "ground_true.csv"

        if not sim_matrix_path.exists() or not updated_sim_matrix_path.exists():
            print(f"  Warning: Files missing, skipping: {res_path}")
            continue

        ground_true = np.loadtxt(ground_true_path, delimiter=",")
        m = ground_true.shape[0]
        gt_has_match = ground_true.max(axis=1) == 1
        gt_indices = [set(np.where(ground_true[i] == 1)[0]) for i in range(m)]

        with open(sim_matrix_path, "r") as f:
            sim_matrix = np.array(json.load(f))
        with open(updated_sim_matrix_path, "r") as f:
            updated_sim_matrix = np.array(json.load(f))

        emb_scores    = [precision_topk(sim_matrix,         gt_has_match, gt_indices, k)[1] for k in k_list]
        refine_scores = [precision_topk(updated_sim_matrix, gt_has_match, gt_indices, k)[1] for k in k_list]

        run_scores[subdir] = {"emb": emb_scores, "refine": refine_scores}

    return run_scores


def print_avg_results(all_run_scores, target_configs, k_list):
    """Aggregate all run results, print mean and standard deviation."""
    valid_runs = len(all_run_scores)
    print(f"\n{'=' * 60}")
    print(f"Final Results Summary ({valid_runs} valid runs)")
    print(f"{'=' * 60}")

    # Print mean for each target
    for config in target_configs:
        subdir = config["subdir"]
        emb_runs    = np.array([r[subdir]["emb"]    for r in all_run_scores if subdir in r])
        refine_runs = np.array([r[subdir]["refine"] for r in all_run_scores if subdir in r])

        print(f"\n[{subdir}]  ({len(emb_runs)} valid runs)")
        print(f"{'Top-k':<8} {'Emb Mean':>10} {'Emb Std':>9} {'Ref Mean':>10} {'Ref Std':>9}")
        for ki, k in enumerate(k_list):
            print(
                f"Top-{k:<4} "
                f"{emb_runs[:, ki].mean():>10.4f} {emb_runs[:, ki].std():>9.4f} "
                f"{refine_runs[:, ki].mean():>10.4f} {refine_runs[:, ki].std():>9.4f}"
            )

    # Cross-target overall mean
    print(f"\n{'=' * 60}")
    print("Average Precision@k across all targets")
    print(f"{'=' * 60}")
    print(f"{'Top-k':<8} {'Emb Mean':>10} {'Emb Std':>9} {'Ref Mean':>10} {'Ref Std':>9}")
    for ki, k in enumerate(k_list):
        # First compute the mean across targets within each run, then compute the mean across runs
        emb_per_run    = [np.mean([r[c["subdir"]]["emb"][ki]    for c in target_configs if c["subdir"] in r]) for r in all_run_scores]
        refine_per_run = [np.mean([r[c["subdir"]]["refine"][ki] for c in target_configs if c["subdir"] in r]) for r in all_run_scores]
        print(
            f"Top-{k:<4} "
            f"{np.mean(emb_per_run):>10.4f} {np.std(emb_per_run):>9.4f} "
            f"{np.mean(refine_per_run):>10.4f} {np.std(refine_per_run):>9.4f}"
        )


# ================================================================
# Experiment Function (identical to the original run_depysm, no modifications)
# ================================================================

def run_depysm(data_name, source_number, target_config, n_instance, force_rebuild_profiles,
               iter_num, kappa, TARGET_PYTHON, data_dir_base,
               max_iter=5, patience=2, k_min=1, k_max=3, decay=0.4, ceiling=1.5):
    target_number = target_config["number"]
    target_subdir = target_config["subdir"]

    data_dir    = f"{data_dir_base}/{data_name}"
    target_path = f"{data_dir}/target/{target_subdir}/target.csv"
    emb_dim     = (source_number + target_number - 1) * 2
    emb_path    = f"{data_dir}/emb/{data_name}.{target_subdir}.{emb_dim}.a.bin"
    res_dir_path = f"{data_dir}/res/{target_subdir}"
    os.makedirs(res_dir_path, exist_ok=True)
    sim_path    = res_dir_path + "/sim_matrix.json"

    print(f"\n========== Starting processing {target_subdir} (target_number={target_number}) ==========")

    TARGET_INSTANCES: list[list[str]] = []
    try:
        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                TARGET_INSTANCES.append(list(row))
        print(f"Successfully read {target_subdir} data, total {len(TARGET_INSTANCES)} instances")
    except FileNotFoundError:
        print(f"Error: File {target_path} not found, skipping this target")
        return

    all_candidates = build_bridge(
        data_dir, TARGET_INSTANCES, source_number,
        bridge_edge_path=f"{data_dir}/target/{target_subdir}/bridge_edges.json",
        force_rebuild_profiles=force_rebuild_profiles
    )

    build_target_edge(
        all_candidates, TARGET_INSTANCES, source_number,
        fd_file=f"{data_dir}/source/fds.txt",
        ind_file=f"{data_dir}/source/inds.txt",
        out_fd_path=f"{data_dir}/target/{target_subdir}/fd_edges.txt",
        out_ind_path=f"{data_dir}/target/{target_subdir}/ind_edges.txt"
    )

    print(f"Starting emb_sm.py to generate embeddings ({target_subdir})...")
    subprocess.call([
        TARGET_PYTHON, "pane/emb_sm.py",
        "--data", data_name,
        "--target", target_subdir,
        "--d", str(emb_dim),
        "--t", str(iter_num),
        "--kappa", str(kappa),
        "--n_instance", str(n_instance)
    ])

    emb_f_path = emb_path + ".f"
    emb_b_path = emb_path + ".b"
    if not (Path(emb_f_path).exists() and Path(emb_b_path).exists()):
        print(f"Error: Embedding files do not exist, skipping Refinement")
        return

    Xf = np.fromfile(emb_f_path, dtype=np.float64).reshape(source_number + target_number, emb_dim // 2)
    Xb = np.fromfile(emb_b_path, dtype=np.float64).reshape(source_number + target_number, emb_dim // 2)
    node_embeddings = np.hstack([Xf, Xb])
    sim_matrix = cosine_similarity(node_embeddings[source_number:], node_embeddings[:source_number]).tolist()

    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_matrix, f, indent=2)

    updated = refinement_iterative(
        similarity_matrix=sim_matrix,
        target_instances=TARGET_INSTANCES,
        data=data_name,
        max_iter=max_iter, patience=patience,
        k_min=k_min, k_max=k_max,
        decay_factor=decay, score_ceiling=ceiling,
        verbose=False,
    )

    updated_filename = (
        f"updated_sim_matrix_iter{max_iter}_p{patience}"
        f"_k{k_min}{k_max}_d{int(decay * 10)}_c{int(ceiling * 10)}.json"
    )
    with open(res_dir_path + f"/{updated_filename}", "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    print(f"========== Finished processing {target_subdir} ==========\n")


# ================================================================
# Main Entry Point
# ================================================================

if __name__ == "__main__":
    # Basic configuration
    data_name = "Movielens"
    source_number = 13
    n_instance = 1010133
    force_rebuild_profiles = True
    iter_num = 5
    kappa = 1024  # Compressed dimension when the number of instances exceeds 10000
    TARGET_PYTHON = "/home/ouyang/anaconda3/envs/PANE/bin/python"
    data_dir_base = "../data"
    base_path       = f"{data_dir_base}/{data_name}/res"  # For evaluation


    target_configs = [
        {"number": 4, "subdir": "target1"},   # First target: 4 columns, directory target1
        {"number": 4, "subdir": "target2"},   # Second target: 4 columns, directory target2
        {"number": 5, "subdir": "target3"},  # Third target: 5 columns, directory target3
        {"number": 10, "subdir": "target4"},  # Fourth target: 10 columns, directory target4
        # Add more target configurations as needed
    ]

    refinement_params = {
        "max_iter": 5, "patience": 2,
        "k_min": 1,   "k_max": 3,
        "decay": 0.4, "ceiling": 1.5
    }

    # Evaluation configuration
    k_list = list(range(1, 11))
    updated_sim_filename = (
        f"updated_sim_matrix"
        f"_iter{refinement_params['max_iter']}"
        f"_p{refinement_params['patience']}"
        f"_k{refinement_params['k_min']}{refinement_params['k_max']}"
        f"_d{int(refinement_params['decay'] * 10)}"
        f"_c{int(refinement_params['ceiling'] * 10)}.json"
    )  # Consistent with the filename generated by run_depysm, no manual maintenance needed

    NUM_RUNS = 3          # Modify this to control the number of runs
    all_run_scores = []   # Collect evaluation results from each run

    for run_id in range(NUM_RUNS):
        print(f"\n{'#' * 60}")
        print(f"# Experiment {run_id + 1} / {NUM_RUNS}")
        print(f"{'#' * 60}")

        # Experiment (results overwrite previous run)
        for target_cfg in target_configs:
            run_depysm(
                data_name=data_name,
                source_number=source_number,
                target_config=target_cfg,
                n_instance=n_instance,
                force_rebuild_profiles=force_rebuild_profiles,
                iter_num=iter_num,
                kappa=kappa,
                TARGET_PYTHON=TARGET_PYTHON,
                data_dir_base=data_dir_base,
                **refinement_params
            )

        # Evaluate the current overwritten results
        print(f"\n--- Evaluation {run_id + 1} ---")
        run_scores = evaluate_all_targets(target_configs, base_path, updated_sim_filename, k_list)
        all_run_scores.append(run_scores)

    # Print final summary
    print_avg_results(all_run_scores, target_configs, k_list)
    print("\n========== All experiments completed ==========")