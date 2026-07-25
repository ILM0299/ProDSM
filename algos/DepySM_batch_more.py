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


def run_depysm(data_name, source_number, target_config, n_instance, force_rebuild_profiles, 
               iter_num, kappa, TARGET_PYTHON, data_dir_base, max_iter=5, 
               patience=2, k_min=1, k_max=3, decay=0.4, ceiling=1.5,
               n_target_instances=None, random_seed=42):
    """
    DepySM run function for a single target group
    :param data_name: Dataset name
    :param source_number: Number of source columns
    :param target_config: Target configuration dictionary containing number and subdir (subdirectory name)
    :param n_instance: Number of instances
    :param force_rebuild_profiles: Whether to force rebuild profiles
    :param iter_num: Number of iteration rounds
    :param kappa: Random walk kappa parameter
    :param TARGET_PYTHON: Path to the Python interpreter
    :param data_dir_base: Base path of the data root directory
    :param max_iter: Maximum number of refinement iterations
    :param patience: Refinement patience
    :param k_min: Refinement k_min
    :param k_max: Refinement k_max
    :param decay: Decay factor
    :param ceiling: Score ceiling
    :param n_target_instances: Number of target instances to use (None means use all); when less than total, random sampling is applied
    :param random_seed: Random seed to ensure reproducible sampling
    """
    target_number = target_config["number"]
    target_subdir = target_config["subdir"]  # Subdirectory for different targets, e.g., target1, target2

    # Dynamically construct paths
    data_dir = f"{data_dir_base}/{data_name}"
    target_path = f"{data_dir}/target/{target_subdir}/target.csv"  # ../data/Movielens/target/target1/target.csv
    emb_dim = (source_number + target_number - 1) * 2
    # Instance count suffix: used to distinguish results from different sampling sizes
    inst_suffix = f"inst{n_target_instances}" if n_target_instances is not None else "instAll"
    emb_path = f"{data_dir}/emb/{data_name}.{target_subdir}.{inst_suffix}.{emb_dim}.a.bin"  # Embedding files distinguished by different targets and instance counts
    res_dir_path = f"{data_dir}/res/{target_subdir}/{inst_suffix}"  # Directory for storing results for different targets and instance counts
    os.makedirs(res_dir_path, exist_ok=True)
    sim_path = res_dir_path + "/sim_matrix.json"
    print(f"\n========== Starting processing {target_subdir} (target_number={target_number}, n_target_instances={n_target_instances}) ==========")
    
    #================================================================
    # step1: Build bridge edges (top-3) and target edges, i.e., construct the extend-graph
    #================================================================

    # Read target instances from CSV, keeping all values as strings (no type conversion)
    TARGET_INSTANCES: list[list[str]] = []
    try:
        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                TARGET_INSTANCES.append(list(row))
        print(f"Successfully read {target_subdir} data, total {len(TARGET_INSTANCES)} instances")
    except FileNotFoundError:
        print(f"Error: File {target_path} not found, skipping this target")
        return

    # Random sampling on demand: if n_target_instances is specified and less than total, randomly sample
    if n_target_instances is not None and n_target_instances < len(TARGET_INSTANCES):
        rng = np.random.default_rng(seed=random_seed)
        sampled_indices = sorted(rng.choice(len(TARGET_INSTANCES), size=n_target_instances, replace=False).tolist())
        TARGET_INSTANCES = [TARGET_INSTANCES[i] for i in sampled_indices]
        print(f"Randomly sampled {n_target_instances} instances (seed={random_seed}), sampled row indices: {sampled_indices}")
    elif n_target_instances is not None and n_target_instances > len(TARGET_INSTANCES):
        print(f"Warning: Requested instance count {n_target_instances} exceeds total {len(TARGET_INSTANCES)}, using all instances")
    
    # build bridge edges for extend-graph
    os.makedirs(f"{data_dir}/target/{target_subdir}/{inst_suffix}", exist_ok=True)  # Directory for storing bridge edge files for different instance counts
    all_candidates = build_bridge(data_dir, TARGET_INSTANCES, source_number,
                                bridge_edge_path=f"{data_dir}/target/{target_subdir}/{inst_suffix}/bridge_edges.json",
                                force_rebuild_profiles=force_rebuild_profiles)

    # Read candidate file (note: if different targets have different candidate files, the path needs to be adjusted)
    # candidate_path = Path(data_dir) / "test" / "all_candidates.json"
    # if candidate_path.exists():
    #     with open(candidate_path, "r", encoding="utf-8") as f:
    #         all_candidates = {"all_candidates": json.load(f)}
    # else:
    #     print(f"Warning: Candidate file {candidate_path} does not exist")
    #     return

    # build target edges for extend-graph (output paths distinguished by different targets and instance counts)
    build_target_edge(
        all_candidates, 
        TARGET_INSTANCES, 
        source_number,
        fd_file = f"{data_dir}/source/fds.txt",
        ind_file = f"{data_dir}/source/inds.txt",
        out_fd_path = f"{data_dir}/target/{target_subdir}/{inst_suffix}/fd_edges.txt",
        out_ind_path = f"{data_dir}/target/{target_subdir}/{inst_suffix}/ind_edges.txt"
    )
 
    #================================================================
    # step2: Random walk-based embedding
    #================================================================
    print(f"Starting emb_sm.py to generate embeddings ({target_subdir}, {inst_suffix})...")
    subprocess.call([
        TARGET_PYTHON,
        "pane/emb_sm.py",
        "--data", data_name,
        "--target", target_subdir,
        "--inst_suffix", inst_suffix,   # Distinguish embedding files by different instance counts
        "--d", str(emb_dim),
        "--t", str(iter_num),
        "--kappa", str(kappa),
        "--n_instance", str(n_instance)
    ])
    
    #================================================================
    # step3: Dependency-based Refinement
    #================================================================
    # Check if embedding files exist


    emb_f_path = emb_path + ".f"
    emb_b_path = emb_path + ".b"
    if not (Path(emb_f_path).exists() and Path(emb_b_path).exists()):
        print(f"Error: Embedding file {emb_f_path} or {emb_b_path} does not exist, skipping Refinement")
        return

    # Compute similarity matrix
    Xf = np.fromfile(emb_f_path, dtype=np.float64).reshape(source_number + target_number, emb_dim//2)
    Xb = np.fromfile(emb_b_path, dtype=np.float64).reshape(source_number + target_number, emb_dim//2)
    node_embeddings = np.hstack([Xf, Xb])  # shape=(n+m, d)
    target_emb = node_embeddings[source_number:]  # shape=(m, d)
    source_emb = node_embeddings[:source_number]  # shape=(n, d)
    sim_matrix = cosine_similarity(target_emb, source_emb).tolist()  # shape=(m, n)
    
    # Save original similarity matrix
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_matrix, f, indent=2)
    print(f"Saved original similarity matrix for {target_subdir} to {sim_path}")

    # Refinement
    print(f"Starting Refinement process for {target_subdir}...")
    updated = refinement_iterative(
        similarity_matrix=sim_matrix,
        target_instances=TARGET_INSTANCES,
        data=data_name,
        max_iter=max_iter,
        patience=patience,
        k_min=k_min,
        k_max=k_max,
        decay_factor=decay,
        score_ceiling=ceiling,
        verbose=False,
    )

    # Save the refined similarity matrix (distinguished by different targets)
    updated_filename = f"updated_sim_matrix_iter{max_iter}_p{patience}_k{k_min}{k_max}_d{int(decay * 10)}_c{int(ceiling * 10)}.json"
    updated_path = res_dir_path +f"/{updated_filename}"
    with open(updated_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    print(f"Saved refined similarity matrix for {target_subdir} to {updated_path}")
    print(f"========== Finished processing {target_subdir} ({inst_suffix}) ==========\n")


# ──────────────────────────────────────────────────────────
# Multi-dataset automatic inference helper functions
# ──────────────────────────────────────────────────────────
def _source_csvs(source_dir):
    """Return the actual data table CSVs in the source directory (excluding cfds/fds/inds and other non-table files, which are .txt)."""
    return sorted(Path(source_dir).glob("*.csv"))


def auto_source_number(source_dir):
    """source_number = sum of all source table column counts (independent of table order)."""
    total = 0
    for p in _source_csvs(source_dir):
        with open(p, encoding="utf-8") as f:
            total += len(next(csv.reader(f)))
    return total


def auto_n_instance(source_dir):
    """n_instance = sum of all source table data row counts (excluding header, independent of table order)."""
    total = 0
    for p in _source_csvs(source_dir):
        with open(p, encoding="utf-8") as f:
            total += sum(1 for _ in f) - 1
    return total


def auto_target_configs(target_dir):
    """Scan target* subdirectories under target/; target_number = number of header columns in target.csv."""
    cfgs = []
    base = Path(target_dir)
    if not base.exists():
        return cfgs
    subdirs = sorted((p for p in base.iterdir() if p.is_dir() and (p / "target.csv").exists()),
                     key=lambda p: (len(p.name), p.name))   # target1..target9 natural order
    for p in subdirs:
        with open(p / "target.csv", encoding="utf-8") as f:
            ncol = len(next(csv.reader(f)))
        cfgs.append({"number": ncol, "subdir": p.name})
    return cfgs


if __name__ == "__main__":
    # ===== Global configuration (shared by all three datasets) =====
    TARGET_PYTHON = "/home/ouyang/anaconda3/envs/PANE/bin/python"
    data_dir_base = "../data"
    iter_num = 5
    kappa = 1024
    refinement_params = {
        "max_iter": 5, "patience": 2, "k_min": 1, "k_max": 3, "decay": 0.4, "ceiling": 1.5,
    }
    n_target_instances_list = [2, 3, 5, 8, 10]
    RANDOM_SEED = 42

    # ===== Configuration for three datasets =====
    # source_number / n_instance / target_configs left as None will be auto-inferred (recommended to avoid manual errors).
    # To fix them, fill in explicitly (e.g., MusicRecordings with 6/658).
    DATASETS = [
        {"data_name": "MusicRecordings", "source_number": None, "n_instance": None, "target_configs": None},
        {"data_name": "Movielens",       "source_number": None, "n_instance": None, "target_configs": None},
        {"data_name": "TPCH",            "source_number": None, "n_instance": None, "target_configs": None},
    ]

    for ds in DATASETS:
        data_name = ds["data_name"]
        data_dir  = f"{data_dir_base}/{data_name}"
        src_dir   = f"{data_dir}/source"
        tgt_dir   = f"{data_dir}/target"

        if not os.path.isdir(src_dir):
            print(f"\n[skip] {data_name}: {src_dir} not found")
            continue

        # Auto-infer missing fields
        source_number  = ds["source_number"] if ds["source_number"] is not None else auto_source_number(src_dir)
        n_instance     = ds["n_instance"]    if ds["n_instance"]    is not None else auto_n_instance(src_dir)
        target_configs = ds["target_configs"] if ds["target_configs"] is not None else auto_target_configs(tgt_dir)

        if not target_configs:
            print(f"\n[skip] {data_name}: no target* subdirectory with target.csv found under {tgt_dir}")
            continue

        tgt_summary = ", ".join("{}(cols={})".format(c["subdir"], c["number"]) for c in target_configs)
        print("\n" + "=" * 70)
        print("=  Dataset {}: source_number={}  n_instance={}".format(data_name, source_number, n_instance))
        print("=   targets=" + tgt_summary)
        print("=" * 70)

        for n_inst in n_target_instances_list:
            print(f"\n{'#' * 60}")
            print(f"# {data_name}  n_target_instances = {n_inst}")
            print(f"{'#' * 60}")
            for ti, target_cfg in enumerate(target_configs):
                # Force rebuild profiles only on the very first time (smallest n_inst, first target) per dataset
                force_rebuild_profiles = (n_inst == n_target_instances_list[0] and ti == 0)
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
                    n_target_instances=n_inst,
                    random_seed=RANDOM_SEED,
                    **refinement_params
                )
            print(f"\n# {data_name}  n_target_instances = {n_inst} done")

    print("\n========== All three datasets processed ==========")