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
               patience=2, k_min=1, k_max=3, decay=0.4, ceiling=1.5):
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
    """
    target_number = target_config["number"]
    target_subdir = target_config["subdir"]  # Subdirectory for different targets, e.g., target1, target2

    # Dynamically construct paths
    data_dir = f"{data_dir_base}/{data_name}"
    target_path = f"{data_dir}/target/{target_subdir}/target.csv"  # ../data/Movielens/target/target1/target.csv
    emb_dim = (source_number + target_number - 1) * 2
    emb_path = f"{data_dir}/emb/{data_name}.{target_subdir}.{emb_dim}.a.bin"  # Embedding files distinguished by different targets
    res_dir_path = f"{data_dir}/res/{target_subdir}"  # Directory for storing sim_matrix and updated_sim_matrix for different targets
    os.makedirs(res_dir_path, exist_ok=True)
    sim_path = res_dir_path+"/sim_matrix.json"
    print(f"\n========== Starting processing {target_subdir} (target_number={target_number}) ==========")
    
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
    
    # build bridge edges for extend-graph
    all_candidates = build_bridge(data_dir, TARGET_INSTANCES, source_number,
                                bridge_edge_path=f"{data_dir}/target/{target_subdir}/bridge_edges.json", 
                                force_rebuild_profiles=force_rebuild_profiles)
    
    # Read candidate file (note: if different targets have different candidate files, the path needs to be adjusted)
    # candidate_path = Path(data_dir) / "test" / "all_candidates.json"
    # if candidate_path.exists():
    #     with open(candidate_path, "r", encoding="utf-8") as f:
    #         all_candidates = {"all_candidates": json.load(f)}
    # else:
    #     print(f"Warning: Candidate file {candidate_path} does not exist")
    #     return
    
    # build target edges for extend-graph (output paths distinguished by different targets)
    build_target_edge(
        all_candidates,
        TARGET_INSTANCES,
        source_number,
        fd_file = f"{data_dir}/source/fds.txt",
        ind_file = f"{data_dir}/source/inds.txt",
        out_fd_path = f"{data_dir}/target/{target_subdir}/fd_edges.txt",  # target1/fd_edges.txt
        out_ind_path = f"{data_dir}/target/{target_subdir}/ind_edges.txt"   # target1/ind_edges.txt
    )
 
    #================================================================
    # step2: Random walk-based embedding
    #================================================================
    print(f"Starting emb_sm.py to generate embeddings ({target_subdir})...")
    subprocess.call([
        TARGET_PYTHON,
        "pane/emb_sm.py",
        "--data", data_name,
        "--target", target_subdir,   # If emb_sm.py needs to distinguish targets
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
    print(f"========== Finished processing {target_subdir} ==========\n")


if __name__ == "__main__":
    # Basic configuration (globally shared)
    data_name = "TPCH"
    source_number = 61
    n_instance = 86806
    force_rebuild_profiles = False
    iter_num = 5
    kappa = 1024  # Compressed dimension when the number of instances exceeds 10000
    TARGET_PYTHON = "/home/ouyang/anaconda3/envs/PANE/bin/python"
    data_dir_base = "../data"

    # Define multiple target configurations (core: list of targets for batch processing)
    # Format: [{"number": number of target columns, "subdir": target subdirectory name}, ...]
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
    
    # Refinement parameters (global)
    refinement_params = {
        "max_iter": 5,
        "patience": 2,
        "k_min": 1,
        "k_max": 3,
        "decay": 0.4,
        "ceiling": 1.5
    }
    
    # Batch execute for each target
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
    
    print("\n========== All targets processed ==========")