import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dependency_refinement import refinement_iterative
from edge_builder import build_target_edge
from bridge_builder import build_bridge 


if __name__ == "__main__":
    data_name = "Movielens"  # Dataset name (used as the --data argument for emb_sm.py)
    source_number = 13  # Number of source columns (used for target column index offset)
    target_number = 5  # Number of target columns (used for the number of rows in the similarity matrix)
    n_instance = 1010133  # Number of instances (used as the --n_instance argument for emb_sm.py)
    
    force_rebuild_profiles = True,  # Whether to force re-scanning CSV to build profiles for bridge edges (ignore disk cache)
    iter_num = 5  # Number of iteration rounds (used as the --t argument for emb_sm.py)
    kappa = 1024  # Kappa parameter for random walks (used as the --kappa argument for emb_sm.py); 0 means no compression
    reward_validate_top = 3  # Number of top-N candidate instances considered in the validation phase of dependency re-ranking
    
    TARGET_PYTHON  = "/home/ouyang/anaconda3/envs/PANE/bin/python"  # Path to the Python interpreter for running pane/emb_sm.py
    target_path = f"../data/{data_name}/target/target.csv"  # Path to the target instance CSV file
    data_dir = f"../data/{data_name}"  # Root directory containing source, target, emb, and res files (includes source/, target/, emb/, and res/ subdirectories)
    emb_dim = (source_number + target_number-1) * 2  # Embedding dimension (used as the --d argument for emb_sm.py)
    emb_path = f"../data/{data_name}/emb/{data_name}.{emb_dim}.a.bin"
    sim_path = f"../data/{data_name}/res/sim_matrix.json"  # Path to the similarity matrix JSON file

    #================================================================
    # step1: Build bridge edges (top-3) and target edges, i.e., construct the extend-graph
    #================================================================

    # Read target instances from CSV, keeping all values as strings (no type conversion)
    TARGET_INSTANCES: list[list[str]] = []
    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            TARGET_INSTANCES.append(list(row))  # Preserve original strings without type conversion

    # build bridge edges for extend-graph
    all_candidates = build_bridge(data_dir, TARGET_INSTANCES, source_number, force_rebuild_profiles = force_rebuild_profiles)
    
    with open(Path(data_dir) / "test" / "all_candidates.json", "r", encoding="utf-8") as f:
        all_candidates = {"all_candidates": json.load(f)}
    
    # build target edges for extend-graph
    build_target_edge(all_candidates, TARGET_INSTANCES, source_number,
                      fd_file = data_dir+"/source/fds.txt",
                      ind_file = data_dir+"/source/inds.txt",
                      out_fd_path = data_dir+"/target/fd_edge.txt",
                      out_ind_path = data_dir+"/target/ind_edge.txt")
 
    #================================================================
    # step2: Random walk-based embedding
    #================================================================

    subprocess.call([
        TARGET_PYTHON,
        "pane/emb_sm.py",
        "--data", data_name,
        "--d", str(emb_dim),
        "--t", str(iter_num),
        "--kappa", str(kappa),
        "--n_instance", str(n_instance)
    ])
    
    #================================================================
    # step3: Dependency-based Refinement
    #================================================================
    max_iter=5
    patience=2
    k_min=1
    k_max=3
    decay=0.4
    ceiling=1.5

    # Compute similarity matrix
    Xf = np.fromfile(emb_path+".f", dtype=np.float64).reshape(source_number+target_number, emb_dim//2)
    Xb = np.fromfile(emb_path+".b", dtype=np.float64).reshape(source_number+target_number, emb_dim//2)
    node_embeddings = np.hstack([Xf, Xb])  # shape=(n+m, d)
    target = node_embeddings[source_number:]  # shape=(m, d)
    source = node_embeddings[:source_number]  # shape=(n, d)
    sim_matrix = cosine_similarity(target, source).tolist()  # shape=(m, n)
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim_matrix, f, indent=2)
    
    # Refinement
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

    # Save the refined similarity matrix
    updated_path = f"../data/{data_name}/res/updated_sim_matrix_iter{max_iter}_p{patience}_k{k_min}{k_max}_d{int(decay * 10)}_c{int(ceiling * 10)}" 
    with open(updated_path, "w") as f:
        json.dump(updated, f, indent=2)

