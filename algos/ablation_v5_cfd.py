# coding: utf-8
"""
ablation_v5_cfd.py
==================
V5 (CFD-mediator) dose-response ablation driver. Fixed n_inst=5, seed=42.

Intervention point: only replace the cfds file read by emb_sm (--cfd_file cfds_f{tag}.txt), everything else remains unchanged.
- adj (FD/IND/bridge/mapped-dep edges) is identical across all fractions => reuse existing inst5 edge files, no rebuild.
- Each fraction only re-runs: embedding (emb_sm) + refinement.
- fraction 1.00 = V0 (full CFD); 0.75/0.50/0.25 = nested subsets; 0.00 = V5 degenerate point (all features empty, not run here).

Prerequisites: first complete the main experiment's inst5 pipeline (generates target/<tgt>/inst5/{bridge_edges,fd_edges,ind_edges}),
      then run make_cfd_subsets.py to generate cfds_f*.txt for each fraction. This script only performs "swap CFD -> re-embed -> re-refine".

Output:
  emb/<DS>.<tgt>.inst5_cfd{tag}.<d>.a.bin.{f,b}
  res/<tgt>/inst5_cfd{tag}/sim_matrix.json
  res/<tgt>/inst5_cfd{tag}/updated_sim_matrix_iter5_p2_k13_d4_c15.json
"""

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dependency_refinement import refinement_iterative


FRAC_TAG = {1.00: "100", 0.75: "075", 0.50: "050", 0.25: "025"}
FRACTIONS = [1.00, 0.75, 0.50, 0.25]   # 0.00=V5 degenerate point not run here (all features empty => PANE fails)
N_INST    = 5
RANDOM_SEED = 42

# Refinement hyperparameters (consistent with main experiment)
REFINE = dict(max_iter=5, patience=2, k_min=1, k_max=3, decay=0.4, ceiling=1.5)


def target_ncols(target_csv):
    """Number of target columns = number of header fields (used for emb_dim, avoiding hardcoding target_number)."""
    with open(target_csv, encoding="utf-8") as f:
        header = next(csv.reader(f))
    return len(header)


def run_one_fraction(data_dir, dataset, target_subdir, source_number, n_instance,
                     frac, iter_num, kappa, target_python, emb_dim):
    tag = FRAC_TAG[frac]
    inst_suffix = f"inst{N_INST}"
    emb_suffix  = f"{inst_suffix}_cfd{tag}"   # Distinguish embeddings/results for different fractions

    data_root  = f"{data_dir}/{dataset}"
    cfd_file   = f"{data_root}/source/cfds_f{tag}.txt"
    emb_path   = f"{data_root}/emb/{dataset}.{target_subdir}.{emb_suffix}.{emb_dim}.a.bin"
    res_dir    = f"{data_root}/res/{target_subdir}/{emb_suffix}"
    os.makedirs(res_dir, exist_ok=True)
    sim_path   = f"{res_dir}/sim_matrix.json"

    # Prerequisite file check (inst5 edges reuse main experiment artifacts)
    bridge = f"{data_root}/target/{target_subdir}/{inst_suffix}/bridge_edges.json"
    if not os.path.exists(bridge):
        print(f"  [skip] Missing inst5 bridge edges {bridge}; please run the main experiment inst5 pipeline first.")
        return False
    if not os.path.exists(cfd_file):
        print(f"  [skip] Missing {cfd_file}; please run make_cfd_subsets.py first.")
        return False

    print(f"  --- {dataset}/{target_subdir}  f={frac:.2f}  (cfds_f{tag}.txt) ---")

    # step2: Embedding (note: inst_suffix is still inst5 => emb_sm reads inst5 bridge/dependency edges; adj unchanged)
    #        Two new parameters: --cfd_file / --out_tag (see emb_sm.py patch)
    subprocess.call([
        target_python, "pane/emb_sm.py",
        "--data", dataset, "--target", target_subdir,
        "--inst_suffix", inst_suffix,
        "--cfd_file", cfd_file,
        "--out_tag", f"_cfd{tag}",
        "--d", str(emb_dim), "--t", str(iter_num),
        "--kappa", str(kappa), "--n_instance", str(n_instance),
    ])

    emb_f, emb_b = emb_path + ".f", emb_path + ".b"
    if not (os.path.exists(emb_f) and os.path.exists(emb_b)):
        print(f"  [error] Embedding file not generated: {emb_f}")
        return False

    target_number = emb_dim // 2 - source_number + 1   # Reverse computation (emb_dim=(src+tgt-1)*2)
    Xf = np.fromfile(emb_f, dtype=np.float64).reshape(source_number + target_number, emb_dim // 2)
    Xb = np.fromfile(emb_b, dtype=np.float64).reshape(source_number + target_number, emb_dim // 2)
    node_emb = np.hstack([Xf, Xb])
    sim = cosine_similarity(node_emb[source_number:], node_emb[:source_number]).tolist()
    json.dump(sim, open(sim_path, "w"), indent=2)

    # step3: Refinement (target instances are the same batch as main experiment: same seed, same sampling)
    target_csv = f"{data_root}/target/{target_subdir}/target.csv"
    rows = []
    with open(target_csv, encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            rows.append(list(row))
    if N_INST < len(rows):
        rng = np.random.default_rng(seed=RANDOM_SEED)
        idx = sorted(rng.choice(len(rows), size=N_INST, replace=False).tolist())
        rows = [rows[i] for i in idx]

    updated = refinement_iterative(
        similarity_matrix=sim, target_instances=rows, data=dataset,
        max_iter=REFINE["max_iter"], patience=REFINE["patience"],
        k_min=REFINE["k_min"], k_max=REFINE["k_max"],
        decay_factor=REFINE["decay"], score_ceiling=REFINE["ceiling"], verbose=False,
    )
    upd_name = (f"updated_sim_matrix_iter{REFINE['max_iter']}_p{REFINE['patience']}"
                f"_k{REFINE['k_min']}{REFINE['k_max']}_d{int(REFINE['decay']*10)}"
                f"_c{int(REFINE['ceiling']*10)}.json")
    json.dump(updated, open(f"{res_dir}/{upd_name}", "w"), indent=2)
    print(f"      done -> {res_dir}")
    return True


if __name__ == "__main__":
    DATA_DIR      = "../data"
    TARGET_PYTHON = "/home/ouyang/anaconda3/envs/PANE/bin/python"
    ITER_NUM      = 5
    KAPPA         = 1024

    # Per-dataset configuration: source_number and n_instance (number of source rows) must match the main experiment.
    # If n_instance for a dataset is uncertain, set it to None => fallback to max mediator id+1 from cfds.txt (only ensures the matrix is large enough).
    DATASETS = {
        "MusicRecordings": dict(source_number=6,  n_instance=658,
                                targets=[f"target{i}" for i in range(1, 7)]),
        "Movielens":       dict(source_number=13, n_instance=1010133,   # Fill in according to your main experiment
                                targets=[f"target{i}" for i in range(1, 7)]),
        "TPCH":            dict(source_number=61, n_instance=86806,   # Fill in according to your main experiment
                                targets=[f"target{i}" for i in range(1, 9)]),
    }

    for dataset, cfg in DATASETS.items():
        src_root = Path(DATA_DIR) / dataset / "source"
        if not src_root.exists():
            print(f"[skip] {dataset}: no source directory")
            continue

        source_number = cfg["source_number"]
        n_instance    = cfg["n_instance"]
        if source_number is None:
            # Auto-infer source column count = total columns across all CSVs in source directory
            n_cols = 0
            for csv_path in sorted(src_root.glob("*.csv")):
                with open(csv_path, encoding="utf-8") as f:
                    n_cols += len(next(csv.reader(f)))
            source_number = n_cols
            print(f"[auto] {dataset} source_number={source_number}")
        if n_instance is None:
            cfd_path = src_root / "cfds.txt"
            maxv = 0
            if cfd_path.exists():
                for line in open(cfd_path, encoding="utf-8"):
                    p = line.split()
                    if len(p) == 2:
                        maxv = max(maxv, int(p[1]))
            n_instance = maxv + 1
            print(f"[auto] {dataset} n_instance={n_instance} (fallback=max CFD mediator id+1)")

        print(f"\n{'#'*70}\n#  V5 CFD Dose-Response  Dataset={dataset}  src={source_number} n_inst_src={n_instance}\n{'#'*70}")
        for tgt in cfg["targets"]:
            target_csv = f"{DATA_DIR}/{dataset}/target/{tgt}/target.csv"
            if not os.path.exists(target_csv):
                print(f"  [skip] No {target_csv}")
                continue
            target_number = target_ncols(target_csv)
            emb_dim = (source_number + target_number - 1) * 2
            for frac in FRACTIONS:
                run_one_fraction(DATA_DIR, dataset, tgt, source_number, n_instance,
                                 frac, ITER_NUM, KAPPA, TARGET_PYTHON, emb_dim)

    print("\n[done] V5 dose-response all fractions completed. Next, run eval_v5_cfd.py for summary and plotting.")