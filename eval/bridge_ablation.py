import random
import json
from pathlib import Path
import subprocess
import numpy as np
from sm_res import precision_topk
from sklearn.metrics.pairwise import cosine_similarity


# ===================== Modifiable Parameters ======================
n_source = 13    # source end value: source = [1,2,...,n-1]
m_target = 18    # target end value: target = [n, n+1, ..., m-1]
data_dir = "../data/Movielens"
n = 13  # number of source nodes
m = 5   # number of target nodes
d = 34  # embedding dimension
# ======================================================

TARGET_PYTHON  = "/home/ouyang/anaconda3/envs/PANE/bin/python"
data_dir = Path(data_dir)
ground_true_path = "../eval/ground_true/Movielens.csv"
# Generate source and target lists
source = [i for i in range(0, n_source)]
target = [i for i in range(n_source, m_target)]


# Generate 10 files
for file_num in range(1, 11):
    result = []
    # Iterate over each source element, randomly select 3 different targets
    for i in source:
        # Select 3 targets without replacement
        random_j = random.sample(target, 3)
        result.append([i, random_j])

    # Define filename
    filename = data_dir / f"target/bridge_edge_random_{file_num}.json"

    # Write JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Generated: {filename}")

print("\n All 10 files generated successfully!")

# Embedding calculation
for file_num in range(1, 11):
    subprocess.call([
        TARGET_PYTHON,
        "/home/ouyang/Code/PANE/algos/pane/emb_sm_random_batch.py",
        "--data", "Movielens",
        "--d", "34",
        "--t", "5",
        "--kappa", "1024",
        "--file_num", str(file_num)
    ])

# Evaluation
# Read ground truth
ground_true = np.loadtxt(ground_true_path, delimiter=",")

# Ground truth info: column indices where value is 1 for each row (may be multiple)
gt_has_match = (ground_true.max(axis=1) == 1)          # whether the row has a true match
gt_indices = [
    set(np.where(ground_true[i] == 1)[0])               # all ground truth column indices per row
    for i in range(m)
]


# Store all top-k precisions across 10 runs
all_precisions = []

for file_num in range(1, 11):
    emb_path = "../algos/pane/emb/Movielens.34.random{}.a.bin".format(file_num)
    ground_true_path = "../eval/ground_true/Movielens.csv"

    Xf = np.fromfile(emb_path+".f", dtype=np.float64).reshape(n+m, d//2)
    Xb = np.fromfile(emb_path+".b", dtype=np.float64).reshape(n+m, d//2)
    node_embeddings = np.hstack([Xf, Xb])  # shape=(n+m, d)

    target = node_embeddings[n:]  # shape=(m, d)
    source = node_embeddings[:n]  # shape=(n, d)
    sim_matrix = cosine_similarity(target, source)  # shape=(m, n)
    
    print("\n------------ Evaluating top-k precision (round {}) ------------------".format(file_num))
    
    # Store all top-k precisions for a single run
    round_precision = []
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        correct, precision = precision_topk(sim_matrix, gt_has_match, gt_indices, k)
        print("Top-{} | Correct: {} / {} | Precision: {:.4f}".format(k, correct, int(gt_has_match.sum()), precision))
        round_precision.append(precision)

    # Add this round's results to the overall list
    all_precisions.append(round_precision)

# ===================== Compute Top-k Averages over 10 Runs =====================
print("\n" + "="*60)
print("                Final Average Precision over 10 Runs")
print("="*60)

# Transpose: [10 rounds, 10 k values] -> [10 k values, 10 rounds]
all_precisions = np.array(all_precisions).T
k_list = [1,2,3,4,5,6,7,8,9,10]

for i, k in enumerate(k_list):
    avg_p = np.mean(all_precisions[i])
    print("Average Top-{} Precision: {:.4f}".format(k, avg_p))