import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

emb_path = "/home/ouyang/Code/DepySM/data/Movielens/emb/Movielens.34.a.bin" # embedding file path; suffix is bin but actually contains two files: .f and .b storing forward and backward embeddings
sim_path = "/home/ouyang/Code/DepySM/data/Movielens/res/sim_matrix_bridge_gt.json"   # output similarity matrix JSON file path
source_number = 13
target_number = 5
emb_dim = 34   


Xf = np.fromfile(emb_path+".f", dtype=np.float64).reshape(source_number+target_number, emb_dim//2)
Xb = np.fromfile(emb_path+".b", dtype=np.float64).reshape(source_number+target_number, emb_dim//2)
node_embeddings = np.hstack([Xf, Xb])  # shape=(n+m, d)
target = node_embeddings[source_number:]  # shape=(m, d)
source = node_embeddings[:source_number]  # shape=(n, d)
sim_matrix = cosine_similarity(target, source).tolist()  # shape=(m, n)
with open(sim_path, "w", encoding="utf-8") as f:
    json.dump(sim_matrix, f)