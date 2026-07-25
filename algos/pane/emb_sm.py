# coding: utf-8
import json
import time
import argparse
import numpy as np
import math
import networkx as nx
from sklearn.utils.extmath import randomized_svd
from scipy import sparse
from sklearn import preprocessing
import scipy.sparse as sp
from sklearn import preprocessing
import fbpca
import os
import psutil
import mycd
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_fd(fd):
    fd = fd.strip()
    if not fd or fd.startswith("#"):
        return []
    if "->" not in fd:
        raise ValueError("Invalid FD format (missing ->): %s", fd)
    lhs_part, rhs_part = fd.split("->", 1)
    lhs = [a.strip() for a in lhs_part.split(",") if a.strip()]
    rhs = [a.strip() for a in rhs_part.split(",") if a.strip()]

    return lhs, rhs

def parse_ind(ind):
    ind = ind.strip()
    if not ind or ind.startswith("#"):
        return []
    if "[=" not in ind:
        raise ValueError("Invalid IND format (missing [=): %s", ind)
    lhs_part, rhs_part = ind.split("[=", 1)
    lhs = [a.strip() for a in lhs_part.split(",") if a.strip()]
    rhs = [a.strip() for a in rhs_part.split(",") if a.strip()]

    return lhs, rhs

def normalize_transition(graph, directed):
    """
    Convert the adjacency matrix of a graph into a random walk transition matrix P = D^{-1}A.

    Args:
        graph (networkx graph): Input graph
        directed (_type_): Whether the graph is directed

    Returns:
        P(scipy.sparse.csr_matrix): Random walk transition matrix
    """

    adj = nx.adjacency_matrix(graph).astype(float)  # adjacency matrix adj
    #adj = adj + sp.eye(len(graph.nodes()))
    print(adj.shape)
    ind = range(len(graph.nodes()))
    degs = [0]*len(graph.nodes())   # diagonal matrix of out-degrees degs
    if directed:
        print("Directed", directed)
        for node in graph.nodes():
            if graph.out_degree(node)>0:
                degs[node] = 1.0/(graph.out_degree(node))
    else:
        for node in graph.nodes():
            if graph.degree(node)>0:
                degs[node] = 1.0/(graph.degree(node))

    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=adj.shape,dtype=np.float)) # compute inverse matrix
    P = degs.dot(adj) # P = D^{-1}A, i.e., random walk transition matrix

    return P


def sigmoid(x):
    return 1. / (1. + math.exp(-x))

def gen_emb(adj, F, k, t, kappa):   
    """Generate embeddings with optional attribute compression.

    Args:
        adj (scipy.sparse.csr_matrix): Random walk transition matrix
        F (_type_): Attribute matrix
        k (_type_): Embedding dimension
        t (_type_): Number of dimensionality reduction iterations
        kappa (_type_): Compressed attribute dimension

    Returns:
        _type_: _description_
    """
    # adj, F, k: number of dimensionality reduction iterations, kappa: compressed attribute dimension
    if kappa>0 and F.shape[1]>10000:
        print("running attribute clustering...")
        F2 = preprocessing.normalize(F.T, norm='l2', axis=1) # L2 normalize each row after transposition
        d = F2.shape[0] # attribute dimension
        U, _, _ = randomized_svd(F2, n_components=kappa, n_iter=t) # randomized SVD dimensionality reduction, complexity reduced to O(dn t), several orders of magnitude faster than traditional SVD
        # U: left singular matrix, dimension (d x kappa), each row corresponds to an original attribute, each column corresponds to a cluster principal component

        # Manually release memory
        del F2
        gc.collect()
        # U is (d x kappa), argmax(axis=1) finds the index of the largest element for each attribute (each row of U) --
        # i.e., assigns each attribute a "cluster label" (0 ~ kappa-1), belonging to "hard clustering" (one attribute belongs to one cluster).
        # cols is a list of length d, storing the cluster label of each attribute.
        cols = U.argmax(axis=1).flatten().tolist()
        del U
        gc.collect()
        # Generate the "attribute clustering indicator matrix" C, dimension (d x kappa), C[i][j] = 1 means the i-th attribute belongs to the j-th cluster, otherwise 0.
        # Essentially, C is the "one-hot encoding matrix" of attribute clustering, and also the core carrier for subsequent dimensionality reduction and inverse mapping.
        C = sp.csr_matrix( ([1]*d, (range(d), cols)), shape=(d,kappa) )
        # Matrix multiplication logic: F (n x d) x C (d x kappa) = new F (n x kappa).
        # Meaning: compress each node's d-dimensional attribute vector to kappa dimensions --
        # equivalent to each node's new attribute being the sum of attributes in each cluster it belongs to
        # (since C is a one-hot matrix, it is essentially the aggregated attribute value of the node on each cluster).
        F = F.dot(C)
        
        Xf, Y, Xb, Y = gen_emb_pane(adj, F, k, t)

        Y = C.dot(Y)

        print(Y.shape)
    else:
        Xf, Y, Xb, Y = gen_emb_pane(adj, F, k, t)

    return Xf, Y, Xb, Y

def gen_emb_pane(adj, features, k, t):
    """Generate PANE embeddings.

    Args:
        adj (scipy.sparse.csr_matrix): Random walk transition matrix
        features (_type_): Attribute matrix
        k (_type_): Embedding dimension
        t (_type_): Number of dimensionality reduction iterations

    Returns:
        _type_: _description_
    """



    print("running PANE...")
    n = adj.shape[0]
    d = features.shape[1]
    t1 = time.time()

    # Forward graph propagation (capturing forward structural information): based on the forward structure of the graph (original adjacency matrix adj),
    # iteratively aggregate nodes' "neighbor features" and "own features", producing the forward affinity matrix (Z).
    # Corresponding formula in the paper: P_f = alpha * sum_{l=0}^{inf} (1 - alpha)^l * P^l * R_r
    features = preprocessing.normalize(features, norm='l1', axis=1)
    Z = features
    alpha = 0.5 # random walk stopping probability
    for i in range(t):
       print("%d iteration", i)
       Z = (1-alpha)*adj.dot(Z) + features

    # Backward graph propagation (capturing backward structural information): based on the backward structure of the graph (transposed adjacency matrix),
    # iteratively aggregate nodes' "neighbor features" and "own features", producing the backward affinity matrix (Y).
    # Corresponding formula in the paper: P_b = alpha * sum_{l=0}^{inf} (1 - alpha)^l * (P^T)^l * R_c
    features = preprocessing.normalize(features, norm='l1', axis=0)
    Y = features
    adj = adj.T
    for i in range(t):
        print("%d iteration", i)
        Y = (1-alpha)*adj.dot(Y) + features

    del features
    del adj
    gc.collect()

    Z = alpha*Z
    Y = alpha*Y
    t2 = time.time()
    process = psutil.Process(os.getpid())
    print("step 1 takes ", t2-t1, process.memory_info().rss/1024.0/1024.0)

    # Compute the final affinity matrices after row normalization and column normalization, following the affinity formulas:
    # F[v_i, r_j] = log(n * p_f(v_i, r_j) / sum_{v_h in V} p_f(v_h, r_j) + 1)
    # B[v_i, r_j] = log(d * p_b(v_i, r_j) / sum_{r_h in R} p_b(v_i, r_h) + 1)
    # P_hat_f^(t)[v_i, r_j] = P_f^(t)[v_i, r_j] / sum_{v_l in V} P_f^(t)[v_l, r_j]
    # P_hat_b^(t)[v_i, r_j] = P_b^(t)[v_i, r_j] / sum_{r_l in R} P_b^(t)[v_i, r_l]
    # F' = log(n * P_hat_f^(t) + 1)
    # B' = log(d * P_hat_b^(t) + 1)
    print("logging...")
    if n<1e6:
        Z = preprocessing.normalize(Z, norm='l1', axis=0)
        Z.data = np.log2(n*Z.data+1)
        Y = preprocessing.normalize(Y, norm='l1', axis=1)
        Y.data = np.log2(d*Y.data+1)
    else: # approximate normalization for efficiency
        Z.data = np.log2(d*Z.data+1)
        Y.data = np.log2(n*Y.data+1)

    t3 = time.time()
    print("step 2 takes ", t3-t2, process.memory_info().rss/1024.0/1024.0)
    
    print("SVD....")

    # Joint Factorization of Affinity Matrices
    # Greedy initialization.(SVD)
    print(np.shape(Z))
    (U, s, Va) = fbpca.pca(Z, k/2, n_iter=t)
    del Z
    gc.collect()
    s = np.diag(s)
    Xf = fbpca.mult(U, s)
    Yf = Va.T
    print(Xf.shape, Yf.shape)
    Xb = fbpca.mult(Y, Yf)

    model = mycd.NMF(n_components=k/2, updateH=True, max_iter=t)
    Xb = model.fit_transform(Y, Xb, Yf.T)
    Yf = model.components_.T

    print(Xb.shape, Yf.shape)
    t4 = time.time()
    print("step 3 takes ", t4-t3, process.memory_info().rss/1024.0/1024.0)
    
    return Xf, Yf, Xb, Yf


def load_data(args):
    """
    Build and load the random walk matrix and attribute matrix.

    Returns:
        adj(scipy.sparse.csr_matrix): Random walk matrix
        features(scipy.sparse.csr_matrix): Attribute matrix
    """
    folder = os.path.join(BASE_DIR, "..", "..", "data") 
    fd_file = os.path.join(folder, args.data, "source", "fds.txt")
    ind_file = os.path.join(folder, args.data, "source", "inds.txt")
    cfd_file = (args.cfd_file if getattr(args, "cfd_file", None) else os.path.join(folder, args.data, "source", "cfds.txt"))
    target_bridge_file = os.path.join(folder, args.data, "target", args.target, args.inst_suffix, "bridge_edges.json") 
    target_edge_fd_file = os.path.join(folder, args.data, "target", args.target, args.inst_suffix, "fd_edges.txt")
    target_edge_ind_file = os.path.join(folder, args.data, "target", args.target, args.inst_suffix, "ind_edges.txt")      

    # Build random walk matrix
    
    
    rows=[]
    cols=[]
    n = 0 # maximum attribute index
    
    # Load random walk matrix from FD (directed)
    with open(fd_file,'r') as fin:
        print("loading from "+ fd_file)
        for line in fin:
            lhs, rhs = parse_fd(line)
            for u in lhs:
                for v in rhs:
                    u,v=int(u),int(v)
                    n = max(n, u, v)
                    rows.append(u)
                    cols.append(v)

    # Load random walk matrix from IND (undirected)
    # Skip when --no_ind_edges is set (ablation study V2: w/o IND-Graph)
    if args.no_ind_edges:
        print("[no_ind_edges] Skipping source IND edges")
    elif not os.path.exists(ind_file):
        print("IND file does not exist -> %s" % ind_file)
    else:
        print("loading from "+ ind_file)
        with open(ind_file,'r') as fin:
            for line in fin:
                lhs, rhs = parse_ind(line)
                for u in lhs:
                    for v in rhs:
                        u,v=int(u),int(v)
                        n = max(n, u, v)
                        rows.append(u)
                        cols.append(v)
                        rows.append(v)
                        cols.append(u)

    # Load random walk matrix from target FD edges (unidirectional)
    if os.path.exists(target_edge_fd_file):
        print("loading from "+ target_edge_fd_file)
        with open(target_edge_fd_file,'r') as fin:
            for line in fin:
                lhs, rhs = parse_fd(line)
                for u in lhs:
                    for v in rhs:
                        u,v=int(u),int(v)
                        n = max(n, u, v)
                        rows.append(u)
                        cols.append(v)
    
    # Load random walk matrix from target IND edges (bidirectional)
    # Skip when --no_ind_edges is set (ablation study V2: w/o IND-Graph)
    if not args.no_ind_edges and os.path.exists(target_edge_ind_file):
        print("loading from "+ target_edge_ind_file)
        with open(target_edge_ind_file,'r') as fin:
            for line in fin:
                lhs, rhs = parse_ind(line)
                for u in lhs:
                    for v in rhs:
                        u,v=int(u),int(v)
                        n = max(n, u, v)
                        rows.append(u)
                        cols.append(v)
                        rows.append(v)
                        cols.append(u)
    elif args.no_ind_edges:
        print("[no_ind_edges] Skipping target mapped IND edges")

    # Load random walk matrix from bridge edges (bidirectional)
    with open(target_bridge_file,'r') as fin:
        print("loading from "+ target_bridge_file)
        bridge_edges = json.load(fin)
        for t, ss in bridge_edges.items():
            for s in ss:
                u, v = int(s), int(t)
                n = max(n, u, v)
                # bridge edges are bidirectional
                rows.append(u)
                cols.append(v)
                rows.append(v)
                cols.append(u)
              
    adj = sparse.csr_matrix(([1]*len(rows), (rows, cols)),shape=(n+1,n+1),dtype=np.float)
    del rows
    del cols
    gc.collect()
    adj = preprocessing.normalize(adj, norm='l1', axis=1) # L1 normalization: convert adjacency matrix to random walk matrix (each row sums to 1)
    print("adjmatrix done")

    # Load attribute-node matrix
    print("loading from "+ cfd_file)
    rows=[]
    cols=[]
    m = args.n_instance # number of instances
    with open(cfd_file,'r') as fin:
        for line in fin:
            u,v = line.strip().split()
            u,v=int(u),int(v)
            rows.append(u)
            cols.append(v)

    features = sparse.csr_matrix(([1]*len(rows), (rows, cols)),shape=(n+1,m+1),dtype=np.float)
    if features.nnz == 0:
        print("[warn] CFD features matrix is empty (fraction=0 / no CFDs): PANE embedding will degenerate, results will be meaningless. This point should be reported as 'degenerate / N/A', not as valid numerical results.")

    del rows
    del cols
    gc.collect()
    print("featuresmatrix done")
    
    return adj, features

def save_emb(Xf,Xb,Yf,Yb,args):
    folder = os.path.join(BASE_DIR, "..", "..", "data", args.data, "emb")
    if not os.path.exists(folder):
        os.makedirs(folder)

    asuffix =".a.bin"
    tsuffix = ".t.bin"

    
    tag = getattr(args, "out_tag", "") or ""
    attr_emb_file = os.path.join(folder, "%s.%s.%s%s.%d%s" % (args.data, args.target, args.inst_suffix, tag, args.d, asuffix))
    tuple_emb_file = os.path.join(folder, "%s.%s.%s%s.%d%s" % (args.data, args.target, args.inst_suffix, tag, args.d, tsuffix))


    print("saving to %s"%attr_emb_file)
    print("saving to %s"%tuple_emb_file)
    with open(attr_emb_file+".f", "wb") as fout:
        np.asarray(Xf, dtype=np.float).tofile(fout)

    with open(attr_emb_file+".b", "wb") as fout:
        np.asarray(Xb, dtype=np.float).tofile(fout)

    with open(tuple_emb_file+".f", "wb") as fout:
        np.asarray(Yf, dtype=np.float).tofile(fout)

    with open(tuple_emb_file+".b", "wb") as fout:
        np.asarray(Yb, dtype=np.float).tofile(fout)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--target', type=str, help='target name')
    parser.add_argument('--inst_suffix', type=str, default="", help='suffix for different target instance numbers')  # distinguish embedding files with different instance counts
    parser.add_argument('--d', type=int, help='embedding dimensionality')
    parser.add_argument('--t', type=int, help='number of iterations')
    parser.add_argument('--kappa', type=int, default=0, help='dim for compressed attributes')  # use the original PANE when kappa=0
    parser.add_argument('--n_instance', type=int, default=0, help='number of instances')
    parser.add_argument('--no_ind_edges', action='store_true', default=False,
                        help='Ablation study V2: skip source inds.txt and target ind_edges.txt when building AID-Graph, '
                             'keep only Bridge Edges and FD edges, to verify the contribution of IND to graph structural information propagation')
    parser.add_argument('--cfd_file', type=str, default=None, help='Override default cfds.txt path (for V5 dose response)')
    parser.add_argument('--out_tag',  type=str, default="",   help='Append tag to embedding output filename, e.g. _cfd075')
    # parser.add_argument("--n_tgt_ins", type=int, default=10, help="number of target instances")
    args = parser.parse_args()

    print("loading data...")
    adj, features = load_data(args)

    print("processing...")
    Xf, Yf, Xb, Yb = gen_emb(adj, features, args.d, args.t, args.kappa)

    print("saving embeddings...") 
    save_emb(Xf,Xb,Yf,Yb,args)

# python2.7 ./emb_sm.py --data Movielens --d 34 --t 5 --kappa 1024 --n_instance 1010133
# Embedding dimension is d. Forward and backward propagation matrices are processed separately; the vectors from the two passes are concatenated,
# since SVD is performed on the attribute matrix each time, d/2 must be less than the number of instances and attributes.
# t is the number of iterations.
# In load_data(), m (number of instances) should be adjusted per dataset; kappa (compressed attribute dimension) can be adjusted based on available memory; 0 means no compression.
# In load_data(), adjust the type of bridge edges loaded: target_bridge_file = folder+args.data+"/target/bridge_edges_top.json"
# Adjust the target_bridge_file parameter and m parameter (number of instances) in load_data().
# The embedding file naming here differs from the original source code; it has been adapted for the schema mapping problem:
# 'a' denotes attribute embeddings, 't' denotes tuple embeddings, 'f'/'b' denote forward/backward respectively. Subsequent evaluation code needs to be adapted to the new file naming and format.