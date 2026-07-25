# coding: utf-8
"""
make_cfd_subsets.py
===================
Generate proportionally sampled cfds files for the V5 (CFD-mediator) dose-response ablation.

Intervention point
------------------
CFD instance mediator nodes = distinct values in the second column (v) of source/cfds.txt
read by emb_sm.py.
emb_sm.py reads (u, v) into the PANE attribute matrix features[u][v]=1
(u=attribute node, v=CFD mediator node).
"Removing CFD mediators" = deleting certain v nodes and all their associated (u, v) edges
=> corresponds to deleting columns from features.
FD/IND/bridge edges are all in adj and are completely unrelated, so adj remains unchanged
across all fractions (as required by advisor).

Why "nested" subsets
--------------------
We do not perform 5x repeated sampling. With a single seed, if each fraction were sampled
independently, curves might be non-monotonic merely due to sampling luck (e.g., the 0.50
sample might miss a critical CFD that 0.25 happened to include). Nested subsets guarantee
0.25 subset 0.50 subset 0.75 subset 1.00: each increment only adds, never removes CFDs,
so any monotonic trend is structural, not sampling noise.
Implementation: fix seed=42, shuffle all mediator nodes once randomly; the first floor(f*N)
nodes form the subset for fraction f.

Fraction semantics
------------------
1.00 = all CFDs = V0 (fully consistent with the existing main experiment)
0.75 / 0.50 / 0.25 = nested decreasing subsets
0.00 = V5 (no CFDs at all) -- features all empty, PANE degenerates, method fails;
       this script writes an empty file and explicitly warns; the experiment reports
       this point as "degenerate / undefined" rather than fabricating values.

Usage
-----
    cd algos
    python make_cfd_subsets.py            # generate for all datasets
Output: data/<DS>/source/cfds_f100.txt, cfds_f075.txt, cfds_f050.txt, cfds_f025.txt
        (and cfds_f000.txt if needed)
Also prints the total CFD mediator node count N per dataset (for cross-checking
the advisor's figures of 35 / 251 / 312).
"""

import os
import sys
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
FRACTIONS   = [1.00, 0.75, 0.50, 0.25]   # actual doses to run; 0.00=V5 degenerate point handled separately
FRAC_TAG    = {1.00: "100", 0.75: "075", 0.50: "050", 0.25: "025", 0.00: "000"}


def read_cfd_edges(cfd_path):
    """Read source/cfds.txt (u v format). Return [(u, v), ...] edge list with stable appearance order."""
    edges = []
    with open(cfd_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"cfds.txt line format should be 'u v', got: {line!r}")
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
    return edges


def nested_mediator_subsets(edges, fractions, seed=RANDOM_SEED):
    """
    Return {fraction: kept_v_set}. Kept sets are shuffled by seed and use prefixes to guarantee nesting.
    Sampling unit = distinct CFD mediator nodes v (removing a node removes all its associated edges).
    """
    # Take distinct v in "first-appearance order" to ensure fully reproducible results with the seed
    seen = {}
    for _u, v in edges:
        if v not in seen:
            seen[v] = len(seen)
    mediators = list(seen.keys())          # distinct v, stable order
    N = len(mediators)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)              # one-time shuffle with fixed seed
    shuffled = [mediators[i] for i in perm]

    out = {}
    for f in fractions:
        k = int(np.floor(f * N + 1e-9))    # ⌊f·N⌋
        out[f] = set(shuffled[:k])         # prefix => nested
    return out, N


def write_subset(edges, kept_v, out_path):
    """Keep only edges where v is in kept_v, write output. If kept_v is empty, write an empty file (V5 degenerate point)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for u, v in edges:
            if v in kept_v:
                f.write(f"{u} {v}\n")


def process_dataset(data_dir, dataset, fractions=FRACTIONS, also_zero=False):
    src = Path(data_dir) / dataset / "source"
    cfd_path = src / "cfds.txt"
    if not cfd_path.exists():
        print(f"  [skip] {dataset}: {cfd_path} not found")
        return None

    edges = read_cfd_edges(cfd_path)
    subsets, N = nested_mediator_subsets(edges, fractions)

    print(f"  {dataset:<16} Total CFD mediator nodes N = {N}  (cross-check advisor's 35/251/312), edge count = {len(edges)}")
    # Self-consistency check: nesting relation 0.25 subset 0.50 subset 0.75 subset 1.00
    ordered = sorted(fractions)
    for a, b in zip(ordered, ordered[1:]):
        assert subsets[a] <= subsets[b], f"Nesting violated: f={a} is not a subset of f={b}"

    for f in fractions:
        tag = FRAC_TAG[f]
        out_path = src / f"cfds_f{tag}.txt"
        write_subset(edges, subsets[f], out_path)
        print(f"      f={f:.2f}  mediators kept {len(subsets[f]):>4}/{N}  -> {out_path.name}")

    if also_zero:
        out_path = src / "cfds_f000.txt"
        write_subset(edges, set(), out_path)
        print(f"      f=0.00  mediators kept    0/{N}  -> {out_path.name}  "
              f"[V5 degenerate point: features all empty, PANE fails, reported as 'degenerate' in experiment]")
    return N


if __name__ == "__main__":
    DATA_DIR = "../data"
    # Dataset names consistent with main experiment (auto-discover those containing source/cfds.txt)
    base = Path(DATA_DIR)
    if not base.exists():
        print(f"[FATAL] {os.path.abspath(DATA_DIR)} does not exist; please run from algos/.")
        sys.exit(1)

    datasets = sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and (p / "source" / "cfds.txt").exists()
    )
    if not datasets:
        print(f"[FATAL] No datasets with source/cfds.txt found under {os.path.abspath(DATA_DIR)}.")
        sys.exit(1)

    print(f"[init] seed={RANDOM_SEED}  fractions={FRACTIONS}  datasets={datasets}\n")
    for ds in datasets:
        process_dataset(DATA_DIR, ds, also_zero=True)
    print("\n[done] CFD subsets for all fractions written to the corresponding source/ directories.")