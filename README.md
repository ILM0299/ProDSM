# ProDSM: Dependency-Driven Progressive Schema Matching

> Official experiment code for the paper:
> **ProDSM: Dependency-Driven Progressive Schema Matching for Zero-Metadata Target Schemas**

## Overview

ProDSM addresses the **instance-only zero-metadata schema matching** problem — a strict setting where the target schema exposes only raw data rows with columns identified by positional index (no column names, types, or descriptions). This scenario arises in anonymous data exports and privacy-scrubbed sharing.

The method exploits three classes of **data dependencies** — Functional Dependencies (FD), Inclusion Dependencies (IND), and Conditional Functional Dependencies (CFD) — in three complementary roles:

1. **Four-signal column profiling** builds cross-table bridge edges by scoring candidate column correspondences.
2. An **Attribute-Instance Dependency Graph (AID-Graph)** embeds FD/IND structure with CFD condition-satisfying instances as mediator nodes.
3. The **ADVR (Adaptive Dependency Validation & Refinement)** algorithm iteratively refines the similarity matrix by validating mapped FDs and INDs against the few available target rows.

To our knowledge, ProDSM is the first approach to use CFDs as graph mediators and FD/IND validation as an online refinement reward in schema matching.

## Project Structure

```
ProDSM/
├── algos/                          # Algorithm implementations
│   ├── DepySM_batch.py             # Main batch experiment runner (single dataset)
│   ├── DepySM_batch_more.py        # Extended batch runner (multi-dataset, instance sampling)
│   ├── DepySM.py                   # Single-target pipeline driver
│   ├── column_profiler.py          # One-pass streaming column profiling (HyperLogLog + reservoir)
│   ├── bridge_builder.py           # Four-signal column mapping scorer & bridge edge builder
│   ├── edge_builder.py             # Target-side dependency edge mapper (FD/IND/CFD -> target)
│   ├── dependency_refinement.py    # ADVR iterative similarity matrix refinement
│   ├── dependency_validator.py     # FD/IND/CFD validation library on relational instances
│   ├── util.py                     # Shared utilities (dependency parsing, top-k selection)
│   ├── DepySM_ablation.py          # Ablation study driver (V2-V5 variants)
│   ├── ablation_v5_cfd.py          # CFD dose-response ablation driver
│   ├── avg_eval.py                 # Multi-run averaging evaluator
│   ├── eval_phi_only.py            # Bridge-signal-only control experiment
│   ├── eval_v5_cfd.py              # CFD dose-response evaluation & plotting
│   ├── gen_cfds_from_rules.py      # Generate full CFD edge file from rule definitions
│   ├── make_cfd_subsets.py         # CFD subset generation (by mediator node)
│   ├── make_cfd_subsets_by_rule.py # CFD subset generation (by rule, authoritative version)
│   ├── pane/                       # PANE graph embedding engine
│   │   ├── emb_sm.py               # Graph construction + PANE random-walk embedding
│   │   ├── mycd.py                 # NMF via coordinate descent (adapted from scikit-learn)
│   │   └── cdnmf_fast.pyx          # Cython performance-critical NMF kernel
│   └── phi_only_out/               # Output of phi-only control experiment
│
├── eval/                           # Evaluation & analysis scripts
│   ├── eval_depysm.py              # Main ProDSM evaluation (Precision@k)
│   ├── eval_depysm_all.py          # Cross-target aggregation
│   ├── eval_depysm_all_dataset.py  # Cross-dataset comparison
│   ├── eval_depysm_all_inst.py     # Instance-size scaling analysis
│   ├── eval_depysm_detail.py       # Per-target detailed metrics
│   ├── eval_depysm_f1.py           # F1-score evaluation
│   ├── eval_ablation_inst5.py      # Ablation study evaluation (n_inst=5)
│   ├── eval_pane.py                # Raw PANE embedding evaluation (no refinement)
│   ├── eval_random.py              # Random baseline evaluation
│   ├── eval_reward_cfd.py          # Reward analysis with CFDs
│   ├── eval_reward_nocfd.py        # Reward analysis without CFDs
│   ├── eval_reward_iteration.py    # Per-iteration convergence analysis
│   ├── bridge_ablation.py          # Bridge edge ablation evaluation
│   └── sim_matrix_generator.py     # Similarity matrix generation utility
│
└── data/                           # Benchmark datasets
    ├── MusicRecordings/            # Single-table dataset (6 source columns)
    ├── Movielens/                  # Multi-table dataset (3 tables, 13 source columns)
    └── TPCH/                       # Complex dataset (8 tables, 61 source columns)
```

## Dependencies

### Python Environment

ProDSM requires **Python 3.10+** with the following packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation, embedding loading |
| `scipy` | Sparse matrices, randomized SVD |
| `scikit-learn` | Cosine similarity, NMF, preprocessing |
| `networkx` | Graph construction for AID-Graph |
| `fbpca` | Fast PCA / randomized SVD (used by PANE) |
| `matplotlib` | Plotting (dose-response curves, evaluation charts) |
| `psutil` | Memory monitoring during embedding |
| `pyximport` | Runtime Cython compilation for `cdnmf_fast.pyx` |
| `Cython` | Required for compiling the NMF kernel |

Install dependencies:

```bash
pip install numpy scipy scikit-learn networkx fbpca matplotlib psutil cython
```

### PANE Embedding Environment

The graph embedding step (`pane/emb_sm.py`) is executed as a **separate subprocess** and may require a dedicated Python environment. The interpreter path is configured via the `TARGET_PYTHON` variable in the batch scripts. Ensure this environment has `numpy`, `scipy`, `networkx`, `fbpca`, and `psutil` installed.

## Data Format

Each dataset follows a standardized directory layout:

```
data/<DatasetName>/
├── source/
│   ├── *.csv              # Source table(s) with headers
│   ├── fds.txt            # Functional dependencies (format: "0,1 -> 2")
│   ├── inds.txt           # Inclusion dependencies (format: "0[=1" or "0,1[=2,3")
│   ├── cfds.txt           # CFD mediator edges (format: "u v", global instance IDs)
│   ├── cfd_raw.txt        # CFD rule definitions (human-readable)
│   └── .profile_cache/    # Cached column profiles (auto-generated, JSON)
├── target/
│   └── target<N>/
│       └── target.csv     # Target instance rows (header = positional indices: 0,1,2,...)
├── emb/                   # PANE embedding binary files (auto-generated)
└── res/
    └── target<N>/
        ├── ground_true.json           # Ground truth mapping
        └── inst<K>/                   # Results for K sampled target instances
            ├── sim_matrix.json        # Raw cosine similarity (Emb baseline)
            └── updated_sim_matrix_*.json  # Refined similarity (ProDSM output)
```

### Dependency File Formats

**FDs** (`fds.txt`): One per line, `LHS -> RHS` where attributes are zero-based global column indices.  
Example: `0,1 -> 2` means columns 0 and 1 functionally determine column 2.

**INDs** (`inds.txt`): One per line, `LHS[=RHS` where `[=` denotes set inclusion.  
Example: `4[=9` means column 4 values are a subset of column 9 values.

**CFDs** (`cfds.txt`): One edge per line, `u v` where `u` is a global attribute index and `v` is a global instance index. These encode mediator nodes — instances that satisfy CFD condition patterns and bridge attribute columns.

### Ground Truth Format

`ground_true.json` maps each target column name to its list of valid source column indices (global):

```json
{
  "movie_id": [0, 10],
  "title": [1],
  "year": [2],
  "genres": [3]
}
```

## Quick Start

### 1. Prepare Data

Ensure the `data/` directory contains at least one dataset with the structure described above. Source CSV files, dependency files (`fds.txt`, `inds.txt`, `cfds.txt`), target CSVs, and ground truth must be in place.

### 2. Configure the Experiment

Edit the `__main__` block in `algos/DepySM_batch_more.py` (recommended) or `algos/DepySM_batch.py`:

```python
TARGET_PYTHON = "/path/to/your/python"  # Python interpreter for PANE embedding
data_dir_base = "../data"                # Root data directory (relative to algos/)
iter_num = 5                             # Random walk iterations for PANE
kappa = 1024                             # Compression dimension (for >10K instances)
```

For `DepySM_batch_more.py`, additional settings:

```python
datasets = ["MusicRecordings", "Movielens", "TPCH"]
n_target_instances_list = [2, 3, 5, 8, 10]  # Instance sampling sizes
RANDOM_SEED = 42
```

### 3. Run the Pipeline

```bash
cd algos
python DepySM_batch_more.py
```

This executes the full ProDSM pipeline for all datasets, all sampling sizes, and all targets.

For a single-dataset run:

```bash
cd algos
python DepySM_batch.py
```

### 4. Evaluate Results

```bash
cd eval
python eval_depysm_all_dataset.py   # Cross-dataset comparison
python eval_depysm_all_inst.py      # Instance-size scaling
python eval_ablation_inst5.py       # Ablation study
```

## Pipeline Details

The ProDSM pipeline consists of three sequential stages:

### Stage 1: Extended Graph Construction

**1a — Column Profiling** (`column_profiler.py`):  
A single O(N) pass over each source CSV compresses all column statistics into lightweight profile dictionaries. Uses HyperLogLog for high-cardinality columns (>500 unique values) and reservoir sampling for quantile estimation. Profiles are cached to disk with fingerprint-based invalidation.

**1b — Bridge Building** (`bridge_builder.py`):  
For every (target column, source column) pair, four compatibility signals are computed and fused:

| Signal | Weight | Description |
|--------|--------|-------------|
| S1: Value Subset | 0.40 | Whether target values fall within the source value domain |
| S2: Type Compatibility | 0.20 | Semantic type match (int, float, str, bool) with cardinality bonus |
| S3: Distribution Similarity | 0.20 | Range/quantile overlap (numeric) or length/charset match (string) |
| S4: Semantic Similarity | 0.20 | Edit-distance-based value matching across domains |

Top-k source columns per target column are selected as bridge edges.

**1c — Target Edge Mapping** (`edge_builder.py`):  
Source-side FDs, INDs, and CFDs are remapped onto the target schema through the bridge candidate mappings, validated against the available target instances, and written as target-side edges.

### Stage 2: Graph Embedding

**PANE Embedding** (`pane/emb_sm.py`):  
The AID-Graph (bridge edges + FD edges + IND edges + CFD mediator feature matrix) is embedded via the PANE algorithm:

1. Forward and backward feature propagation through random walks.
2. Log-normalization to produce affinity matrices.
3. Joint factorization: SVD for forward embedding, NMF (coordinate descent with Cython kernel) for backward embedding refinement.
4. Attribute dimension compression via randomized SVD when instances exceed 10,000 rows.

The output is a pair of binary embedding files (forward `.f` and backward `.b`), from which a cosine similarity matrix (target x source) is computed.

### Stage 3: ADVR Refinement

**Adaptive Dependency Validation and Refinement** (`dependency_refinement.py`):  
The raw similarity matrix is iteratively refined:

1. **Dynamic top-k selection**: Dual-factor candidate narrowing based on iteration progress and matrix sharpness.
2. **Adaptive reward computation**: Two-phase reward — decay-based in round 0, stability-exploration balanced in subsequent rounds.
3. **Dependency validation**: FDs and INDs mapped to the target are validated against available rows; validated mappings receive similarity rewards.
4. **Convergence detection**: Early stopping via relative Frobenius norm + top-1 stability with patience.
5. **Oscillation handling**: Detects alternating-pattern oscillation and resolves by averaging recent matrices.

## Ablation Studies

Run the full ablation suite:

```bash
cd algos
python DepySM_ablation.py      # V2(no IND graph), V3(random bridge), V4(FD-only), V5(IND-only)
python ablation_v5_cfd.py      # CFD dose-response at fractions 0.25/0.50/0.75/1.00
```

### Ablation Variants

| Variant | Description | What it tests |
|---------|-------------|---------------|
| V1 (Full Emb) | Full AID-Graph embedding, no refinement | Baseline embedding quality |
| V0 (Full Ref) | V1 + full ADVR refinement | Complete ProDSM |
| V2 (No IND-Graph) | Remove IND edges from graph | IND contribution to embedding |
| V3 (Random Bridge) | Replace signal-based bridge with random edges | Bridge edge quality matters |
| V4 (FD-only Ref) | ADVR with only FD validation | FD vs IND in refinement |
| V5 (IND-only Ref) | ADVR with only IND validation | IND vs FD in refinement |
| CFD dose-response | Vary CFD fraction: 1.00 -> 0.75 -> 0.50 -> 0.25 | CFD mediator contribution |

### Control Experiment

```bash
cd algos
python eval_phi_only.py        # Bridge-signal-only matching (no graph, no embedding, no refinement)
```

This establishes a staircase baseline: Random < phi-only(S1) < phi-only(full) < Emb < Ref.

## Benchmark Datasets

| Dataset | Source Tables | Source Columns | Total Rows | Targets | Description |
|---------|:---:|:---:|---:|:---:|-------------|
| MusicRecordings | 1 | 6 | ~100 | 6 | Single-table music metadata |
| Movielens | 3 | 13 | ~1,010,133 | 6 | Multi-table movie ratings (movies + users + ratings) |
| TPC-H | 8 | 61 | ~6,000,000+ | 8 | Complex TPC-H benchmark (8 tables with rich dependencies) |

Each dataset includes multiple target schemas of increasing complexity (single-table views to full cross-table joins).

## Refinement Hyperparameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `max_iter` | 5 | Maximum ADVR refinement iterations |
| `patience` | 2 | Consecutive converged rounds before early stop |
| `k_min` | 1 | Minimum dynamic top-k |
| `k_max` | 3 | Maximum dynamic top-k |
| `decay` | 0.4 | Reward decay factor per iteration (experiment value; function default is 0.6) |
| `ceiling` | 1.5 | Maximum similarity score ceiling |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{prodsim2025,
  title     = {ProDSM: Dependency-Driven Progressive Schema Matching for Zero-Metadata Target Schemas},
  author    = {Ouyang, Junjie and ...},
  booktitle = {...},
  year      = {2025}
}
```

## License

This project is released for academic research purposes. Please refer to the paper for data licensing information regarding the benchmark datasets (MusicRecordings, Movielens, TPC-H).
