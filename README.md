# Nonlocal-RWNN: Exploring Nonlocal Random Walks in Non-Convolutional Graph Neural Networks

Recurrent, walk-based graph learning for node classification with **two interchangeable walkers**:

* **Uniform walker** – local, unbiased transitions over adjacency.
* **Non-local walker** – distance-aware transitions $G_{\alpha}$ that enable long-range hops via a decaying kernel (exponential or power/Lévy) with teleportation.

The model aggregates along **reversed random walks** using a **two-stage RNN**: a structural encoder on anonymous walk encodings and a merge GRU that fuses structure and node features. It tracks the **Dirichlet energy** of hidden states to diagnose over-smoothing and includes GCN/GAT baselines.

---

## Highlights

* **Recurrent aggregation over random walks**: long-range context without repeated local smoothing.
* **Plug-in walkers**: `uniform` (local) or `nonlocal` (global, distance-aware $G_{\alpha}$).
* **Built-in diagnostics**: Dirichlet energy of embeddings (lower energy ↔ stronger smoothing).
* **Baselines**: GCN/GAT with the same data splits + Dirichlet reporting.
* **Ray Tune** recipes for hyper-parameter search.
* **Works with common PyG datasets** (Planetoid, WebKB, Amazon, OGB-ArXiv, etc.).

---

## Method at a Glance

Given a reversed walk $w=(v_\ell,\ldots,v_0=v)$, we form:

- **Structural sequence** $\omega_u(w)$ via anonymous walk indices $\lambda_t$ and a sinusoidal map $\theta_t=2\pi\lambda_t/(\ell+1)$, $u_t=[\sin\theta_t,\cos\theta_t]$.
- **Semantic sequence** $\omega_x(w)=(\mathbf{h}_ {v_\ell},\ldots,\mathbf{h}_{v_0})$ (current node embeddings, plus optional degree).

A bidirectional GRU $\phi_u$ encodes $\omega_u(w)$ to per-step contexts $y_t$ and a pooled state $h^{(u)}_ {\text{init}}$.  
A merge GRU $\phi_x$ consumes zₜ = [ h<sub>vₜ</sub> ‖ yₜ (‖ d(vₜ)) ]with initial state $h^{(u)}_{\text{init}}$ and outputs the walk representation $h(w)$ for the terminal node $v$.


<p align="center"><b>h</b>′<sub>v</sub> = ψ(v) = (1/K) · Σ<sub>j=1…K</sub> h( w<sub>j</sub><sup>(v)</sup> )</p>

**Non-local walker** uses a precomputed transition:

$$
G_\alpha=c\tilde P+(1-c)\frac{1}{n}\mathbf{1}\mathbf{1}^\top,\qquad
\tilde P_{ij}\propto
\begin{cases}
\exp\big(-\alpha\[d(i,j)-1\]\big), & \text{exp}\\
d(i,j)^{-\alpha}, & \text{power/L\'evy}
\end{cases}
$$

with shortest-path distance $d(\cdot,\cdot)$, optional cutoff, and tiny self-loops for stability.


---

## Repository Structure

```
Nonlocal_RWNN/
├─ run.py                      # Main training script (uniform/nonlocal walkers, NRUM layer stack)
├─ model.py                    # Model wrapper (encoder → stacked RUMLayer → decoder + losses)
├─ layer.py                    # RUMLayer: walk sampling, anonymous enc., BiGRU(ϕ_u), GRU(ϕ_x)
├─ walker.py                   # Uniform local walker (fast CSR-like sampling)
├─ nonlocal_walker.py          # Non-local transition P_α + multinomial walk sampler (caching)
├─ rnn.py                      # Thin wrappers around torch.nn.GRU/LSTM (batch_first, reshape)
├─ data.py / TNMdata.py        # Dataset loaders + synthetic TNM utilities
├─ baselines_gcn_gat_dirichlet.py # GCN/GAT baselines with Dirichlet tracking
├─ aggregate.py                # Aggregate JSON results → summary + bootstrap CI
├─ tune.py                     # Ray Tune search utilities (Optuna, pruning, ETA logging)
├─ Dockerfile                  # Minimal CPU image with PyG/OGB/Ray dependencies
└─ data/                       # Example cached datasets (Cora, Texas) and raw splits
```

---

## Installation

### Option A: Conda + pip (recommended)

1. Create env and install PyTorch (choose CUDA/CPU build as needed):

```bash
conda create -n nonlocal-rwnn python=3.10 -y
conda activate nonlocal-rwnn

# Example: CPU builds (see https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# PyTorch Geometric + companions (match your torch/OS/CUDA)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f \
  https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA}.html
```

2. Install Python deps:

```bash
pip install ogb networkx tqdm ray[tune] optuna pandas numpy scikit-learn matplotlib
```

### Option B: Docker (CPU)

```bash
cd Nonlocal_RWNN
docker build -t nonlocal-rwnn .
docker run --rm -it -v $PWD:/workspace -w /workspace nonlocal-rwnn /bin/bash
```

> For GPU, install a CUDA-enabled PyTorch and the matching PyG wheels inside the container or use a CUDA base image.

---

## Quick Start

### 1) Train with the **uniform** walker

```bash
cd Nonlocal_RWNN
python run.py \
  --dataset Cora \
  --walker uniform \
  --nhid 64 --nlayer 1 --num_samples 4 --length 8 \
  --learning_rate 0.01 --dropout 0.1 --max_epoch 1000 \
  --out runs/cora/seed_42_uniform.json \
  --seed 42
```

### 2) Train with the **non-local** walker

```bash
python run.py \
  --dataset Cora \
  --walker nonlocal \
  --alpha 1.0 --c 0.15 --walk_mode exp --max_dist 6 \
  --nhid 64 --nlayer 1 --num_samples 4 --length 8 \
  --learning_rate 0.01 --dropout 0.1 --max_epoch 1000 \
  --out runs/cora/seed_42_nl.json \
  --seed 42
```

**Notes**

* `--num_samples` = $K$ walks per node; `--length` = walk length $\ell$ (reversed in the layer).
* Non-local distance cache is saved as `dist_{N}_{max_distance}.npy`.
* Outputs include **Dirichlet energy** of penultimate embeddings (proxy for over-smoothing).

---

## CLI: Key Arguments (from `run.py`)

* Data: `--dataset` (e.g., `Cora`, `Citeseer`, `Pubmed`, WebKB, OGB-ArXiv), `--data_dir`, `--split`.
* Model: `--nhid`, `--nlayer`, `--num_samples`, `--length`, `--dropout`, `--rnn_nlayer`, `--activation` (default `SiLU`).
* Regularisation: `--self_supervise_weight`, `--consistency_weight`, `--consistency_temperature`.
* Optimisation: `--optimizer`, `--learning_rate`, `--weight_decay`, `--max_epoch`, `--patience`,
  `--lr_decay_factor`, `--lr_patience`.
* Walker: `--walker {uniform,nonlocal}`, and for non-local:
  `--alpha`, `--c` (teleport), `--walk_mode {exp,power}`, `--max_dist` (or `None`).
* Repeats & I/O: `--repeats`, `--out`, `--seed`, `--eval_every`, `--device`.

---

## Results & Aggregation

Each run can save a JSON summary (`--out`) with:

```json
{
  "dataset": "...", "walker": "...",
  "K": 4, "L": 8, "nhid": 64, "nlayer": 1, "dropout": 0.1, "seed": 42,
  "best_val_acc": 82.4, "best_test_acc": 80.9, "epoch_last": 713,
  "time": 1724243452.00, "dirichlet": 0.038
}
```

Aggregate multiple seeds with bootstrap CIs:

```bash
python aggregate.py --runs_dir runs/cora --pattern "seed_*.json" --save_csv runs/cora/summary.csv
```

---

## Baselines

Run GCN/GAT baselines with identical splits and Dirichlet reporting:

```bash
# GCN
python baselines_gcn_gat_dirichlet.py --model gcn --dataset Cora --num_layers 2 --nhid 64 --dropout 0.5 \
  --max_epoch 1000 --patience 100 --out runs/cora_gcn/seed_42.json --seed 42

# GAT
python baselines_gcn_gat_dirichlet.py --model gat --dataset Cora --num_layers 2 --nhid 64 --heads 8 --dropout 0.5 \
  --max_epoch 1000 --patience 100 --out runs/cora_gat/seed_42.json --seed 42
```

---

## Hyper-parameter Tuning (Ray Tune)

`Nonlocal_RWNN/tune.py` contains a ready Optuna + Ray Tune setup (with ETA logging and pruning).
Example pattern (edit search spaces inside `tune.py` as needed):

```bash
python tune.py --data Wisconsin --walker nonlocal --num_trials 400 --gpu_per_trial 0.5 
```

---

## Datasets

* **Planetoid**: `Cora`, `Citeseer`, `Pubmed`
* **Amazon**: `Computers`, `Photo`
* **Coauthor**: `CS`
* **WebKB**, **WikipediaNetwork**, **Twitch**
* **OGB**: `ogbn-arxiv` variants
* **Synthetic TNM** (tree-pair graphs) via `TNMdata.py` / `synthetic_tnm.py` with flags
  `--tnm_r`, `--tnm_branching`, `--tnm_ntrain`, `--tnm_nval`, `--tnm_ntest`.

Datasets download automatically into `--data_dir` (default `./data`). Pre-cached examples for **Cora** and **Texas** are bundled under `Nonlocal_RWNN/data/`.

---

## Reproducibility

We set deterministic flags in `utils.set_seed` (PyTorch/CuDNN) and expose `--seed`.
Hardware/driver variation may still introduce minor non-determinism; we log the random `seed` in results.

---

## How This Addresses Over-smoothing & Over-squashing

* **No repeated Laplacian smoothing**: recurrent, path-wise integration preserves feature diversity; we track **Dirichlet energy** of embeddings.
* **Bypasses bottlenecks**: non-local transitions $_\alpha$ insert effective short-cuts, reducing long-path compression and easing over-squashing.
* **Long-range with fewer layers**: random walks reach distant context without deep stacks of local averaging.

(See thesis methods chapter for the full derivation and references.)

---

## Citation

If you find this repository useful, please consider citing the associated thesis/paper (replace with your citation):

```bibtex
@misc{2025nonlocalrwnn,
  title  = {Exploring Nonlocal Random Walks in Non-Convolutional Graph Neural Networks},
  author = {Yunxiang Wang and Francesco Tudisco and Kevin Zhang},
  year   = {2025},
  note   = {GitHub repository: https://github.com/Xiang227/Nonlocal_RWNN}
}
```

Related foundational references:

* Wang & Cho (2024), *Non-convolutional Graph Neural Networks* (Random Walk with Unifying Memory).
* Cipolla, Durastante & Tudisco (2021), *Nonlocal PageRank*.

---

## Acknowledgements

Built on **PyTorch Geometric**, **OGB**, **NetworkX**, and **Ray Tune**. Portions of the data utilities are adapted from Twitter Research’s *graph-neural-pde* (acknowledged in code comments).

---

## License

No license file is currently provided. If you plan to use this code beyond academic research, please open an issue to discuss appropriate licensing.

---

### Contact

Issues and pull requests are welcome. For questions about the non-local walker or reproducing reported metrics, please open a GitHub issue with your environment details and command line.
