
from typing import Optional, Tuple, List
import os
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import to_undirected

from synthetic_tnm import make_tnm_tree_pair_graph



def _parse_int_token(parts: List[str], prefix: str, default: Optional[int] = None) -> Optional[int]:
    """Parses integer values like 'r8' / 'b2' / 'n60' from a list of name-split tokens."""
    for p in parts:
        if p.startswith(prefix):
            try:
                return int(p[len(prefix):])
            except Exception:
                pass
    return default


# ========= TNM：path  =========

def _make_path_graph(r: int, start_id: int = 0) -> Tuple[torch.Tensor, int, int]:
    """
    Construct an undirected path of length r: node start_id . start_id+r, root is start_id.
    Returns (edge_index, root_id, n_nodes).
    """
    edges = []
    for i in range(r):
        u = start_id + i
        v = start_id + i + 1
        edges.append((u, v)); edges.append((v, u))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    root = start_id
    n_nodes = r + 1
    return edge_index, root, n_nodes


def _make_path_pair(r: int, mismatch: bool = False, start_id: int = 0) -> Tuple[torch.Tensor, Tuple[int, int], int]:
    """
    Generate two paths of length r (P1, P2):
      - match: two identical paths;
      - mismatch: one more leaf for P2 only at the "end of the distance from root = r" (minimum difference).
    The labels are written uniformly in the build phase (here only the structure is generated).
    """
    e1, root1, n1 = _make_path_graph(r, start_id)
    e2, root2, n2 = _make_path_graph(r, start_id + n1)

    if mismatch:
        end = start_id + n1 + r      # P2 Terminal.
        new = start_id + n1 + n2     # New leaves
        extra = torch.tensor([[end, new], [new, end]], dtype=torch.long)
        e2 = torch.cat([e2, extra.t()], dim=1)
        n2 = n2 + 1

    edge_index = torch.cat([e1, e2], dim=1)
    n_total = n1 + n2
    return edge_index, (root1, root2), n_total


class TNMDataset(InMemoryDataset):
    """Wrapping a Data into an InMemoryDataset for consistent use with other PyG datasets"""
    def __init__(self, data: Data, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self.collate([data])

    @property
    def num_node_features(self) -> int:
        return int(self[0].num_features)

    @property
    def num_classes(self) -> int:
        y = self[0].y
        y = y[y >= 0]
        return int(y.max().item()) + 1 if y.numel() > 0 else 0


def build_tnm_path_biggraph(
    r: int,
    num_train: int = 2000,
    num_val: int = 500,
    num_test: int = 1000,
    seed: int = 0,
    max_nodes_cap: int = 100_000,
    min_pairs_train: int = 200,
    min_pairs_val: int = 50,
    min_pairs_test: int = 100,
) -> InMemoryDataset:
    """
   Put many samples into one big (undirected) graph, labelling only the two "root nodes" of each sample:
        match -> label 1
        mismatch -> label 0
    """
    rng = np.random.RandomState(seed)

    # of nodes per pair ~ 2*(r+1) + 1 (one more leaf for mismatch)
    nodes_per_pair = 2 * (r + 1) + 1
    total_pairs = num_train + num_val + num_test

    # Scaled to the ceiling
    if nodes_per_pair * total_pairs > max_nodes_cap:
        scale = max_nodes_cap / float(nodes_per_pair * total_pairs)
        num_train = max(1, int(num_train * scale))
        num_val   = max(1, int(num_val   * scale))
        num_test  = max(1, int(num_test  * scale))

    # Forced lower limit
    num_train = max(min_pairs_train, num_train)
    num_val   = max(min_pairs_val,   num_val)
    num_test  = max(min_pairs_test,  num_test)

    parts, roots, labels = [], [], []
    nid = 0

    def _append_pair(mismatch: bool):
        nonlocal nid
        ei, (a, b), nadd = _make_path_pair(r, mismatch, start_id=nid)
        nid += nadd
        parts.append(ei)
        lab = 0 if mismatch else 1
        roots.extend([a, b])
        labels.extend([lab, lab]) 

    for _ in range(num_train): _append_pair(bool(rng.randint(0, 2)))
    for _ in range(num_val):   _append_pair(bool(rng.randint(0, 2)))
    for _ in range(num_test):  _append_pair(bool(rng.randint(0, 2)))

    edge_index = torch.cat(parts, dim=1)
    N = nid

    # Node features: root1 / root2 / other three types of one-hot
    x = torch.zeros((N, 3), dtype=torch.float)
    for i, rid in enumerate(roots):
        x[rid, i % 2] = 1.0
    x[:, 2] = (x.sum(dim=1) == 0).float()

    # labels and masks: labels only on the root node
    y = torch.full((N,), -100, dtype=torch.long)
    for rid, lab in zip(roots, labels):
        y[rid] = lab

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    ntr, nva = 2 * num_train, 2 * num_val
    train_nodes = roots[:ntr]
    val_nodes   = roots[ntr:ntr + nva]
    test_nodes  = roots[ntr + nva:]
    train_mask[train_nodes] = True
    val_mask[val_nodes]     = True
    test_mask[test_nodes]   = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = 2
    return TNMDataset(data)



def get_planetoid(dataset_name: str, data_dir: str = './data/', use_lcc: bool = False, seed: int = 12345) -> InMemoryDataset:
    assert dataset_name in ['Cora', 'Citeseer', 'Pubmed']
    dataset = Planetoid(data_dir, dataset_name)

    if use_lcc:
        lcc = get_largest_connected_component(dataset)
        x_new = dataset._data.x[lcc]
        y_new = dataset._data.y[lcc]

        row, col = dataset._data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size(0), dtype=torch.bool),
            test_mask=torch.zeros(y_new.size(0), dtype=torch.bool),
            val_mask=torch.zeros(y_new.size(0), dtype=torch.bool)
        )
        dataset._data = set_train_val_test_split(seed, data)

    return dataset


def get_webkb(dataset_name: str, data_dir: str = './data/', split: int = 0) -> InMemoryDataset:
    """Priority is given to reading the official preset split; if it is missing, the split is randomised 60/20/20 (and a reminder is printed)."""
    assert dataset_name in ['Cornell', 'Texas', 'Wisconsin']
    dataset = WebKB(data_dir, name=dataset_name)

    npz_path = os.path.join(data_dir, dataset_name, 'raw', f'{dataset_name}_split_0.6_0.2_{split}.npz')
    if os.path.exists(npz_path):
        splits = np.load(npz_path)
        dataset._data.train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool)
        dataset._data.val_mask   = torch.tensor(splits['val_mask'],   dtype=torch.bool)
        dataset._data.test_mask  = torch.tensor(splits['test_mask'],  dtype=torch.bool)
    else:
        # Fallback: random splitting to avoid FileNotFoundError interrupting experiments
        print(f"[WebKB] split file not found: {npz_path}. Falling back to random 60/20/20 split.", flush=True)
        data = dataset._data
        N = data.num_nodes
        perm = torch.randperm(N)
        n_tr = int(0.6 * N)
        n_va = int(0.2 * N)
        tr = perm[:n_tr]; va = perm[n_tr:n_tr + n_va]; te = perm[n_tr + n_va:]
        data.train_mask = torch.zeros(N, dtype=torch.bool); data.train_mask[tr] = True
        data.val_mask   = torch.zeros(N, dtype=torch.bool); data.val_mask[va] = True
        data.test_mask  = torch.zeros(N, dtype=torch.bool); data.test_mask[te] = True

    return dataset


def get_dataset(dataset_name: str, data_dir: str = './data/', use_lcc: bool = False,
                seed: int = 12345, split: int = 0) -> InMemoryDataset:
    """
    Unified entry:
      - 'TNM_tree_r{R}_b{B}_n{N}' -> small tree-pair synthetic (synthetic_tnm)
      - 'TNM_path_r{R}' -> path synthetic large image
      - Planetoid / WebKB -> real map data
    """
    name = dataset_name.strip()
    lower = name.lower()

    if lower.startswith('tnm_tree'):
        parts = name.split('_')
        r = _parse_int_token(parts, 'r', None)
        b = _parse_int_token(parts, 'b', 2)
        n = _parse_int_token(parts, 'n', 60)
        assert r is not None, 

        data = make_tnm_tree_pair_graph(
            r=r, b=b, num_pairs=n, mismatch_ratio=0.5,
            feature_dim=1, seed=seed
        )
        return TNMDataset(data)

    # --- TNM_path: benchmark for equal-precision/long-range dependencies ---
    if lower.startswith('tnm_path') or lower == 'tnm':
        # Compatible with 'TNM_path_r8' or 'TNM_r8' (default as path)
        parts = name.split('_')
        r = _parse_int_token(parts, 'r', 6)
        return build_tnm_path_biggraph(r=r, seed=seed)

    # --- Real map: Planetoid / WebKB ---
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        return get_planetoid(dataset_name, data_dir=data_dir, use_lcc=use_lcc, seed=seed)

    if dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        # split<0 random split id; otherwise use the specified split
        if split < 0:
            split = int(np.random.randint(0, 10))
        return get_webkb(dataset_name, data_dir=data_dir, split=split)

    raise Exception('Unknown dataset.')



def get_node_mapper(lcc: np.ndarray) -> dict:
    return {int(node): i for i, node in enumerate(lcc)}


def remap_edges(edges: list, mapper: dict) -> list:
    row = [mapper[int(e[0])] for e in edges]
    col = [mapper[int(e[1])] for e in edges]
    return [row, col]


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset._data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.add(current_node)
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [int(n) for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset._data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(max(comps, key=len)))


def set_train_val_test_split(
    seed: int,
    data: Data,
    num_development: int = 1500,
    num_per_class: int = 20
) -> Data:
    """Do a repeatable 20/20/remainder split when the dataset doesn't provide a standard mask."""
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    num_classes = int(data.y.max().item()) + 1

    # Pre-picked development sets
    development_idx = rnd_state.choice(num_nodes, min(num_development, num_nodes), replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    for c in range(num_classes):
        cls_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        take = min(num_per_class, len(cls_idx))
        if take > 0:
            train_idx.extend(rnd_state.choice(cls_idx, take, replace=False))

    val_idx = [int(i) for i in development_idx if i not in train_idx]

    def _mask(idx):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        if len(idx) > 0:
            m[torch.tensor(idx, dtype=torch.long)] = True
        return m

    data.train_mask = _mask(train_idx)
    data.val_mask   = _mask(val_idx)
    data.test_mask  = _mask(test_idx)
    return data
