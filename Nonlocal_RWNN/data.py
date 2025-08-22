
'''
Code adapted from https://github.com/twitter-research/graph-neural-pde/blob/main/src/data.py
'''

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, ZINC, HeterophilousGraphDataset, \
    WikipediaNetwork, Twitch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch

from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import os



# TODO: ZINC
def get_dataset(dataset_name, data_dir='./data/', use_lcc=False, seed=12345, split=0) -> InMemoryDataset:
    if split < 0:
        split = np.random.randint(0, 10)
    if (dataset_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS', 'ogbn-arxiv']
            or dataset_name.startswith('ogbn-arxiv')):
        return get_homo_dataset(dataset_name, data_dir, use_lcc, seed)
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        return get_webkb_dataset(dataset_name, data_dir, split=split)
    elif dataset_name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions', 'Chameleon',
                          'Crocodile', 'Squirrel']:
        return get_other_dataset(dataset_name, data_dir, split=split)
    elif dataset_name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']:
        return get_twitch_dataset(dataset_name, data_dir)
    else:
        raise Exception('Unknown dataset.')


def get_homo_dataset(dataset_name, data_dir='./data/', use_lcc=False, seed=12345) -> InMemoryDataset:
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(data_dir, dataset_name)
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(data_dir, dataset_name)
    elif dataset_name == 'CoauthorCS':
        dataset = Coauthor(data_dir, 'CS')
    elif dataset_name.startswith('ogbn-arxiv'):
        import os, shutil, torch, re
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import subgraph  
        from torch_geometric.data import Data

        base_name = 'ogbn-arxiv'  

        # ---- Allow PyG type-safe deserialisation (compatible with old processed caches) ----
        try:
            from torch.serialization import add_safe_globals
            try:
                from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage, TensorAttr
                add_safe_globals([GlobalStorage, NodeStorage, EdgeStorage, TensorAttr])
            except Exception:
                pass
            try:
                from torch_geometric.data.data import Data as _PyGData, DataEdgeAttr, DataTensorAttr
                add_safe_globals([_PyGData, DataEdgeAttr, DataTensorAttr])
            except Exception:
                pass
        except Exception:
            pass

        # Temporarily turn off weights_only in torch.load for compatibility with the old cache.
        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _orig_load(*args, **kwargs)

        try:
            torch.load = _patched_load
            dataset = PygNodePropPredDataset(name=base_name, root=data_dir)  
        except Exception as e:
            print(f"[{dataset_name}] safe-load failed: {e}; reprocess...", flush=True)
            proc = os.path.join(data_dir, base_name, "processed")
            shutil.rmtree(proc, ignore_errors=True)
            for aux in ("pre_transform.pt", "pre_filter.pt"):
                p = os.path.join(data_dir, base_name, aux)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except:
                        pass
            dataset = PygNodePropPredDataset(name=base_name, root=data_dir)
        finally:
            torch.load = _orig_load

        # ---- Take the raw data and make sure there is an edge_index with boolean masks ----
        raw = dataset._data
        N = raw.num_nodes
        ei = raw.edge_index
        if (ei is None) and hasattr(raw, "adj_t"):
            row, col, _ = raw.adj_t.t().coo()  
            ei = torch.stack([row, col], dim=0)
        if ei is None:
            raise RuntimeError("ogbn-arxiv: missing edge_index/adj_t")

        y = raw.y.view(-1).long()
        split = dataset.get_idx_split()
        train_idx = split["train"];
        val_idx = split["valid"];
        test_idx = split["test"]

        # ---- Parsing subgraph scale (e.g. ogbn-arxiv-sub20 => 20%) ----
        sub_rate = 1.0
        if dataset_name != base_name:
            m = re.search(r"sub(\d+)", dataset_name)
            if m:
                sub_rate = max(0.05, min(1.0, int(m.group(1)) / 100.0))

        # ---- Proportionally sampled induced subgraphs (no to_undirected, saves memory; self to_undirected if needed) ----
        if sub_rate < 1.0:
            torch.manual_seed(0)
            keep = torch.randperm(N)[: int(N * sub_rate)]
            keep = keep.sort().values
            ei, _ = subgraph(keep, ei, relabel_nodes=True, num_nodes=N)
            mapping = -torch.ones(N, dtype=torch.long)
            mapping[keep] = torch.arange(keep.numel())

            def _map(idx):
                mid = mapping[idx]
                return mid[mid >= 0]

            train_idx = _map(train_idx);
            val_idx = _map(val_idx);
            test_idx = _map(test_idx)
            x = raw.x[keep];
            y = y[keep];
            N = keep.numel()
        else:
            x = raw.x

        train_mask = torch.zeros(N, dtype=torch.bool);
        train_mask[train_idx] = True
        val_mask = torch.zeros(N, dtype=torch.bool);
        val_mask[val_idx] = True
        test_mask = torch.zeros(N, dtype=torch.bool);
        test_mask[test_idx] = True

        dataset._data = Data(x=x, edge_index=ei, y=y,
                             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

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
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset._data = data

    train_mask_exists = True
    try:
        dataset._data.train_mask
    except AttributeError:
        train_mask_exists = False

    if dataset_name == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset._data.edge_index)
        data = Data(
            x=dataset._data.x,
            edge_index=ei,
            y=dataset._data.y,
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid'])
        dataset._data = data
        train_mask_exists = True

    if use_lcc or not train_mask_exists:
        dataset._data = set_train_val_test_split(
            seed,
            dataset._data,
            num_development=5000 if dataset_name == 'CoauthorCS' else 1500)

    return dataset


def get_webkb_dataset(dataset_name, data_dir='./data/', split=0):
    """WebKB Heterogeneous Small Maps: If the 60/20/20 delineation file does not exist, it is generated and saved on-site on a hierarchical basis."""
    assert dataset_name in ['Cornell', 'Texas', 'Wisconsin']

    dataset = WebKB(data_dir, name=dataset_name)

    # Target division file path
    split_path = os.path.join(
        data_dir, dataset_name, "raw",
        f"{dataset_name}_split_0.6_0.2_{split}.npz"
    )

    # If not present, generate a stratified 60/20/20 division and save it
    if not os.path.exists(split_path):
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        y = dataset._data.y.cpu().numpy()
        rng = np.random.RandomState(split + 12345)

        num_nodes = len(y)
        classes = np.unique(y)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask   = np.zeros(num_nodes, dtype=bool)
        test_mask  = np.zeros(num_nodes, dtype=bool)

        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_tr = max(1, int(round(0.6 * n)))
            n_va = max(1, int(round(0.2 * n)))
            n_te = max(1, n - n_tr - n_va) if n >= 3 else max(0, n - n_tr - n_va)

            # Callback to test/val in case of overrun
            while n_tr + n_va + n_te > n:
                if n_te > 1: n_te -= 1
                elif n_va > 1: n_va -= 1
                else: n_tr -= 1

            train_mask[idx[:n_tr]] = True
            val_mask[idx[n_tr:n_tr+n_va]] = True
            test_mask[idx[n_tr+n_va:n_tr+n_va+n_te]] = True

        np.savez(split_path,
                 train_mask=train_mask,
                 val_mask=val_mask,
                 test_mask=test_mask)

    # Read divisions (either just generated or existing)
    splits_file = np.load(split_path)
    train_mask = splits_file['train_mask']
    val_mask   = splits_file['val_mask']
    test_mask  = splits_file['test_mask']

    # applied to the data, and uniformly undirected (usually reviewed by undirected）
    dataset._data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    dataset._data.val_mask   = torch.tensor(val_mask,   dtype=torch.bool)
    dataset._data.test_mask  = torch.tensor(test_mask,  dtype=torch.bool)
    dataset._data.edge_index = to_undirected(dataset._data.edge_index)

    return dataset



def get_other_dataset(dataset_name, data_dir='./data/', split=0):
    if dataset_name == 'ZINC':
        train_set = ZINC(data_dir, subset=True, split='train')
        val_set   = ZINC(data_dir, subset=True, split='train')
        test_set  = ZINC(data_dir, subset=True, split='train')
        return train_set, val_set, test_set

    elif dataset_name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
        dataset = HeterophilousGraphDataset(data_dir, name=dataset_name)

    elif dataset_name in ['Chameleon', 'Crocodile', 'Squirrel']:
        dataset = WikipediaNetwork(data_dir, name=dataset_name)

    else:
        raise Exception('Unknown dataset.')

    # These datasets typically provide multiple splitting masks of [N, num_splits]
    # Here, the split column is chosen; for one-dimensionality, it's straightforward to use the
    for key in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(dataset._data, key)
        if mask.dim() == 2:
            num_splits = mask.size(1)
            if split < 0 or split >= num_splits:
                raise ValueError(f"{dataset_name}: split={split} out of range [0,{num_splits-1}]")
            setattr(dataset._data, key, mask[:, split])

    # uniform anisotropy
    dataset._data.edge_index = to_undirected(dataset._data.edge_index)
    return dataset



def get_twitch_dataset(dataset_name, data_dir='./data/'):
    assert dataset_name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']

    dataset = Twitch(data_dir, name=dataset_name)
    num_nodes = dataset._data.num_nodes
    perm = torch.randperm(num_nodes)

    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    dataset._data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset._data.train_mask[train_idx] = True

    dataset._data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset._data.val_mask[val_idx] = True

    dataset._data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset._data.test_mask[test_idx] = True
    return dataset


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset._data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
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
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data
