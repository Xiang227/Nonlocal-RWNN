import torch
from torch_geometric.data import Data
import numpy as np
from collections import deque


def _build_bary_tree(side_root, depth, b, start_id, edges):
    """
    Starting from side_root, generate a b-cross tree with depth (not including the level of side_root itself).
    Returns: last_id (largest node id assigned), list of bottom (depth=depth) leaf nodes.
    """
    curr_level = [side_root]
    last_id = start_id
    for d in range(1, depth + 1):
        next_level = []
        for u in curr_level:
            children = []
            for _ in range(b):
                last_id += 1
                v = last_id
                edges.append([u, v])
                edges.append([v, u])  
                children.append(v)
            next_level.extend(children)
        curr_level = next_level
    return last_id, curr_level  


def make_tnm_tree_pair_graph(
    r: int,
    b: int = 2,
    num_pairs: int = 600,
    mismatch_ratio: float = 0.5,
    feature_dim: int = 1,
    seed: int = 1,
    marker_range=(0.8, 1.2),   
    feat_noise=0.02,           
    use_r_scaling=False,
):
    """
    match=one random leaf marking on each left and right side; mismatch=right side only.
    - Leaf position: randomly selected among the leaves at depth r;
    - Amplitude: marker_range uniform sampling;
    - Global noise: each node ~ N(0, feat_noise^2).
    - For use_r_scaling=True, multiply b***max(0,(r-4)/2) for amplitude (to compensate for only a portion of the depth decay).
    """
    rng = np.random.RandomState(seed)

    edges = []
    node_features = []
    node_labels = []
    node_train_mask = []
    node_val_mask = []
    node_test_mask = []

    nid = -1
    num_mismatch = int(round(num_pairs * mismatch_ratio))
    labels = np.array([1] * num_mismatch + [0] * (num_pairs - num_mismatch))
    rng.shuffle(labels)

    n_tr = int(0.6 * num_pairs)
    n_va = int(0.2 * num_pairs)

    r_scale = (b ** max(0, (r - 4) / 2.0)) if use_r_scaling else 1.0

    for i in range(num_pairs):
        nid += 1
        root = nid

        feat = rng.normal(loc=0.0, scale=feat_noise, size=(feature_dim,)).astype(np.float32)
        node_features.append(feat)

        y = labels[i]  # 1=mismatch, 0=match
        node_labels.append(y)

        m_tr = (i < n_tr)
        m_va = (n_tr <= i < n_tr + n_va)
        m_te = (i >= n_tr + n_va)
        node_train_mask.append(m_tr)
        node_val_mask.append(m_va)
        node_test_mask.append(m_te)

        # There is a b-tree of depth r on each side.
        nid, left_leaves  = _build_bary_tree(root, r, b, nid, edges)
        nid, right_leaves = _build_bary_tree(root, r, b, nid, edges)

        # Randomly selected leaves as signal source
        left_target  = int(rng.choice(left_leaves))
        right_target = int(rng.choice(right_leaves))

        # Extend the array and fill the new node with noise
        while len(node_features) <= nid:
            node_features.append(
                rng.normal(loc=0.0, scale=feat_noise, size=(feature_dim,)).astype(np.float32)
            )
            node_labels.append(0)
            node_train_mask.append(False)
            node_val_mask.append(False)
            node_test_mask.append(False)

        # Amplitude draw + optional scaling
        amp_left  = rng.uniform(*marker_range) * r_scale
        amp_right = rng.uniform(*marker_range) * r_scale

        # match = add both left and right; mismatch = add right side only
        if y == 0:
            node_features[left_target][0]  += amp_left
        node_features[right_target][0] += amp_right

    x = torch.from_numpy(np.stack(node_features, axis=0))
    y = torch.from_numpy(np.array(node_labels, dtype=np.int64))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=torch.tensor(node_train_mask, dtype=torch.bool),
        val_mask=torch.tensor(node_val_mask, dtype=torch.bool),
        test_mask=torch.tensor(node_test_mask, dtype=torch.bool),
    )
    return data

