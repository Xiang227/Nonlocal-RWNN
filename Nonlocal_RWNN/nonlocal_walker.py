
import torch
import numpy as np
import networkx as nx
from walker import Walker
from torch_geometric.utils import to_networkx



class NonLocalWalker(Walker):
    """A Walker that follows the non‑local transition matrix P_α from NPR.

    Parameters
    ----------
    edge_index : torch.LongTensor [2, E]
        Graph connectivity.
    num_nodes : int
        Number of nodes in the graph.
    alpha : float, optional
        Decay parameter α (default = 1.0).
    teleport_c : float, optional
        Teleport probability c in (0, 1).
        The larger c is, the more it relies on this non-local kernel. (default = 0.15)
    mode : {"exp", "power"}
        If "exp" uses   w_ij = exp(−α·d(i,j)).
        If "power" (≡ Lévy) uses w_ij = d(i,j)⁻α.
    max_distance : int, optional
        Shortest‑path cut‑off when approximating the all‑pairs distance
        matrix. For Cora and similar citation graphs a value of 4 is
        sufficient. (default = 4)
    device : torch.device, optional
        Where to keep the transition matrix. If not given, will inherit
        from ``edge_index``.
    topk : int or None, optional
        Retain only the top‑k outgoing probabilities for memory
        efficiency. If ``None`` keeps the full dense transition matrix.
    """

    def __init__(self,
                 edge_index: torch.Tensor,
                 num_nodes: int = None,
                 *,
                 alpha: float = 1.0,
                 teleport_c: float = 0.15,
                 mode: str = "exp",
                 max_distance: int | None = None,
                 device: torch.device | None = None,
                 topk: int | None = None):

        super().__init__(edge_index, num_nodes)
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1

        self.alpha = alpha
        self.c = teleport_c
        self.mode = mode
        self.max_distance = max_distance
        self.topk = topk
        self.device = device or edge_index.device

        # Pre‑compute the non‑local transition matrix P_α  (num_nodes × num_nodes)
        with torch.no_grad():
            # 1) Full map distance - already done in _compute_distance_matrix()
            dist = self._compute_distance_matrix()

            # 2) Distance → weight
            if self.mode == "exp":
                if self.max_distance is not None:
                    dist[dist > self.max_distance] = np.inf
                dist = np.where(dist < 1, 1.0, dist)
                W = np.exp(-self.alpha * (dist - 1))
                W[np.isinf(dist)] = 0
            else:  # "power" / "levy"
                W = np.power(np.maximum(dist, 1.0), -self.alpha)  # avoid 0^-α

            # ∞ Zero weighting at distance
            W[np.isinf(dist)] = 0

            # 3) Renormalise the self-loop weights.
            np.fill_diagonal(W, 1e-6)  #
            row_sums = W.sum(axis=1, keepdims=True)  # shape (N,1)，Calculate rows sums
            dangling = (row_sums == 0)  # Finding the zero row.
            if dangling.any():
                W[dangling.squeeze(), :] = 1.0 / W.shape[0]  # Direct to uniform Probability 1/N
            row_sums = W.sum(axis=1, keepdims=True)  # Recalculate the row sums once (all row sums > 0)）
            # W = W / row_sums        # row normalisation
            D = np.diagflat((1.0 / row_sums).squeeze())  # shape (N,N)
            W = D @ W

            # 4) Teleportation mixing (Eq. 3.2): G_alpha = c * P_tilde_alpha + (1 - c) * 11^T/n
            # the larger the c the more NON-LOCAL (i.e., follow W)
            n = W.shape[0]
            J = np.full((n, n), 1.0 / n, dtype=W.dtype)  # uniform teleportation matrix
            P = self.c * W + (1.0 - self.c) * J

            walk_probs_np = P.astype(np.float32)

            # # 5) top-k sparse (optional)
            # if self.topk is not None and self.topk < self.num_nodes:
            #     idx = np.argpartition(-walk_probs_np, self.topk, axis=1)[:, :self.topk]
            #     keep = np.zeros_like(walk_probs_np, dtype=bool)
            #     rows = np.arange(self.num_nodes)[:, None]
            #     keep[rows, idx] = True
            #     # Retention of teleport elements
            #     tele_mask = np.isclose(P, (1.0 - self.c) / self.num_nodes, atol=1e-8)
            #     keep |= tele_mask
            #     walk_probs_np = np.where(keep, walk_probs_np, 0.0)
            #     row = walk_probs_np.sum(axis=1, keepdims=True)
            #     row[row == 0] = 1.0
            #     walk_probs_np = walk_probs_np / row

            self.walk_probs = torch.from_numpy(walk_probs_np).to(self.device)

        # print("DEBUG  Pα  | rows with ≤10 non-zeros:",
        #       (self.walk_probs.to('cpu') > 0).sum(1).le(10).sum().item(),
        #       "/", self.num_nodes)

    # Sampling
    def uniform_random_walk(self, start_nodes: torch.Tensor, walk_length: int,
                            num_walks_per_node: int = 1):  # type: ignore[override]
        """Overrides the base class to follow non‑local probabilities.
        Keeps the output tensor shape identical to RUM's expectation:
        (num_walks_per_node, |start_nodes|, walk_length+1).
        """
        assert hasattr(self, "walk_probs"), "walk_probs was not initialised – check __init__"
        start_nodes = torch.unique(start_nodes).to(self.device)
        start_nodes = start_nodes.repeat(num_walks_per_node)
        num_walks = start_nodes.size(0)

        paths = torch.full((num_walks, walk_length + 1), -1, dtype=torch.long, device=self.device)
        current = start_nodes.clone()
        paths[:, 0] = current

        if walk_length == 0:
            return paths.view(num_walks_per_node, -1, walk_length + 1)

        for step in range(1, walk_length + 1):
            probs = self.walk_probs[current]  # shape [num_walks, N]
            next_nodes = torch.multinomial(probs, 1).squeeze(1)
            paths[:, step] = next_nodes
            current = next_nodes

        return paths.view(num_walks_per_node, -1, walk_length + 1)

    def _compute_distance_matrix(self):
        """All-pairs unweighted shortest-path up to `max_distance`."""

        # 1. (art) composition
        src, dst = self.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(zip(src, dst))

        # 2. initialisation
        D = np.full((self.num_nodes, self.num_nodes),
                    fill_value=np.inf,
                    dtype=np.float32)
        np.fill_diagonal(D, 0)

        # 3. Full node BFS
        for s, lengths in nx.all_pairs_shortest_path_length(G):
            for t, d in lengths.items():
                D[s, t] = d

        #  4. disk drive caching
        cache = f"dist_{self.num_nodes}_{self.max_distance}.npy"
        np.save(cache, D)
        return D
