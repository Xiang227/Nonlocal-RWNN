import os, json, time, copy
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GCNConv, GATConv

from data import get_dataset
from utils import EarlyStopping, set_seed


class GCNNet(nn.Module):
    def __init__(self, nin, nhid, nout, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 1
        layers = []
        in_dim = nin
        for _ in range(num_layers - 1):
            layers.append(GCNConv(in_dim, nhid))
            in_dim = nhid
        self.convs = nn.ModuleList(layers)
        self.dec_in_dim = in_dim
        self.dec = nn.Linear(self.dec_in_dim, nout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.dec(x)
        return out, x  


class GATNet(nn.Module):
    def __init__(self, nin, nhid, nout, num_layers=2, dropout=0.5, heads=8):
        super().__init__()
        assert num_layers >= 1
        layers = []
        in_dim = nin
        for li in range(num_layers - 1):
            layers.append(GATConv(in_dim, nhid, heads=heads, concat=False, dropout=dropout))
            in_dim = nhid
        self.convs = nn.ModuleList(layers)
        self.dec_in_dim = in_dim
        self.dec = nn.Linear(self.dec_in_dim, nout)
        self.dropout = dropout
        self.heads = heads

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.dec(x)
        return out, x  


@torch.no_grad()
def dirichlet_energy(h_node, edge_index):
    """
    h_node: [N, D] Node Embedding（penultimate）
    edge_index: [2, E]
    """
    ei0, ei1 = edge_index[0].long(), edge_index[1].long()
    diff = h_node[ei0] - h_node[ei1]
    return (diff * diff).sum(dim=1).mean().item()


def run(args):
    torch.cuda.empty_cache()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.seed < 0:
        args.seed = np.random.randint(0, 1_000_000)
    set_seed(args.seed)

    dataset = get_dataset(args.dataset, data_dir=args.data_dir, split=args.split, seed=args.seed)
    data = dataset[0].to(device)

    if args.model.lower() == "gcn":
        model = GCNNet(nin=dataset.num_node_features,
                       nhid=args.nhid, nout=dataset.num_classes,
                       num_layers=args.num_layers, dropout=args.dropout)
    elif args.model.lower() == "gat":
        model = GATNet(nin=dataset.num_node_features,
                       nhid=args.nhid, nout=dataset.num_classes,
                       num_layers=args.num_layers, dropout=args.dropout,
                       heads=args.heads)
    else:
        raise ValueError("model must be gcn|gat")
    model.to(device)

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  patience=args.lr_patience, factor=args.lr_decay_factor)

    early = EarlyStopping(patience=args.patience)

    best_val, best_test, best_E = 0.0, 0.0, float("nan")

    disable_tqdm = __name__ != "__main__"
    with tqdm(total=args.max_epoch, unit='epoch', disable=disable_tqdm) as bar:
        for epoch in range(args.max_epoch):
            # ---- train ----
            model.train()
            optimizer.zero_grad()
            out, _ = model(data)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # ---- eval ----
            model.eval()
            with torch.no_grad():
                out, penult = model(data)
                pred = out.argmax(dim=1)

                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item() * 100
                val_acc   = (pred[data.val_mask]   == data.y[data.val_mask]).float().mean().item() * 100
                test_acc  = (pred[data.test_mask]  == data.y[data.test_mask]).float().mean().item() * 100

                E = dirichlet_energy(penult, data.edge_index)  

            lr = optimizer.param_groups[0]['lr']

            bar.set_postfix(dict(
                loss=f"{loss.item():.4f}", train=f"{train_acc:.2f}",
                val=f"{val_acc:.2f}", test=f"{test_acc:.2f}", lr=f"{lr:.1e}"
            ))
            bar.update(1)

            if val_acc > best_val:
                best_val, best_test, best_E = val_acc, test_acc, E

            if early([-val_acc]):
                break

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            dataset=args.dataset,
            model=args.model.lower(),
            L=args.num_layers,      
            nhid=args.nhid,
            dropout=args.dropout,
            seed=args.seed,
            best_val_acc=best_val,
            best_test_acc=best_test,
            dirichlet=best_E,
            time=time.time(),
        )
        with open(args.out, "w") as f:
            json.dump(payload, f)
        print(f"Saved to {args.out}")

    return best_val, best_test, best_E


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--split", type=int, default=-1)
    p.add_argument("--seed", type=int, default=-1)

    p.add_argument("--model", type=str, choices=["gcn", "gat"], default="gcn")
    p.add_argument("--num_layers", type=int, default=2)   
    p.add_argument("--nhid", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--heads", type=int, default=8)        

    p.add_argument("--optimizer", type=str, default="Adam")
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    p.add_argument("--max_epoch", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--lr_decay_factor", type=float, default=0.1)
    p.add_argument("--lr_patience", type=int, default=500000)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)

    args = p.parse_args()
    run(args)
