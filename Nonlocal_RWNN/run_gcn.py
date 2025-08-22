import argparse, json, time, re
import torch, torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from pathlib import Path
from data import get_dataset
from synthetic_tnm import make_tnm_tree_pair_graph  # 新增：流式评测要用

# ----------------- Models -----------------
class GCN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, nlayer, dropout):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        if nlayer == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hid))
            for _ in range(nlayer-2):
                self.convs.append(GCNConv(hid, hid))
            self.convs.append(GCNConv(hid, out_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1:
                x = F.relu(x); x = self.drop(x)
        return F.log_softmax(x, dim=-1)

class GAT(nn.Module):
    def __init__(self, in_dim, hid, out_dim, nlayer, heads, dropout):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        if nlayer == 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, dropout=dropout, add_self_loops=True))
        else:
            self.convs.append(GATConv(in_dim, hid//heads, heads=heads, dropout=dropout, add_self_loops=True))
            for _ in range(nlayer-2):
                self.convs.append(GATConv(hid, hid//heads, heads=heads, dropout=dropout, add_self_loops=True))
            self.convs.append(GATConv(hid, out_dim, heads=1, dropout=dropout, add_self_loops=True))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1:
                x = F.elu(x); x = self.drop(x)
        return F.log_softmax(x, dim=-1)

# ----------------- Train/Eval -----------------
def train_eval(model, data, device, lr=3e-3, wd=5e-4, max_epoch=800, patience=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_test = 0., 0.
    no_improve = 0
    with tqdm(total=max_epoch, unit='ep', disable=False) as pbar:
        for ep in range(max_epoch):
            model.train(); opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=-1)
                tr = (pred[data.train_mask]==data.y[data.train_mask]).float().mean().item()*100
                va = (pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()*100
                te = (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()*100
            pbar.set_postfix(loss=f"{loss.item():.4f}", tr=f"{tr:.2f}", va=f"{va:.2f}", te=f"{te:.2f}"); pbar.update(1)

            if va > best_val:
                best_val, best_test, no_improve = va, te, 0
            else:
                no_improve += 1
                if no_improve >= patience: break
    return best_val, best_test

# ----------------- Streaming Eval for TNM_tree -----------------
def _parse_r_b_from_name(name: str, default_b=2):
    r = None; b = default_b
    m = re.search(r"r(\d+)", name.lower())
    if m: r = int(m.group(1))
    m = re.search(r"b(\d+)", name.lower())
    if m: b = int(m.group(1))
    if r is None:
        raise ValueError(f"Cannot parse r from dataset name: {name}")
    return r, b

@torch.no_grad()
def eval_stream_tnm_tree(model, feat_dim, device, dataset_name, rounds=50, pairs=100, base_seed=1000):
    """多次生成仅用于前向的小测试图，累计精度；只在根节点(y>=0)上计分。"""
    r, b = _parse_r_b_from_name(dataset_name)
    model.eval()
    hits, tots = 0, 0
    for t in range(rounds):
        data_te = make_tnm_tree_pair_graph(
            r=r, b=b, num_pairs=pairs, mismatch_ratio=0.5,
            feature_dim=feat_dim, seed=base_seed + t
        )
        data_te = data_te.to(device)
        out = model(data_te.x, data_te.edge_index)           # [N, C]
        pred = out.argmax(dim=-1)
        mask = (data_te.y >= 0)                              # 根节点
        hits += (pred[mask] == data_te.y[mask]).sum().item()
        tots += int(mask.sum().item())
    acc = 100.0 * hits / max(1, tots)
    return acc, hits, tots

# ----------------- Main -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", choices=["gcn","gat"], required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_epoch", type=int, default=800)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    # 新增：流式评测参数
    ap.add_argument("--eval_stream", action="store_true",
                    help="Use streaming evaluation on TNM_tree (generate fresh test graphs).")
    ap.add_argument("--eval_pairs", type=int, default=100,
                    help="Pairs per streaming-eval round.")
    ap.add_argument("--eval_rounds", type=int, default=50,
                    help="Number of streaming-eval rounds.")
    ap.add_argument("--eval_seed", type=int, default=1000,
                    help="Base seed for streaming eval.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = get_dataset(args.dataset, data_dir="./data", split=0, seed=args.seed)
    data = ds[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    out_dim = int(data.y.max().item()) + 1
    if args.model == "gcn":
        net = GCN(data.num_features, args.hidden, out_dim, args.layers, args.dropout)
    else:
        net = GAT(data.num_features, args.hidden, out_dim, args.layers, args.heads, args.dropout)
    net = net.to(device)

    bv, bt = train_eval(net, data, device, lr=args.lr, wd=args.wd,
                        max_epoch=args.max_epoch, patience=args.patience)
    print(f"Best val {bv:.2f} | Best test {bt:.2f}")

    # Streaming review: enabled only for TNM_tree data
    if args.eval_stream and args.dataset.lower().startswith("tnm_tree"):
        acc, hits, tots = eval_stream_tnm_tree(
            net, feat_dim=data.num_features, device=device, dataset_name=args.dataset,
            rounds=args.eval_rounds, pairs=args.eval_pairs, base_seed=args.eval_seed
        )
        print(f"[Streaming-Eval] test_acc={acc:.2f}% over {tots} roots (hits={hits})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(dataset=args.dataset, model=args.model, layers=args.layers,
                           hidden=args.hidden, dropout=args.dropout, seed=args.seed,
                           best_val_acc=bv, best_test_acc=bt, time=time.time()), f)
