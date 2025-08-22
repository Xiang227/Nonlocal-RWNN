
from model import RUMModel
from utils import EarlyStopping
from data import get_dataset
# from walker import Walker
from utils import set_seed

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch
from ray import tune

from tqdm import tqdm
import numpy as np
import copy

from pathlib import Path
import json
import time


def run(args):
    def _to_node_embeddings(h, num_nodes, num_samples=None):
        """
        Regularises the tensor caught by hook to [N, D], doing read-only operations without affecting the gradient.
        Several input shapes allowed:
          [K, N, D] -> average along K to get [N, D]
          [N, K, D] -> average along K
          [N, D] -> return as is
          [K*N, D] -> reshape into [K, N, D] then average along K (requires num_samples)
        """
        if h is None:
            return None
        h = h.detach()  

        if h.dim() == 2:
            
            if h.size(0) == num_nodes:
                return h
            if h.size(1) == num_nodes:
                return h.t()

        if h.dim() == 3:
            K, A, B = h.size(0), h.size(1), h.size(2)
            if A == num_nodes:  # [K, N, D]
                return h.mean(0)
            if K == num_nodes:  # [N, K, D]
                return h.mean(1)

        if h.dim() == 2 and num_samples is not None and h.size(0) == num_samples * num_nodes:
            return h.view(num_samples, num_nodes, -1).mean(0)

        raise RuntimeError(f"Unexpected penultimate shape {tuple(h.shape)} for num_nodes={num_nodes}")

    def _dirichlet_energy(h_node, edge_index_cpu):
        # h_node: [N, D] on CPU; edge_index_cpu: [2, E] on CPU
        ei0, ei1 = edge_index_cpu[0].long(), edge_index_cpu[1].long()
        diff = h_node[ei0] - h_node[ei1]
        return (diff * diff).sum(dim=1).mean().item()


    torch.cuda.empty_cache()

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seed < 0:
        args.seed = np.random.randint(0, 1000000)

    dataset = get_dataset(args.dataset, data_dir=args.data_dir, split=args.split, seed=args.seed)
    data = dataset[0].to(args.device)
    

    if args.walker == 'uniform':
        from walker import Walker
        walker = Walker(data.edge_index, num_nodes=data.num_nodes)
    else:
        from nonlocal_walker import NonLocalWalker
        walker = NonLocalWalker(
            data.edge_index,
            num_nodes=data.num_nodes,
            alpha=args.alpha,
            teleport_c=args.c,
            mode=args.walk_mode,
            max_distance=args.max_dist,
            # topk=args.topk,
            device=args.device,
        )

    set_seed(args.seed)  # seed should be set after loading dataset

    directed = False  # TODO: test whether pyg dataset is directed or not
    if directed:
        _g = copy.deepcopy(g)
        g = dgl.to_bidirected(g, copy_ndata=True)
        src, dst = g.edges()
        has_fwd_edge = _g.has_edges_between(src.flatten(), dst.flatten()) * 1.0
        has_bwd_edge = _g.has_edges_between(dst.flatten(), src.flatten()) * 1.0
        has_edge = has_fwd_edge - has_bwd_edge
        e = has_edge.unsqueeze(-1)
        e.to(args.device)
    else:
        e = None

    model = RUMModel(
        nin=dataset.num_node_features,
        nhid=args.nhid,
        nout=dataset.num_classes,
        nlayer=args.nlayer,
        num_samples=args.num_samples,
        length=args.length,
        temperature=args.consistency_temperature,
        dropout=args.dropout,
        rnn_nlayer=args.rnn_nlayer,
        self_supervise_weight=args.self_supervise_weight,
        consistency_weight=args.consistency_weight,
        activation=getattr(torch.nn, args.activation)(),
        edge_features=e.shape[-1] if e is not None else 0,
        walker=walker,
    )
    model.to(args.device)
    edge_index_cpu = data.edge_index.cpu()

    penult = {"h": None}

    def _grab_penultimate(module, inputs):
        penult["h"] = inputs[0].detach().cpu()   

    handle = model.dec.register_forward_pre_hook(_grab_penultimate)

    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=args.lr_patience,
        factor=args.lr_decay_factor,
    )

    early_stopping = EarlyStopping(patience=args.patience)

    best_val_acc, best_test_acc = 0, 0
    train_accs, val_accs, test_accs = [], [], []
    best_dirichlet = float("nan")
    disable_tqdm = __name__ != '__main__'
    eval_every = max(1, args.eval_every)


    with tqdm(total=args.max_epoch, unit='epoch', disable=disable_tqdm) as pbar:
        for i in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            out, loss = model(data, e=e)
            out = out.mean(
                0).log()  # My understanding is it averages the classificaiton over walks. Log is for computing nll loss.
            classification_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss += classification_loss
            loss.backward()
            optimizer.step()

            do_eval = (i % eval_every == 0) or (i == args.max_epoch - 1)
            if do_eval:
                model.eval()
                with torch.no_grad():
                    out, _ = model(data, e=e)
                    out = out.mean(0)
                    # Regularise to [N, D] and recalculate the energy - the key correction.
                    h_node = _to_node_embeddings(penult["h"], num_nodes=data.num_nodes, num_samples=args.num_samples)
                    dirichlet = _dirichlet_energy(h_node, edge_index_cpu)

                    train_acc = (out.argmax(dim=1)[data.train_mask] == data.y[data.train_mask]).float().mean().item() * 100
                    val_acc = (out.argmax(dim=1)[data.val_mask] == data.y[data.val_mask]).float().mean().item() * 100
                    # val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
                    test_acc = (out.argmax(dim=1)[data.test_mask] == data.y[data.test_mask]).float().mean().item() * 100

            else:
                train_acc = train_accs[-1] if train_accs else 0.0
                val_acc = val_accs[-1] if val_accs else 0.0
                test_acc = test_accs[-1] if test_accs else 0.0
                dirichlet = best_dirichlet

            penult["h"] = None
            torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]['lr']
            # scheduler.step(val_acc)

            pbar.set_postfix({
                'train_loss': f'{loss.item():.4f}',
                'train_acc': f'{train_acc:.2f}',
                'val_acc': f'{val_acc:.2f}',
                # 'val_loss': f'{val_loss:.4f}',
                'test_acc': f'{test_acc:.2f}',
                'lr': f'{lr:.1e}'
            })
            pbar.update(1)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            if do_eval and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_dirichlet = dirichlet

            if early_stopping([-val_acc]):
                break

        if args.in_tune_session:
            tune.report(dict(
                epoch=i - args.patience,
                val_acc=best_val_acc,
                test_acc=best_test_acc,
                done=True,
            ))

    if not disable_tqdm:
        print(f'Best validation accuracy: {best_val_acc:.2f}, Best test accuracy: {best_test_acc:.2f}', flush=True)

    # return [best_val_acc, best_test_acc], [train_accs, val_accs, test_accs]
    def save_result(args, best_val_acc, best_test_acc, i):
        if getattr(args, "out", None):
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset": args.dataset,
                "walker": args.walker,
                "K": args.num_samples,
                "L": args.length,
                "nhid": args.nhid,
                "nlayer": args.nlayer,
                "dropout": args.dropout,
                "seed": args.seed,
                "best_val_acc": best_val_acc,
                "best_test_acc": best_test_acc,
                "epoch_last": int(i),
                "time": time.time(),
                "dirichlet": best_dirichlet,
            }
            with open(args.out, "w") as f:
                json.dump(payload, f)

    save_result(args, best_val_acc, best_test_acc, i)
    if not disable_tqdm:
        print(f'Best validation accuracy: {best_val_acc:.2f}, Best test accuracy: {best_test_acc:.2f}', flush=True)
    handle.remove()
    return [best_val_acc, best_test_acc], [train_accs, val_accs, test_accs]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store the dataset')
    parser.add_argument('--split', type=int, default=-1, help='Dataset split')
    parser.add_argument('--seed', type=int, default=-1)

    parser.add_argument('--nhid', type=int, default=64, help='Number of features of the encoded walks')
    parser.add_argument('--nlayer', type=int, default=1, help='Number of RW layers, typically 1')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of sampled walk per starting node')
    parser.add_argument('--length', type=int, default=8,
                        help='Length of walks, starting node excluded, and thus one node longer than RUM')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--rnn_nlayer', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--activation', type=str, default='SiLU', help='Activation function')
    # parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for consistency loss') # to my best understanding, unused argument
    parser.add_argument('--self_supervise_weight', type=float, default=1.0, help='Weight for self-supervised loss')
    parser.add_argument('--consistency_weight', type=float, default=1, help='Weight for consistency loss')
    parser.add_argument('--consistency_temperature', type=float, default=0.5, help='Temperature for consistency loss')
    parser.add_argument('--directed', type=int, default=0)

    parser.add_argument('--optimizer', type=str, default='Adam', help='torch.nn.optim optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10, help='Weight decay')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor for learning rate reduction')
    # large value to disable
    parser.add_argument('--lr_patience', type=int, default=500000)

    parser.add_argument('--in_tune_session', action='store_true', help='Whether running in Ray Tune session')
    parser.add_argument('--device', type=str, default=None)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=0.15)
    parser.add_argument('--walk_mode', choices=['exp', 'power'], default='exp')
    parser.add_argument('--max_dist', default=None, type=lambda x: None if x.lower() == 'none' else int(x),
                        help="maximum shortest‐path distance to consider; ""use an integer or 'None' for no truncation"
                        )
    parser.add_argument(
        '--walker',
        choices=['uniform', 'nonlocal'],
        default='nonlocal',
        help='uniform: Raw uniform random walker；nonlocal: Non-local random walker'
    )
    parser.add_argument('--repeats', type=int, default=3,
                        help='Run the same hyperparameters this many times (default: 3)')
    parser.add_argument('--out', type=str, default=None, help='Path to save JSON metrics for this run')
    # ---- TNM options (optional) ----
    parser.add_argument('--tnm_r', type=int, default=6)
    parser.add_argument('--tnm_branching', type=int, default=3)
    parser.add_argument('--tnm_ntrain', type=int, default=2000)
    parser.add_argument('--tnm_nval', type=int, default=500)
    parser.add_argument('--tnm_ntest', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1)

    args = parser.parse_args()
    best_acc, accs = run(args)
