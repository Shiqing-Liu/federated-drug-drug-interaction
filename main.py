"""FedAvg vs centralized vs local-only GNN link prediction on ogbl-ddi.

Usage:
  python main.py --mode centralized --encoder sage
  python main.py --mode federated  --encoder sage --num_clients 4
  python main.py --mode federated  --encoder gcn  --num_clients 4
  python main.py --mode local      --encoder sage --num_clients 4
"""
import argparse
import copy
import csv
import os
import random

import numpy as np
import torch

from data import load_ddi, build_edge_index, shard_train_edges
from model import LinkPredModel
from engine import train_epochs, evaluate, fedavg

CSV_FIELDS = [
    "round", "loss",
    "valid_hits@10", "valid_hits@20", "valid_hits@30",
    "test_hits@10", "test_hits@20", "test_hits@30",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def csv_path(args):
    if args.mode == "centralized":
        name = f"centralized_{args.encoder}_seed{args.seed}.csv"
    else:
        name = f"{args.mode}_{args.encoder}_K{args.num_clients}_seed{args.seed}.csv"
    return os.path.join(args.out, name)


def is_complete(path, rounds):
    """A run is complete iff its CSV holds at least `rounds` data rows."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        n = sum(1 for _ in f) - 1  # minus header
    return n >= rounds


class CsvLogger:
    """Writes the header then one row per round, flushing each round so a
    Colab disconnect leaves a valid partial CSV."""

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS)
        self.w.writeheader()
        self.f.flush()

    def log(self, row):
        self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def fmt(row):
    return (f"round {row['round']:02d} | loss {row['loss']:.4f} | "
            f"valid H@10/20/30 {row['valid_hits@10']:.4f}/{row['valid_hits@20']:.4f}/"
            f"{row['valid_hits@30']:.4f} | "
            f"test H@10/20/30 {row['test_hits@10']:.4f}/{row['test_hits@20']:.4f}/"
            f"{row['test_hits@30']:.4f}")


def make_model(args, num_nodes, device):
    return LinkPredModel(num_nodes, hidden=args.hidden, encoder=args.encoder).to(device)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_centralized(args, num_nodes, split_edge, evaluator, device, logger):
    """Full train graph, single model, rounds*local_epochs epochs total,
    evaluated every local_epochs epochs so the x-axis aligns with FL rounds."""
    train_edges = split_edge["train"]["edge"]
    mp_edge_index = build_edge_index(train_edges, num_nodes)

    model = make_model(args, num_nodes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for r in range(args.rounds):
        loss = train_epochs(model, optimizer, train_edges, mp_edge_index, num_nodes,
                            args.batch_size, args.local_epochs, device, args.max_batches)
        res = evaluate(model, mp_edge_index, split_edge, evaluator, num_nodes, device)
        row = {"round": r, "loss": loss, **res}
        logger.log(row)
        print(fmt(row), flush=True)


def run_federated(args, num_nodes, split_edge, evaluator, device, logger):
    """K clients, each with its own disjoint shard + own message-passing graph.
    FedAvg every round; evaluate the global model on the full train graph."""
    shards = shard_train_edges(split_edge["train"]["edge"], args.num_clients, args.seed)
    client_mp = [build_edge_index(s, num_nodes) for s in shards]
    client_weights = [s.size(0) for s in shards]
    full_mp = build_edge_index(split_edge["train"]["edge"], num_nodes)

    global_model = make_model(args, num_nodes, device)
    client_model = make_model(args, num_nodes, device)  # reused per client

    for r in range(args.rounds):
        client_states, losses = [], []
        for c in range(args.num_clients):
            client_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr)
            loss = train_epochs(client_model, optimizer, shards[c], client_mp[c],
                                num_nodes, args.batch_size, args.local_epochs,
                                device, args.max_batches)
            client_states.append(copy.deepcopy(client_model.state_dict()))
            losses.append(loss)

        fedavg(global_model, client_states, client_weights)
        res = evaluate(global_model, full_mp, split_edge, evaluator, num_nodes, device)
        row = {"round": r, "loss": float(np.mean(losses)), **res}
        logger.log(row)
        print(fmt(row), flush=True)


def run_local(args, num_nodes, split_edge, evaluator, device, logger):
    """K clients train alone on their shard for the same budget, no aggregation.
    Each client evaluates on its own shard graph; report the mean over clients."""
    shards = shard_train_edges(split_edge["train"]["edge"], args.num_clients, args.seed)
    client_mp = [build_edge_index(s, num_nodes) for s in shards]

    models = [make_model(args, num_nodes, device) for _ in range(args.num_clients)]
    optims = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in models]

    for r in range(args.rounds):
        losses, dicts = [], []
        for c in range(args.num_clients):
            loss = train_epochs(models[c], optims[c], shards[c], client_mp[c],
                                num_nodes, args.batch_size, args.local_epochs,
                                device, args.max_batches)
            res = evaluate(models[c], client_mp[c], split_edge, evaluator, num_nodes, device)
            losses.append(loss)
            dicts.append(res)
        mean_res = {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}
        row = {"round": r, "loss": float(np.mean(losses)), **mean_res}
        logger.log(row)
        print(fmt(row), flush=True)


MODES = {
    "centralized": run_centralized,
    "federated": run_federated,
    "local": run_local,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), default="centralized")
    p.add_argument("--encoder", choices=["sage", "gcn"], default="sage")
    p.add_argument("--num_clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64 * 1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/")
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_batches", type=int, default=None,
                   help="cap batches per epoch (smoke testing only)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = csv_path(args)
    if is_complete(path, args.rounds):
        print(f"[skip] {path} already complete ({args.rounds} rounds).", flush=True)
        return

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"mode={args.mode} encoder={args.encoder} device={device} -> {path}", flush=True)

    from ogb.linkproppred import Evaluator
    evaluator = Evaluator(name="ogbl-ddi")
    num_nodes, split_edge = load_ddi(root=args.data_root)

    logger = CsvLogger(path)
    try:
        MODES[args.mode](args, num_nodes, split_edge, evaluator, device, logger)
    finally:
        logger.close()
    print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
