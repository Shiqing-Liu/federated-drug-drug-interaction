"""Plot valid and test Hits@20 vs communication round for all 4 methods."""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def runs(out, k, seed):
    return [
        ("Centralized SAGE", f"centralized_sage_seed{seed}.csv"),
        ("Federated SAGE", f"federated_sage_K{k}_seed{seed}.csv"),
        ("Federated GCN", f"federated_gcn_K{k}_seed{seed}.csv"),
        ("Local-only SAGE", f"local_sage_K{k}_seed{seed}.csv"),
    ]


def read_csv(path):
    rounds, valid, test = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            valid.append(float(row["valid_hits@20"]))
            test.append(float(row["test_hits@20"]))
    return rounds, valid, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/")
    p.add_argument("--num_clients", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
    for label, fname in runs(args.out, args.num_clients, args.seed):
        path = os.path.join(args.out, fname)
        if not os.path.exists(path):
            print(f"[warn] missing {path}, skipping")
            continue
        r, v, t = read_csv(path)
        ax_v.plot(r, v, marker="o", ms=3, label=label)
        ax_t.plot(r, t, marker="o", ms=3, label=label)

    for ax, title in [(ax_v, "Valid Hits@20"), (ax_t, "Test Hits@20")]:
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Hits@20")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out_png = os.path.join(args.out, "hits20.png")
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
