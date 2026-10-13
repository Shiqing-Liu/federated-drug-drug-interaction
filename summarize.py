"""Write results/summary.md: per-method best-valid test Hits@20, convergence
round, and the CV numbers. Model selection is always on VALID, never on test.
"""
import argparse
import csv
import os


def read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def best_valid(rows):
    """Return (best_round_index, test@20 at that round, best valid@20)."""
    valids = [r["valid_hits@20"] for r in rows]
    best_i = max(range(len(valids)), key=lambda i: valids[i])
    return best_i, rows[best_i]["test_hits@20"], valids[best_i]


def convergence_round(rows):
    """First round where valid Hits@20 reaches >= 95% of its best value."""
    valids = [r["valid_hits@20"] for r in rows]
    best = max(valids)
    thresh = 0.95 * best
    for i, v in enumerate(valids):
        if v >= thresh:
            return int(rows[i]["round"])
    return int(rows[-1]["round"])


def load(out, fname):
    path = os.path.join(out, fname)
    if not os.path.exists(path):
        return None
    rows = read_csv(path)
    return rows if rows else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/")
    p.add_argument("--num_clients", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    k, seed = args.num_clients, args.seed

    methods = [
        ("Centralized SAGE", f"centralized_sage_seed{seed}.csv"),
        ("Federated SAGE", f"federated_sage_K{k}_seed{seed}.csv"),
        ("Federated GCN", f"federated_gcn_K{k}_seed{seed}.csv"),
        ("Local-only SAGE", f"local_sage_K{k}_seed{seed}.csv"),
    ]

    stats = {}
    lines = ["# Results summary", "",
             "Test Hits@20 is reported at the round with the best **valid** "
             "Hits@20 (selection never uses test).", "",
             "| Method | Test Hits@20 (best valid) | Best valid Hits@20 | "
             "Convergence round |", "|---|---|---|---|"]

    for label, fname in methods:
        rows = load(args.out, fname)
        if rows is None:
            lines.append(f"| {label} | _missing_ | _missing_ | _missing_ |")
            continue
        _, test20, valid20 = best_valid(rows)
        conv = convergence_round(rows)
        stats[label] = {"test20": test20, "valid20": valid20, "conv": conv}
        lines.append(f"| {label} | {test20:.4f} | {valid20:.4f} | {conv} |")

    # CV numbers
    lines += ["", "## CV numbers", ""]
    cen = stats.get("Centralized SAGE")
    fed = stats.get("Federated SAGE")
    gcn = stats.get("Federated GCN")
    loc = stats.get("Local-only SAGE")

    def pct(x):
        return f"{x * 100:.2f}%"

    lines.append(f"- K (num_clients): {k}")
    if fed:
        lines.append(f"- Federated SAGE test Hits@20: {pct(fed['test20'])}")
        lines.append(f"- Convergence round (federated SAGE): {fed['conv']}")
    if fed and gcn and gcn["test20"] > 0:
        imp = (fed["test20"] - gcn["test20"]) / gcn["test20"]
        lines.append(f"- Improvement over federated GCN: {pct(imp)}  "
                     f"(SAGE {pct(fed['test20'])} vs GCN {pct(gcn['test20'])})")
    if fed and cen and cen["test20"] > 0:
        ret = fed["test20"] / cen["test20"]
        gap = (cen["test20"] - fed["test20"]) * 100
        lines.append(f"- Retention (federated / centralized): {pct(ret)}")
        lines.append(f"- Centralized test Hits@20: {pct(cen['test20'])}")
        lines.append(f"- Gap (centralized - federated): {gap:.2f} Hits@20 points")
    if loc:
        lines.append(f"- Local-only test Hits@20: {pct(loc['test20'])}")

    out_md = os.path.join(args.out, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
