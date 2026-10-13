"""Training, evaluation, and FedAvg aggregation."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

EPS = 1e-15


def train_epochs(model, optimizer, train_edges, mp_edge_index, num_nodes,
                 batch_size, epochs, device, max_batches=None):
    """Train `model` for `epochs` local epochs on `train_edges`.

    Each mini-batch recomputes the full-graph encoder forward on `mp_edge_index`
    (the message-passing graph), then scores only that batch's positive edges
    plus an equal number of freshly sampled negatives. BCE loss, grad clip 1.0.
    Negative sampling excludes the edges in `mp_edge_index`.
    """
    model.train()
    pos_edges = train_edges.to(device)          # [E, 2]
    mp_edge_index = mp_edge_index.to(device)     # [2, 2E]
    loader = DataLoader(range(pos_edges.size(0)), batch_size=batch_size, shuffle=True)

    last_loss = 0.0
    for _ in range(epochs):
        total_loss = total_examples = 0
        for b, perm in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            optimizer.zero_grad()
            h = model.encode(mp_edge_index)

            edge = pos_edges[perm].t()           # [2, B]
            pos_out = model.decode(h, edge)
            pos_loss = -torch.log(pos_out + EPS).mean()

            neg_edge = negative_sampling(
                mp_edge_index, num_nodes=num_nodes,
                num_neg_samples=edge.size(1), method="sparse",
            ).to(device)
            neg_out = model.decode(h, neg_edge)
            neg_loss = -torch.log(1 - neg_out + EPS).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = edge.size(1)
            total_loss += loss.item() * n
            total_examples += n
        last_loss = total_loss / total_examples
    return last_loss


@torch.no_grad()
def _score_edges(model, h, edges, device, chunk=70000):
    outs = []
    edges = edges.to(device)
    for i in range(0, edges.size(0), chunk):
        edge = edges[i:i + chunk].t()
        outs.append(model.decode(h, edge))
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, mp_edge_index, split_edge, evaluator, num_nodes, device,
             ks=(10, 20, 30)):
    """Evaluate with the official OGB Evaluator at K=10/20/30.

    `mp_edge_index` is the message-passing graph used to encode node reps.
    Returns dict like {'valid_hits@20': .., 'test_hits@20': .., ...}.
    """
    model.eval()
    h = model.encode(mp_edge_index.to(device))

    results = {}
    for split in ("valid", "test"):
        pos = _score_edges(model, h, split_edge[split]["edge"], device)
        neg = _score_edges(model, h, split_edge[split]["edge_neg"], device)
        for k in ks:
            evaluator.K = k
            hits = evaluator.eval({"y_pred_pos": pos, "y_pred_neg": neg})[f"hits@{k}"]
            results[f"{split}_hits@{k}"] = hits
    return results


def fedavg(global_model, client_state_dicts, client_weights):
    """Average full state_dicts weighted by client edge counts, load into global."""
    total = float(sum(client_weights))
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        stacked = torch.stack([
            client_state_dicts[i][key].float() * (client_weights[i] / total)
            for i in range(len(client_state_dicts))
        ], dim=0)
        global_dict[key] = stacked.sum(dim=0)
    global_model.load_state_dict(global_dict)
