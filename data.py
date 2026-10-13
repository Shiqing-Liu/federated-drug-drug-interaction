"""Data loading and federated sharding for ogbl-ddi.

ogbl-ddi: 4,267 drug nodes, no node/edge features, undirected graph.
Train positives are stored once per edge in split_edge['train']['edge'] ([E, 2]).
Valid/test use official `edge` positives and fixed `edge_neg` negatives.
"""
import torch
from torch_geometric.utils import to_undirected

NUM_NODES = 4267


def _patch_torch_load():
    """Newer PyTorch defaults torch.load(weights_only=True), which breaks OGB
    loading its pickled processed data. Default it back to False defensively."""
    _orig = torch.load
    if getattr(_orig, "_ddi_patched", False):
        return

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._ddi_patched = True
    torch.load = _load


def load_ddi(root="dataset"):
    """Return (num_nodes, split_edge) for ogbl-ddi.

    split_edge = {'train': {'edge': [E,2]},
                  'valid': {'edge': [Ev,2], 'edge_neg': [Nv,2]},
                  'test':  {'edge': [Et,2], 'edge_neg': [Nt,2]}}
    """
    _patch_torch_load()
    from ogb.linkproppred import PygLinkPropPredDataset

    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    num_nodes = data.num_nodes  # 4267
    return num_nodes, split_edge


def build_edge_index(edges, num_nodes):
    """edges: [E, 2] tensor stored once per edge -> undirected edge_index [2, 2E]."""
    edge_index = edges.t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def shard_train_edges(train_edges, num_clients, seed):
    """Shuffle train edges with a fixed seed and split into K disjoint, equal shards.

    Returns a list of [Ek, 2] tensors (the leftover from non-divisibility is
    dropped so shards are exactly equal)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(train_edges.size(0), generator=g)
    shuffled = train_edges[perm]
    shard_size = shuffled.size(0) // num_clients
    shards = [shuffled[i * shard_size:(i + 1) * shard_size] for i in range(num_clients)]
    return shards
