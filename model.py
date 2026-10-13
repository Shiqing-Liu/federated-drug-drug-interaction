"""Link prediction model for ogbl-ddi.

Based on OGB's examples/linkproppred/ddi/gnn.py. One nn.Module bundles the
learnable node embedding table, the GNN encoder, and the MLP decoder, so a
single state_dict covers everything that FedAvg averages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv


class LinkPredModel(nn.Module):
    def __init__(self, num_nodes, hidden=256, encoder="sage", dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Learnable node input: nn.Embedding with Xavier init.
        self.emb = nn.Embedding(num_nodes, hidden)

        # Encoder: 2-layer GraphSAGE (default) or 2-layer GCN.
        if encoder == "sage":
            self.convs = nn.ModuleList([
                SAGEConv(hidden, hidden),
                SAGEConv(hidden, hidden),
            ])
        elif encoder == "gcn":
            # cached=False is required here: in federated/local modes the model
            # encodes each client's local graph during training but the union
            # graph at eval, so a cached normalization would reuse the wrong
            # graph. (It is already the PyG default, but we pin it to be safe.)
            self.convs = nn.ModuleList([
                GCNConv(hidden, hidden, cached=False),
                GCNConv(hidden, hidden, cached=False),
            ])
        else:
            raise ValueError(f"unknown encoder {encoder}")

        # Decoder: Hadamard product h_u * h_v -> 2-layer MLP -> sigmoid.
        self.dec_lin1 = nn.Linear(hidden, hidden)
        self.dec_lin2 = nn.Linear(hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
        self.dec_lin1.reset_parameters()
        self.dec_lin2.reset_parameters()

    def encode(self, edge_index):
        """Full-graph forward: returns node representations [N, hidden]."""
        h = self.emb.weight
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(self, h, edge):
        """edge: [2, B] node-index pairs -> sigmoid scores [B]."""
        x = h[edge[0]] * h[edge[1]]
        x = F.relu(self.dec_lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dec_lin2(x)
        return torch.sigmoid(x).squeeze(-1)
