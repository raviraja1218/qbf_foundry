# src/models/gnn.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(nn.Module):
    """
    Small GCN-based regressor that takes node features + adjacency and
    predicts a scalar property per-graph (e.g., mean hydrophobicity proxy).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.pool = global_mean_pool
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        # x: [N_nodes_total, in_channels]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.pool(x, batch)  # [batch_size, hidden_channels]
        out = self.head(x).squeeze(-1)
        return out
