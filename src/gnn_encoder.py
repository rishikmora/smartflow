"""GAT-based state encoder scaffold for corridor traffic-light observations."""

from __future__ import annotations

import torch
from torch import nn


class CorridorGATEncoder(nn.Module):
    """Two-layer Graph Attention Network for traffic-light state enrichment."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, embedding_dim: int = 16, heads: int = 2) -> None:
        """Initialize the GAT encoder."""
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except Exception as exc:
            raise RuntimeError("torch_geometric is required for CorridorGATEncoder.") from exc
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.activation = nn.ELU()
        self.gat2 = GATConv(hidden_dim * heads, embedding_dim, heads=1, concat=False)

    def forward(self, data) -> torch.Tensor:
        """Return enriched node embeddings with shape ``(num_junctions, embedding_dim)``."""
        x = self.gat1(data.x, data.edge_index)
        x = self.activation(x)
        x = self.gat2(x, data.edge_index)
        if torch.isnan(x).any():
            raise ValueError("GNN encoder produced NaN embeddings.")
        return x
