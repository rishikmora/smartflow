"""Build a PyTorch Geometric graph representation from the corridor SUMO net."""

from __future__ import annotations

import logging
import os
import sys

import torch
import sumolib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week4_config import CORRIDOR_NET, CORRIDOR_TLS_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def build_edge_index() -> torch.Tensor:
    """Return directed traffic-light adjacency as a PyTorch edge_index tensor."""
    try:
        net = sumolib.net.readNet(CORRIDOR_NET)
    except Exception as exc:
        raise RuntimeError(f"Could not read {CORRIDOR_NET}: {exc}") from exc
    tls_to_idx = {tls_id: idx for idx, tls_id in enumerate(CORRIDOR_TLS_IDS)}
    edges: set[tuple[int, int]] = set()
    for edge in net.getEdges():
        src = edge.getFromNode().getID()
        dst = edge.getToNode().getID()
        if src in tls_to_idx and dst in tls_to_idx and src != dst:
            edges.add((tls_to_idx[src], tls_to_idx[dst]))
    if not edges:
        raise ValueError("No traffic-light adjacency edges were found in corridor net.")
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def build_fake_data(feature_dim: int = 8):
    """Build a PyG Data object with fake node observations for encoder tests."""
    try:
        from torch_geometric.data import Data
    except Exception as exc:
        raise RuntimeError("torch_geometric is required for graph Data objects.") from exc
    x = torch.zeros((len(CORRIDOR_TLS_IDS), feature_dim), dtype=torch.float32)
    return Data(x=x, edge_index=build_edge_index())


def main() -> None:
    """CLI entry point."""
    data = build_fake_data()
    log.info("nodes=%d edges=%d feature_dim=%d", data.num_nodes, data.num_edges, data.num_node_features)


if __name__ == "__main__":
    main()
