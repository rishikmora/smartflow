"""Build the corridor's junction graph as a PyTorch Geometric structure.

The adjacency itself comes from :func:`smartflow_env.neighbor_map`, which is the one
place in the project that derives "which junction feeds which" from the SUMO network.
Keeping a second copy of that logic here would let the graph the GAT encoder trains on
drift away from the graph the reward shaping reasons about.

This module exposes the same adjacency in the ``edge_index`` form PyTorch Geometric
expects, for inspection and for any whole-corridor (rather than per-junction star)
graph work.

Note that :class:`gnn_encoder.CorridorGATEncoder` does **not** consume this global
graph at training time. RLlib batches experience per policy, so each agent's forward
pass only sees its own rows; the neighbourhood is shipped inside the observation
instead and decoded into a per-sample star graph. This module is the corridor-level
view of the same relation.

Usage:
    python src/build_corridor_graph.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartflow_env import neighbor_map
from week4_config import CORRIDOR_NET, CORRIDOR_TLS_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def build_edge_index(net_file: str = CORRIDOR_NET,
                     tls_ids: list[str] | None = None) -> tuple[torch.Tensor, list[str]]:
    """Return the corridor adjacency as a PyTorch Geometric ``edge_index``.

    Args:
        net_file: path to the SUMO ``.net.xml``.
        tls_ids: junctions to include as nodes; defaults to the full corridor.

    Returns:
        ``(edge_index, node_order)`` where ``edge_index`` is a ``(2, E)`` long tensor of
        directed ``source -> destination`` edges and ``node_order`` maps row indices to
        junction ids.

    Raises:
        ValueError: if the network yields no edges between the given junctions.
    """
    nodes = list(tls_ids or CORRIDOR_TLS_IDS)
    index_of = {ts: i for i, ts in enumerate(nodes)}
    neighbours = neighbor_map(net_file, nodes)

    edges = sorted(
        (index_of[src], index_of[dst])
        for src, dsts in neighbours.items()
        for dst in dsts
        if dst in index_of
    )
    if not edges:
        raise ValueError(
            f"No junction-to-junction edges found in {net_file} for {nodes}. "
            "Check that the traffic-light ids match the network's junction ids."
        )
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), nodes


def build_data(feature_dim: int = 11, net_file: str = CORRIDOR_NET) -> Any:
    """Build a PyG ``Data`` object with zeroed node features.

    Args:
        feature_dim: per-junction feature width (matches the aligned observation).
        net_file: path to the SUMO ``.net.xml``.

    Returns:
        A ``torch_geometric.data.Data`` instance.

    Raises:
        RuntimeError: if torch_geometric is unavailable.
    """
    try:
        from torch_geometric.data import Data
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"torch_geometric is required to build graph Data objects: {exc}"
        ) from exc

    edge_index, nodes = build_edge_index(net_file)
    features = torch.zeros((len(nodes), feature_dim), dtype=torch.float32)
    return Data(x=features, edge_index=edge_index)


def main() -> None:
    """Print the corridor graph's shape and per-junction degrees."""
    edge_index, nodes = build_edge_index()
    log.info("nodes=%d edges=%d", len(nodes), edge_index.shape[1])
    out_degree = torch.bincount(edge_index[0], minlength=len(nodes))
    in_degree = torch.bincount(edge_index[1], minlength=len(nodes))
    for index, name in enumerate(nodes):
        log.info("  %-3s out=%d in=%d", name, int(out_degree[index]), int(in_degree[index]))

    data = build_data()
    log.info("PyG Data: num_nodes=%d num_edges=%d num_node_features=%d",
             data.num_nodes, data.num_edges, data.num_node_features)


if __name__ == "__main__":
    main()
