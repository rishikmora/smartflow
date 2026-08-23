"""Graph-attention state encoder for the SmartFlow corridor (Week 4).

Motivation
----------
The default MLP policy sees only its own junction's queues and phase. A junction's
best action, though, depends on what its neighbours are doing — a green wave only
exists if B1 knows C1 is about to release a platoon towards it. This module replaces
the MLP torso with a two-layer Graph Attention Network (PyTorch Geometric ``GATConv``)
that attends over the junction and its directly-downstream neighbours.

Why a star graph per sample
---------------------------
RLlib batches experience *per policy*, so a policy's forward pass only ever receives
its own agent's rows — never the full 12-junction graph at a synchronised timestep.
The environment therefore ships each agent's neighbourhood inside its observation
(``marl_env.CorridorParallelEnv`` with ``neighbor_context=True``), laid out as::

    [ own node features (D) | (neighbour features (D), present flag (1)) x K ]

Each row is decoded back into a ``K+1``-node star graph — neighbours pointing at the
centre — and a batch of ``B`` rows becomes one disjoint union of ``B`` stars, which is
exactly how PyTorch Geometric batches graphs. Absent neighbours are dropped from
``edge_index`` via their presence flag rather than being fed in as zero nodes, so the
attention softmax is never diluted by padding.

The encoder is trained end-to-end with PPO; it is not a frozen feature extractor.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def _require_pyg() -> Any:
    """Import ``torch_geometric``'s ``GATConv``, with an actionable error message.

    Returns:
        The ``GATConv`` class.

    Raises:
        RuntimeError: if torch_geometric is not installed.
    """
    try:
        from torch_geometric.nn import GATConv
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            f"torch_geometric is required for the GAT encoder: {exc}. "
            "Install it into the venv with 'pip install torch-geometric'."
        ) from exc
    return GATConv


class CorridorGATEncoder(nn.Module):
    """Two-layer graph-attention encoder over a batch of junction star graphs.

    Args:
        node_dim: feature width of one junction's aligned observation.
        max_neighbors: neighbour slots per junction.
        hidden_dim: width of the first GAT layer (per attention head).
        embed_dim: width of the returned centre-node embedding.
        heads: number of attention heads in the first layer.
    """

    def __init__(
        self,
        node_dim: int,
        max_neighbors: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        heads: int = 4,
    ) -> None:
        super().__init__()
        gat_conv = _require_pyg()
        self.node_dim = int(node_dim)
        self.max_neighbors = int(max_neighbors)
        self.embed_dim = int(embed_dim)
        self.gat1 = gat_conv(self.node_dim, hidden_dim, heads=heads, concat=True)
        self.activation = nn.ELU()
        self.gat2 = gat_conv(hidden_dim * heads, self.embed_dim, heads=1, concat=False)
        # edge_index depends only on the batch size and the presence mask, so the
        # per-graph offsets are rebuilt per call but the node template is fixed.
        self.nodes_per_graph = self.max_neighbors + 1

    @property
    def expected_obs_dim(self) -> int:
        """Observation width this encoder expects from the environment."""
        return self.node_dim + self.max_neighbors * (self.node_dim + 1)

    def decode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a flat observation batch into node features and a presence mask.

        Args:
            obs: ``(B, expected_obs_dim)`` float tensor.

        Returns:
            ``(nodes, mask)`` where ``nodes`` is ``(B, K+1, node_dim)`` (index 0 is the
            centre junction) and ``mask`` is ``(B, K)`` marking present neighbours.

        Raises:
            ValueError: if ``obs`` has the wrong width.
        """
        if obs.dim() != 2 or obs.shape[1] != self.expected_obs_dim:
            raise ValueError(
                f"CorridorGATEncoder expected observations of shape (B, {self.expected_obs_dim}), "
                f"got {tuple(obs.shape)}. Was the env built with neighbor_context=True?"
            )
        batch = obs.shape[0]
        centre = obs[:, : self.node_dim]
        rest = obs[:, self.node_dim :].reshape(batch, self.max_neighbors, self.node_dim + 1)
        neighbours = rest[:, :, : self.node_dim]
        mask = rest[:, :, self.node_dim]
        nodes = torch.cat([centre.unsqueeze(1), neighbours], dim=1)
        return nodes, mask

    def build_edge_index(self, mask: torch.Tensor) -> torch.Tensor:
        """Build a batched ``edge_index`` of neighbour -> centre edges plus self-loops.

        PyTorch Geometric represents a batch of graphs as one big disjoint graph, so
        graph *b*'s node *j* lives at global index ``b * (K+1) + j``.

        Args:
            mask: ``(B, K)`` tensor, 1 where a neighbour slot is occupied.

        Returns:
            ``(2, E)`` long tensor of directed edges.
        """
        batch, k = mask.shape
        device = mask.device
        offsets = torch.arange(batch, device=device, dtype=torch.long) * self.nodes_per_graph

        # self-loops: centre -> centre, so an isolated junction still has a message
        self_src = offsets
        self_dst = offsets

        slot = torch.arange(1, k + 1, device=device, dtype=torch.long).unsqueeze(0)  # (1, K)
        src = (offsets.unsqueeze(1) + slot).reshape(-1)                              # (B*K,)
        dst = offsets.unsqueeze(1).expand(batch, k).reshape(-1)                      # (B*K,)
        keep = mask.reshape(-1) > 0.5
        src, dst = src[keep], dst[keep]

        return torch.stack([torch.cat([self_src, src]), torch.cat([self_dst, dst])], dim=0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations into centre-junction embeddings.

        Args:
            obs: ``(B, expected_obs_dim)`` float tensor.

        Returns:
            ``(B, embed_dim)`` embeddings for the centre junction of each star.
        """
        nodes, mask = self.decode(obs)
        batch = nodes.shape[0]
        x = nodes.reshape(batch * self.nodes_per_graph, self.node_dim)
        edge_index = self.build_edge_index(mask)
        x = self.activation(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        x = x.reshape(batch, self.nodes_per_graph, self.embed_dim)
        return x[:, 0, :]


def make_gat_module_class() -> type:
    """Build the RLlib module class, binding to the installed RLlib at call time.

    Returns:
        A ``TorchRLModule`` subclass implementing ``ValueFunctionAPI``.
    """
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.apis import ValueFunctionAPI
    from ray.rllib.core.rl_module.torch import TorchRLModule

    class CorridorGATModule(TorchRLModule, ValueFunctionAPI):
        """PPO policy whose torso is a graph-attention encoder over neighbours.

        ``model_config`` keys: ``node_dim``, ``max_neighbors``, ``hidden_dim``,
        ``embed_dim``, ``heads``.
        """

        def setup(self) -> None:
            """Build the encoder and the policy/value heads."""
            cfg = self.model_config or {}
            node_dim = int(cfg.get("node_dim", 11))
            max_neighbors = int(cfg.get("max_neighbors", 4))
            self.encoder = CorridorGATEncoder(
                node_dim=node_dim,
                max_neighbors=max_neighbors,
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                embed_dim=int(cfg.get("embed_dim", 32)),
                heads=int(cfg.get("heads", 4)),
            )
            embed_dim = self.encoder.embed_dim
            num_actions = int(self.action_space.n)
            self.policy_head = nn.Sequential(
                nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, num_actions)
            )
            self.value_head = nn.Sequential(
                nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        def _embed(self, batch: dict[str, Any]) -> torch.Tensor:
            """Run the encoder over a batch's observations."""
            obs = batch[Columns.OBS]
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs)
            return self.encoder(obs.float())

        def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            """Default forward pass: action logits only."""
            return {Columns.ACTION_DIST_INPUTS: self.policy_head(self._embed(batch))}

        def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            """Training forward pass: logits plus a cached embedding for the critic."""
            embeddings = self._embed(batch)
            return {
                Columns.ACTION_DIST_INPUTS: self.policy_head(embeddings),
                Columns.EMBEDDINGS: embeddings,
            }

        def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> torch.Tensor:
            """Return state values, reusing the training embedding when available."""
            if embeddings is None:
                embeddings = self._embed(batch)
            return self.value_head(embeddings).squeeze(-1)

    return CorridorGATModule


def build_gnn_module_spec(policy_ids: Any, node_dim: int = 11, max_neighbors: int = 4,
                          hidden_dim: int = 64, embed_dim: int = 32, heads: int = 4) -> Any:
    """Build a ``MultiRLModuleSpec`` that gives every policy a GAT torso.

    Args:
        policy_ids: iterable of policy ids to configure.
        node_dim: per-junction feature width (the aligned observation width).
        max_neighbors: neighbour slots per junction.
        hidden_dim: first GAT layer width per head.
        embed_dim: encoder output width.
        heads: attention heads in the first layer.

    Returns:
        A ``MultiRLModuleSpec`` mapping each policy id to a GAT module spec.
    """
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    module_class = make_gat_module_class()
    model_config = {
        "node_dim": node_dim,
        "max_neighbors": max_neighbors,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "heads": heads,
    }
    return MultiRLModuleSpec(
        rl_module_specs={
            pid: RLModuleSpec(module_class=module_class, model_config=model_config)
            for pid in policy_ids
        }
    )


def main() -> None:
    """Self-check: encode a random batch and report the output shape."""
    encoder = CorridorGATEncoder(node_dim=11, max_neighbors=4)
    batch = 8
    obs = torch.rand(batch, encoder.expected_obs_dim)
    obs[:, 11 + 11] = 0.0  # mark the first neighbour slot absent for every row
    out = encoder(obs)
    log.info("input=%s -> embedding=%s", tuple(obs.shape), tuple(out.shape))
    if torch.isnan(out).any():
        raise ValueError("GAT encoder produced NaN embeddings.")
    log.info("GAT encoder self-check passed.")


if __name__ == "__main__":
    main()
