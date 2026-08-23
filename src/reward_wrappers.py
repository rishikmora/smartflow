"""Reward shaping utilities for Phase B/C multi-agent traffic control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Weights used by the cooperative, fairness, and emissions reward terms."""

    coordination_alpha: float = 0.1
    fairness_lambda: float = 0.5
    max_wait_s: float = 120.0
    emissions_beta: float = 0.001


def shaped_reward(
    local_reward: float,
    downstream_queue_delta: float,
    lane_avg_wait_s: float,
    co2_mg_per_vehicle: float = 0.0,
    weights: RewardWeights | None = None,
) -> float:
    """Combine local, green-wave, fairness, and emissions terms into one reward."""
    cfg = weights or RewardWeights()
    coordination = -downstream_queue_delta
    fairness_penalty = max(0.0, lane_avg_wait_s - cfg.max_wait_s)
    co2_penalty = co2_mg_per_vehicle / 1_000_000.0
    return (
        local_reward
        + cfg.coordination_alpha * coordination
        - cfg.fairness_lambda * fairness_penalty
        - cfg.emissions_beta * co2_penalty
    )
