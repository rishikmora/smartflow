"""Self-checks for the pure logic behind SmartFlow's Week 3-6 results.

A formal test suite is scheduled for Phase D, so this is deliberately narrow: it
covers only the pieces where a silent bug would corrupt reported numbers without
crashing anything. Each of these has already bitten this project or is a plausible
place for it to.

* **Observation alignment.** Junctions have different lane counts, so a naive flat
  pad would put one agent's queue values where another's densities live. A shared
  policy would then read different quantities from the same input unit and still
  train "fine".
* **Reward shaping.** The shaped reward must reproduce sumo-rl's `diff-waiting-time`
  exactly when the extra terms are off, or Week 5 is not comparable to Week 4.
* **Dual ascent.** The multiplier must rise on violation, decay on satisfaction, and
  never go negative or unbounded.
* **Metric tails.** The p95 and worst-case statistics are what the fairness claim
  rests on.
* **Scenario construction.** Demand thinning must hit the documented ratio, and the
  east-west / north-south classification must match the grid's edge-naming.
* **GAT encoder.** Shapes and neighbour masking.

Runs standalone (no SUMO, no Ray) and is pytest-compatible.

Usage:
    python src/test_core_logic.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


# ── observation alignment ────────────────────────────────────────────────────
def test_aligner_places_fields_at_matching_indices() -> None:
    """A 3-lane and a 4-lane junction must expose density/queue at the same indices."""
    from marl_env import ObservationAligner

    aligner = ObservationAligner({"small": (2, 3), "big": (2, 4)})
    assert aligner.dim == 2 + 1 + 2 * 4, aligner.dim

    # layout: [phase(2) | min_green(1) | density(L) | queue(L)]
    small_raw = [1, 0,  1,  0.11, 0.12, 0.13,  0.21, 0.22, 0.23]
    big_raw = [0, 1,  1,  0.31, 0.32, 0.33, 0.34,  0.41, 0.42, 0.43, 0.44]

    small = aligner.align("small", small_raw)
    big = aligner.align("big", big_raw)

    density = slice(3, 7)
    queue = slice(7, 11)
    assert np.allclose(small[density], [0.11, 0.12, 0.13, 0.0]), small[density]
    assert np.allclose(small[queue], [0.21, 0.22, 0.23, 0.0]), small[queue]
    assert np.allclose(big[density], [0.31, 0.32, 0.33, 0.34]), big[density]
    assert np.allclose(big[queue], [0.41, 0.42, 0.43, 0.44]), big[queue]
    # min-green flag must land in the same slot for both
    assert small[2] == 1.0 and big[2] == 1.0


def test_aligner_rejects_wrong_length_observation() -> None:
    """A mis-sized observation must fail loudly rather than be silently padded."""
    from marl_env import ObservationAligner

    aligner = ObservationAligner({"a": (2, 3)})
    try:
        aligner.align("a", [0.0] * 5)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a wrong-length observation.")


def test_aligner_agent_id_one_hot_is_optional_and_unique() -> None:
    """Enabling agent ids must widen the vector and set exactly one identity bit."""
    from marl_env import ObservationAligner

    aligner = ObservationAligner({"a": (2, 3), "b": (2, 3)}, include_agent_id=True)
    assert aligner.dim == 2 + 1 + 2 * 3 + 2
    vec_a = aligner.align("a", [0.0] * 9)
    vec_b = aligner.align("b", [0.0] * 9)
    assert vec_a[-2:].tolist() == [1.0, 0.0]
    assert vec_b[-2:].tolist() == [0.0, 1.0]


# ── reward shaping ───────────────────────────────────────────────────────────
class _FakeEnv:
    """Stand-in for sumo-rl's environment, just to receive fairness stats."""


class _FakeLaneApi:
    """Stand-in for TraCI's lane domain."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def getLastStepVehicleNumber(self, lane: str) -> int:  # noqa: N802 - TraCI's name
        return self._counts[lane]


class _FakeSumo:
    """Stand-in for the TraCI connection a TrafficSignal holds."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.lane = _FakeLaneApi(counts)


class _FakeTrafficSignal:
    """Minimal stand-in for a sumo-rl ``TrafficSignal``.

    ``lane_counts`` defaults to one vehicle per lane, so a lane's summed accumulated
    wait equals its per-vehicle mean and the simple cases stay easy to read.
    """

    def __init__(self, lane_waits: list[float], out_density: list[float],
                 lane_counts: list[int] | None = None) -> None:
        self.id = "X1"
        self.env = _FakeEnv()
        self.last_measure = 0.0
        self.lanes = [f"lane{i}" for i in range(len(lane_waits))]
        counts = lane_counts if lane_counts is not None else [1] * len(lane_waits)
        self.sumo = _FakeSumo(dict(zip(self.lanes, counts)))
        self._lane_waits = lane_waits
        self._out_density = out_density

    def get_accumulated_waiting_time_per_lane(self) -> list[float]:
        return self._lane_waits

    def get_out_lanes_density(self) -> list[float]:
        return self._out_density


def test_shaped_reward_matches_diff_waiting_time_when_terms_disabled() -> None:
    """With shaping off, the reward must equal sumo-rl's diff-waiting-time exactly."""
    from rewards import RewardWeights, ShapedReward

    weights = RewardWeights(coordination_alpha=0.0, fairness_lambda=0.0, max_wait_s=120.0)
    reward_fn = ShapedReward(weights, use_fairness=False, use_coordination=False)

    ts = _FakeTrafficSignal(lane_waits=[100.0, 50.0], out_density=[0.9])
    first = reward_fn(ts)
    # sumo-rl: reward = last_measure - sum(lane_waits)/100 ; last_measure starts at 0
    assert abs(first - (0.0 - 150.0 / 100.0)) < 1e-9, first
    assert abs(ts.last_measure - 1.5) < 1e-9

    # a second call with a smaller queue must yield a positive (improving) reward
    ts._lane_waits = [20.0, 10.0]
    second = reward_fn(ts)
    assert abs(second - (1.5 - 0.3)) < 1e-9, second


def test_coordination_term_penalises_full_downstream_links() -> None:
    """Discharging into a congested downstream link must lower the reward."""
    from rewards import RewardWeights, ShapedReward

    weights = RewardWeights(coordination_alpha=0.2, fairness_lambda=0.0)
    reward_fn = ShapedReward(weights, use_fairness=False, use_coordination=True)

    empty = reward_fn(_FakeTrafficSignal([10.0], out_density=[0.0]))
    full = reward_fn(_FakeTrafficSignal([10.0], out_density=[1.0]))
    assert full < empty, (full, empty)
    assert abs((empty - full) - 0.2) < 1e-9


def test_fairness_penalty_applies_only_above_the_cap() -> None:
    """The Lagrangian term must be inactive while the constraint is satisfied."""
    import rewards
    from rewards import RewardWeights, ShapedReward

    rewards.set_lambda(1.0)
    weights = RewardWeights(coordination_alpha=0.0, max_wait_s=120.0)
    reward_fn = ShapedReward(weights, use_fairness=True, use_coordination=False)

    under = reward_fn(_FakeTrafficSignal([100.0], out_density=[]))
    over = reward_fn(_FakeTrafficSignal([200.0], out_density=[]))
    # under the cap: pure diff-waiting-time
    assert abs(under - (-1.0)) < 1e-9, under
    # over the cap: extra -lambda * (200-120)/100 = -0.8 on top of the base term
    assert over < -2.0, over


def test_fairness_constraint_uses_per_vehicle_wait_not_the_queue_sum() -> None:
    """The cap must apply to per-vehicle wait, not the lane's summed accumulated wait.

    sumo-rl's ``get_accumulated_waiting_time_per_lane`` sums over every vehicle on the
    lane, so it scales with queue length. Constraining that sum against a seconds-valued
    cap makes the constraint violated by a huge factor at all times and the penalty
    swamps the delay reward.
    """
    from rewards import RewardWeights, ShapedReward

    weights = RewardWeights(coordination_alpha=0.0, max_wait_s=120.0)
    reward_fn = ShapedReward(weights, use_fairness=True, use_coordination=False)

    # 20 vehicles, 100 s summed wait -> 5 s each: comfortably inside the cap.
    ts = _FakeTrafficSignal([100.0], out_density=[], lane_counts=[20])
    reward_fn(ts)
    assert ts.env.fairness_stats["X1"]["max_lane_wait_s"] == 5.0
    assert ts.env.fairness_stats["X1"]["violation_s"] == 0.0

    # 2 vehicles, 400 s summed wait -> 200 s each: a real violation.
    ts2 = _FakeTrafficSignal([400.0], out_density=[], lane_counts=[2])
    reward_fn(ts2)
    assert ts2.env.fairness_stats["X1"]["max_lane_wait_s"] == 200.0
    assert ts2.env.fairness_stats["X1"]["violation_s"] == 80.0


def test_fairness_penalty_is_scale_invariant_in_the_cap() -> None:
    """At twice the cap the penalty must be lambda * fairness_scale, for any cap value.

    Retuning the cap must not silently change how hard violations are punished; only
    lambda (learned) and fairness_scale (fixed units) control that.
    """
    import rewards
    from rewards import RewardWeights, ShapedReward

    rewards.set_lambda(1.0)
    for cap in (30.0, 120.0, 600.0):
        weights = RewardWeights(coordination_alpha=0.0, max_wait_s=cap)
        reward_fn = ShapedReward(weights, use_fairness=True, use_coordination=False)
        no_penalty = ShapedReward(weights, use_fairness=False, use_coordination=False)
        # one vehicle waiting exactly twice the cap -> violation ratio of 1.0
        penalised = reward_fn(_FakeTrafficSignal([2 * cap], out_density=[], lane_counts=[1]))
        plain = no_penalty(_FakeTrafficSignal([2 * cap], out_density=[], lane_counts=[1]))
        assert abs((plain - penalised) - weights.fairness_scale) < 1e-9, (cap, plain - penalised)


def test_fairness_stats_are_published_for_the_env() -> None:
    """The reward must stash per-junction fairness stats for the env to aggregate."""
    import rewards
    from rewards import RewardWeights, ShapedReward

    rewards.set_lambda(0.5)
    ts = _FakeTrafficSignal([300.0, 10.0], out_density=[])
    ShapedReward(RewardWeights(max_wait_s=120.0))(ts)
    stats = ts.env.fairness_stats["X1"]
    assert stats["max_lane_wait_s"] == 300.0
    assert stats["violation_s"] == 180.0


def test_dual_ascent_rises_decays_and_stays_bounded() -> None:
    """The multiplier must respond to the sign of the violation and stay in range."""
    from rewards import dual_ascent

    assert dual_ascent(1.0, violation_ratio=2.0, step_size=0.5) == 2.0      # violated -> rises
    assert dual_ascent(1.0, violation_ratio=-1.0, step_size=0.5) == 0.5     # satisfied -> decays
    assert dual_ascent(0.1, violation_ratio=-10.0, step_size=1.0) == 0.0    # never negative
    assert dual_ascent(19.0, violation_ratio=100.0, step_size=1.0, lam_max=20.0) == 20.0


def test_make_reward_fn_rejects_unknown_names() -> None:
    """An unknown reward name must fail at config time, not silently fall back."""
    from rewards import make_reward_fn

    assert make_reward_fn("diff-waiting-time") == "diff-waiting-time"
    try:
        make_reward_fn("not-a-reward")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an unknown reward name.")


# ── metric tails ─────────────────────────────────────────────────────────────
def test_wait_percentile_and_worst_case() -> None:
    """Percentile and worst-case helpers must read the completed-trip distribution."""
    from metrics import MetricsCollector

    collector = MetricsCollector.__new__(MetricsCollector)  # bypass the SUMO connection
    collector._completed_waits = list(range(1, 101))        # 1..100
    collector._veh_stopped_s = {"still_driving": 250}
    assert collector.wait_percentile(95) == 95.0
    assert collector.wait_percentile(0) == 1.0
    # the worst case must include vehicles that never completed their trip
    assert collector.max_lane_wait_s() == 250.0

    empty = MetricsCollector.__new__(MetricsCollector)
    empty._completed_waits = []
    empty._veh_stopped_s = {}
    assert empty.wait_percentile(95) == 0.0
    assert empty.max_lane_wait_s() == 0.0


# ── scenario construction ────────────────────────────────────────────────────
def test_edge_axis_classification_matches_grid_naming() -> None:
    """Edge ids encode direction: A1B1 runs east-west, A1A2 north-south."""
    from scenarios import _edge_axis

    assert _edge_axis("A1B1") == "ew"
    assert _edge_axis("B1A1") == "ew"
    assert _edge_axis("A1A2") == "ns"
    assert _edge_axis("D2D1") == "ns"
    assert _edge_axis("weird") is None


def test_asymmetric_thinning_hits_the_documented_ratio() -> None:
    """The north-south keep fraction must match ASYMMETRIC_NS_KEEP, not round to 1/N."""
    import xml.etree.ElementTree as ET

    import scenarios

    def make(axis: str) -> ET.Element:
        vehicle = ET.Element("vehicle", {"id": "0", "depart": "0"})
        edges = "A1B1 B1C1" if axis == "ew" else "A1A2 A2A3"
        ET.SubElement(vehicle, "route", {"edges": edges})
        return vehicle

    vehicles = [make("ns") for _ in range(1000)]
    kept = scenarios._build_asymmetric(vehicles)
    ratio = len(kept) / len(vehicles)
    assert abs(ratio - scenarios.ASYMMETRIC_NS_KEEP) < 0.01, ratio

    ew = [make("ew") for _ in range(50)]
    assert len(scenarios._build_asymmetric(ew)) == 50, "east-west traffic must be untouched"


# ── GAT encoder ──────────────────────────────────────────────────────────────
def test_gat_encoder_shapes_and_masking() -> None:
    """The encoder must return one embedding per row and honour the presence mask."""
    import torch

    from gnn_encoder import CorridorGATEncoder

    encoder = CorridorGATEncoder(node_dim=11, max_neighbors=4, hidden_dim=16, embed_dim=8, heads=2)
    assert encoder.expected_obs_dim == 11 + 4 * 12

    obs = torch.rand(6, encoder.expected_obs_dim)
    out = encoder(obs)
    assert out.shape == (6, 8), out.shape
    assert not torch.isnan(out).any()

    # a row with no neighbours must still produce a finite embedding (self-loop only)
    isolated = torch.zeros(1, encoder.expected_obs_dim)
    isolated[0, :11] = torch.rand(11)
    assert torch.isfinite(encoder(isolated)).all()

    # edge_index must drop masked-out neighbour slots
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    edge_index = encoder.build_edge_index(mask)
    # 2 self-loops + 1 neighbour for row 0 + 4 for row 1 = 7 edges
    assert edge_index.shape == (2, 7), edge_index.shape


def test_gat_encoder_rejects_wrong_width() -> None:
    """A mismatched observation width must raise rather than silently reshape."""
    import torch

    from gnn_encoder import CorridorGATEncoder

    encoder = CorridorGATEncoder(node_dim=11, max_neighbors=4, hidden_dim=8, embed_dim=4, heads=1)
    try:
        encoder(torch.rand(2, 11))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a wrong-width observation batch.")


# ── checkpoint selection ─────────────────────────────────────────────────────
def test_checkpoint_tiebreak_prefers_the_more_trained_policy() -> None:
    """Equal validation scores must resolve to the most-trained checkpoint.

    Once the policy converges, many checkpoints score identically on the validation
    episode. Breaking those ties by list order would report a 25k-step policy from a
    200k-step run for no reason other than scoring order.
    """
    from select_best_checkpoint import pick_best

    tied = [
        {"label": "25000_steps", "score": 7.24},
        {"label": "101200_steps", "score": 7.24},
        {"label": "final", "score": 7.24},
        {"label": "50000_steps", "score": 17.02},
    ]
    assert pick_best(tied, "avg_wait_time_s")["label"] == "final"

    # a genuinely better score still wins, however few steps it had
    clear = [{"label": "25000_steps", "score": 5.0}, {"label": "final", "score": 7.24}]
    assert pick_best(clear, "avg_wait_time_s")["label"] == "25000_steps"

    # higher-is-better metrics invert the comparison
    throughput = [{"label": "25000_steps", "score": 700.0}, {"label": "final", "score": 740.0}]
    assert pick_best(throughput, "throughput_veh")["label"] == "final"


# ── analysis helpers ─────────────────────────────────────────────────────────
def test_improvement_sign_follows_metric_direction() -> None:
    """Lower-is-better and higher-is-better metrics must both report + when better."""
    from analysis import improvement

    assert improvement(100.0, 50.0, "avg_wait_time_s") == 50.0     # halved wait = better
    assert improvement(100.0, 150.0, "avg_wait_time_s") == -50.0   # worse
    assert improvement(100.0, 150.0, "throughput_veh") == 50.0     # more throughput = better
    assert improvement(0.0, 5.0, "throughput_veh") == 0.0          # zero baseline guard


def main() -> None:
    """Run every check in this module and report a pass/fail summary."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures: list[str] = []
    for test in tests:
        try:
            test()
            log.info("PASS  %s", test.__name__)
        except Exception as exc:  # noqa: BLE001 - this is the reporting boundary
            failures.append(f"{test.__name__}: {exc}")
            log.error("FAIL  %s -> %s", test.__name__, exc)

    log.info("%d/%d checks passed", len(tests) - len(failures), len(tests))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
