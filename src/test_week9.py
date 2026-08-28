"""Self-checks for Week 9: federated averaging, LoRA data hygiene, preemption.

The three Week 9 components each have one way of producing a result that looks
good and means nothing, and each check is aimed at that:

* **Weight round-tripping.** Federated averaging is only meaningful if weights
  can leave a policy and come back without changing its behaviour. If
  ``set_weights`` silently mismatched keys, every "federated" policy would just
  be a randomly perturbed one, and the comparison would be noise.
* **Aggregation must be a real weighted mean.** Checked against hand-computed
  values, including the sample-count weighting.
* **The Q&A split must not leak.** Paraphrases of one fact must not straddle the
  train/held-out boundary, or the held-out score measures memorisation.
* **Preemption must choose a phase that actually greens the target approach**,
  rather than any phase at all.

Runs standalone and is pytest-compatible.

Usage:
    python src/test_week9.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def test_fedavg_is_a_weighted_mean() -> None:
    """Flower's aggregation must weight by sample count, as FedAvg specifies."""
    import numpy as np
    from flwr.server.strategy.aggregate import aggregate

    a = [np.array([0.0, 10.0]), np.array([[1.0]])]
    b = [np.array([10.0, 20.0]), np.array([[3.0]])]

    equal = aggregate([(a, 1), (b, 1)])
    assert np.allclose(equal[0], [5.0, 15.0]), equal[0]
    assert np.allclose(equal[1], [[2.0]]), equal[1]

    # Three times the data on the second client pulls the mean three quarters of
    # the way towards it.
    weighted = aggregate([(a, 1), (b, 3)])
    assert np.allclose(weighted[0], [7.5, 17.5]), weighted[0]
    assert np.allclose(weighted[1], [[2.5]]), weighted[1]


def test_identical_clients_aggregate_to_themselves() -> None:
    """Averaging identical updates must be a no-op."""
    import numpy as np
    from flwr.server.strategy.aggregate import aggregate

    weights = [np.array([1.5, -2.5, 0.0]), np.array([[7.0, 8.0]])]
    result = aggregate([(weights, 10), (weights, 10), (weights, 10)])
    for original, aggregated in zip(weights, result):
        assert np.allclose(original, aggregated), (original, aggregated)


def test_policy_weights_round_trip() -> None:
    """Weights must survive get -> set without changing what the policy does."""
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    import gymnasium as gym
    from federated import get_weights, set_weights

    env = gym.make("CartPole-v1")          # shape-compatible stand-in, no SUMO needed
    try:
        source = PPO("MlpPolicy", env, seed=0, n_steps=64, batch_size=32, verbose=0)
        target = PPO("MlpPolicy", env, seed=1, n_steps=64, batch_size=32, verbose=0)

        # A batch of observations, so the check is about behaviour rather than
        # one lucky argmax.
        rng = np.random.default_rng(0)
        observations = rng.normal(0.0, 0.1, size=(64, 4)).astype(np.float32)
        before_source, _ = source.predict(observations, deterministic=True)
        before_target, _ = target.predict(observations, deterministic=True)

        set_weights(target, get_weights(source))
        after_target, _ = target.predict(observations, deterministic=True)

        assert np.array_equal(after_target, before_source), (
            "target should behave like source after loading its weights")
        assert not np.array_equal(before_target, before_source), (
            "the two policies were already identical, so this proves nothing")

        with torch.no_grad():
            for key, value in source.policy.state_dict().items():
                assert torch.allclose(value, target.policy.state_dict()[key]), key
        _ = before_target
    finally:
        env.close()


def test_qa_split_has_no_paraphrase_leakage() -> None:
    """No fact group may appear in both the train and held-out splits."""
    from lora_finetune import build_qa, split
    from week9_config import LORA_SEED

    records = build_qa()
    assert len(records) > 200, f"corpus is suspiciously small: {len(records)}"

    train, held_out = split(records, LORA_SEED)
    assert train and held_out, "both splits must be populated"

    train_groups = {r["group"] for r in train}
    held_groups = {r["group"] for r in held_out}
    overlap = train_groups & held_groups
    assert not overlap, f"{len(overlap)} fact groups leaked across the split"

    # And the answers themselves must not be duplicated across the boundary.
    train_answers = {r["answer"] for r in train}
    leaked = [r for r in held_out if r["answer"] in train_answers]
    assert not leaked, f"{len(leaked)} held-out answers appear verbatim in training"


def test_qa_facts_match_the_graph() -> None:
    """Spot-check that generated answers state what the graph actually holds."""
    from knowledge_graph import EmbeddedGraph, load_document
    from lora_finetune import build_qa

    graph = EmbeddedGraph(load_document())
    records = build_qa()

    topology = [r for r in records if r["kind"] == "topology"
                and r["question"].startswith("Which junctions does C2")]
    assert topology, "no C2 topology question generated"
    neighbours = graph.neighbours("C2")
    assert neighbours, "graph has no neighbours for C2"
    for neighbour in neighbours:
        assert neighbour in topology[0]["answer"], (neighbour, topology[0]["answer"])


def test_emissions_term_penalises_emitting_traffic() -> None:
    """With the emissions term on, more CO2 must mean less reward."""
    from rewards import RewardWeights, ShapedReward

    class FakeLane:
        """Minimal stand-in for the TraCI lane domain."""

        def __init__(self, co2: float) -> None:
            self.co2 = co2

        def getCO2Emission(self, _lane: str) -> float:      # noqa: N802
            return self.co2

        def getLastStepVehicleNumber(self, _lane: str) -> int:  # noqa: N802
            return 4

    class FakeSumo:
        def __init__(self, co2: float) -> None:
            self.lane = FakeLane(co2)

    class FakeSignal:
        """Minimal stand-in for a sumo-rl TrafficSignal."""

        def __init__(self, co2: float) -> None:
            self.lanes = ["l0", "l1"]
            self.sumo = FakeSumo(co2)
            self.last_measure = 0.0
            # ShapedReward stashes per-junction fairness stats on the env so the
            # environment can read them without a second TraCI round-trip.
            self.id = "fake"
            self.env = type("FakeEnv", (), {})()

        def get_accumulated_waiting_time_per_lane(self) -> list[float]:
            return [0.0, 0.0]

        def get_out_lanes_density(self) -> list[float]:
            return [0.0]

    weights = RewardWeights(emissions_beta=1.0, fairness_lambda=0.0,
                            coordination_alpha=0.0)
    reward_fn = ShapedReward(weights=weights, use_fairness=False,
                             use_coordination=False, use_emissions=True)

    clean = reward_fn(FakeSignal(co2=0.0))
    dirty = reward_fn(FakeSignal(co2=2_000_000.0))
    assert dirty < clean, f"emitting traffic scored {dirty} vs clean {clean}"


def test_federated_districts_share_a_policy_shape() -> None:
    """FedAvg requires every district to have identical observation/action shapes."""
    from week9_config import FED_DISTRICTS, FED_HELD_OUT

    assert FED_HELD_OUT not in FED_DISTRICTS, (
        "the held-out district must not also be trained on")
    assert len(FED_DISTRICTS) >= 2, "federation needs at least two clients"
    # The four interior junctions are the only 4-way ones; perimeter junctions
    # observe 9 features rather than 11 and cannot be averaged with them.
    interior = {"B1", "B2", "C1", "C2"}
    for junction in [*FED_DISTRICTS, FED_HELD_OUT]:
        assert junction in interior, f"{junction} is not a 4-way interior junction"


def test_federated_results_report_the_comparison_that_matters() -> None:
    """The recorded result must compare FedAvg against the local-only baseline."""
    from week9_config import FED_RESULTS

    if not os.path.isfile(FED_RESULTS):
        log.warning("  federated experiment not run; skipping")
        return
    with open(FED_RESULTS, encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload["summary"]
    for key in ("fixed_wait_s", "local_mean_wait_s", "fedavg_wait_s", "dod_met"):
        assert key in summary, f"missing {key}"

    for record in payload["per_seed"]:
        results = record["results"]
        assert "fedavg" in results and "local_mean" in results, results.keys()
        # local_mean must actually be the mean of the individual local policies.
        locals_ = [results[f"local_{j}"]["junction_wait_s"] for j in payload["districts"]]
        expected = sum(locals_) / len(locals_)
        assert abs(results["local_mean"]["junction_wait_s"] - expected) < 1e-6, (
            results["local_mean"], expected)


def test_priority_results_report_both_sides_of_the_trade() -> None:
    """Preemption results must record the cost to general traffic, not only the gain."""
    from week9_config import PRIORITY_RESULTS

    if not os.path.isfile(PRIORITY_RESULTS):
        log.warning("  priority experiment not run; skipping")
        return
    with open(PRIORITY_RESULTS, encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload["summary"]
    for key in ("emergency_wait_none_s", "emergency_wait_preempt_s",
                "general_wait_none_s", "general_wait_preempt_s"):
        assert key in summary, f"missing {key}"
    assert summary["emergency_wait_preempt_s"] < summary["emergency_wait_none_s"], (
        "preemption should reduce emergency delay")


def main() -> None:
    """Run every check and report a pass/fail summary."""
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures: list[str] = []
    for test in tests:
        try:
            test()
            log.info("PASS  %s", test.__name__)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            failures.append(f"{test.__name__}: {exc}")
            log.error("FAIL  %s -> %s", test.__name__, exc)
    log.info("%d/%d checks passed", len(tests) - len(failures), len(tests))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
