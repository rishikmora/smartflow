"""Week 5 reward shaping: green-wave coordination + Lagrangian fairness.

Weeks 2-4 use sumo-rl's stock ``diff-waiting-time`` reward unchanged, so those
results stay comparable with the literature. Week 5 adds two shaping terms on top of
that same base signal::

    r_i = base_i + alpha * coordination_i - lambda * fairness_violation_i - beta * emissions_i

``base_i`` — sumo-rl's ``diff-waiting-time``
    The reduction in total accumulated waiting time at junction *i* since its last
    decision, divided by 100. Reproduced exactly (including the ``last_measure``
    bookkeeping) so Week 5's reward is a strict superset of Week 4's.

``coordination_i`` — the green wave
    ``-mean(density of junction i's outgoing lanes)``. A junction is penalised for
    discharging a platoon into a link that is already full. Rewarding a junction only
    for its own queue makes it push congestion onto its neighbour; this term is what
    turns twelve selfish agents into a corridor. It is a backpressure-flavoured signal
    and needs no extra TraCI calls beyond what sumo-rl already exposes.

``fairness_violation_i`` — the constrained-MDP term
    The constraint is "no approach lane makes its vehicles wait more than
    ``MAX_WAIT_S`` seconds on average". ``lambda`` is a Lagrange multiplier updated by
    **dual ascent** in the training loop (see ``train_marl_corridor.py``), not a
    hand-tuned penalty weight: it rises while the constraint is violated and decays
    once it is satisfied. A pure average-delay objective is happy to starve a side
    street forever, which is exactly what this prevents.

    The constrained quantity is **per-vehicle** wait, not sumo-rl's raw
    ``get_accumulated_waiting_time_per_lane()``. That function returns the *sum* of
    accumulated waiting time over every vehicle on the lane, so it grows with queue
    length and reaches several thousand seconds under congestion on this corridor.
    Constraining that sum against a 120 s cap makes the constraint violated by a factor
    of ~40 at all times, and the penalty then dwarfs the O(1) delay reward — measured
    episode returns near -1.6e5, with the policy effectively optimising the penalty
    alone. Dividing by the lane's vehicle count gives "average wait per vehicle on the
    worst approach", which is both the quantity the cap is meant to express and a scale
    the reward can actually balance. The violation is then normalised by the cap, so
    the penalty is O(1) at twice the cap regardless of its absolute value.

``emissions_i``
    Reserved for Week 9's emission-smoothing term; ``beta`` defaults to 0 so it is
    inert in Week 5.

The multiplier lives in a module-level :class:`FairnessState` because each RLlib env
runner is its own process: the driver broadcasts a new ``lambda`` to every runner
through ``foreach_env_runner`` between training iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the shaped reward terms.

    Attributes:
        coordination_alpha: weight on the green-wave (downstream density) term.
        fairness_lambda: the *initial* Lagrange multiplier; dual ascent moves it.
        max_wait_s: cap on the mean per-vehicle waiting time of the worst approach lane.
        fairness_scale: fixed unit conversion putting the penalty on the same per-step
            scale as the delay term (see the module docstring).
        emissions_beta: weight on the emissions term (0 until Week 9).
    """

    coordination_alpha: float = 0.15
    fairness_lambda: float = 0.5
    # Chosen from measurement, not preference: an untrained shared policy on this
    # corridor already keeps the worst approach's mean per-vehicle wait near 45 s, so a
    # 120 s cap is never active and the "constraint" would do nothing. 30 s binds.
    max_wait_s: float = 30.0
    fairness_scale: float = 0.05
    emissions_beta: float = 0.0


class FairnessState:
    """Process-local holder for the Lagrange multiplier.

    Every RLlib env runner is a separate process with its own import of this module,
    so the driver keeps them in sync by broadcasting :func:`set_lambda`.
    """

    lam: float = 0.5

    @classmethod
    def get(cls) -> float:
        """Return the current multiplier."""
        return cls.lam

    @classmethod
    def set(cls, value: float) -> None:
        """Set the multiplier, clamped to be non-negative."""
        cls.lam = max(0.0, float(value))


def set_lambda(value: float) -> None:
    """Set this process's Lagrange multiplier (broadcast target)."""
    FairnessState.set(value)


def get_lambda() -> float:
    """Return this process's Lagrange multiplier."""
    return FairnessState.get()


def dual_ascent(lam: float, violation_ratio: float, step_size: float, lam_max: float = 2.0) -> float:
    """Take one dual-ascent step on the Lagrange multiplier.

    Standard constrained-MDP update: raise the price of the constraint while it is
    violated, lower it once satisfied, and never let it go negative.

    The update uses a *relative* violation, ``(max_lane_wait - cap) / cap``, rather
    than the raw second count. Raw violations on this corridor reach several hundred
    seconds, so an absolute update would drive ``lam`` to its clamp in a single
    iteration and turn the constraint back into a fixed penalty. Normalising keeps the
    dual variable on a scale comparable to the task reward.

    Args:
        lam: current multiplier.
        violation_ratio: ``(max_lane_wait - max_wait_s) / max_wait_s``; negative when
            the constraint is satisfied, which decays ``lam``.
        step_size: dual learning rate.
        lam_max: upper clamp, so a persistently infeasible constraint cannot let the
            penalty swamp the task reward.

    Returns:
        The updated multiplier in ``[0, lam_max]``.
    """
    return float(min(lam_max, max(0.0, lam + step_size * violation_ratio)))


class ShapedReward:
    """Callable reward function passed to sumo-rl as ``reward_fn``.

    sumo-rl calls this once per decision per junction with that junction's
    ``TrafficSignal``. Implemented as a class (not a closure) so RLlib can pickle it
    by reference into remote env runners.
    """

    def __init__(self, weights: RewardWeights | None = None, use_fairness: bool = True, use_emissions: bool = False,
                 use_coordination: bool = True) -> None:
        """Configure which shaping terms are active.

        Args:
            weights: reward weights; defaults to :class:`RewardWeights`.
            use_fairness: enable the Lagrangian fairness penalty.
            use_coordination: enable the green-wave term.
        """
        self.weights = weights or RewardWeights()
        self.use_fairness = use_fairness
        self.use_emissions = use_emissions
        self.use_coordination = use_coordination

    def __call__(self, ts: Any) -> float:
        """Compute the shaped reward for one traffic signal.

        Args:
            ts: a sumo-rl ``TrafficSignal``.

        Returns:
            The shaped scalar reward.
        """
        cfg = self.weights

        # ── base term: sumo-rl's diff-waiting-time, reproduced exactly ────────
        # The per-lane list is fetched once and reused for the fairness term, so
        # shaping costs no extra TraCI round-trips beyond the lane occupancies below.
        lane_waits = ts.get_accumulated_waiting_time_per_lane()
        ts_wait = sum(lane_waits) / 100.0
        reward = ts.last_measure - ts_wait
        ts.last_measure = ts_wait

        # ── green-wave coordination ──────────────────────────────────────────
        if self.use_coordination and cfg.coordination_alpha:
            out_density = ts.get_out_lanes_density()
            if out_density:
                reward -= cfg.coordination_alpha * (sum(out_density) / len(out_density))

        # ── emission smoothing (Week 9) ──────────────────────────────────────
        # CO2 per vehicle rather than total, for the same reason the fairness term
        # is per-vehicle: a total rewards emptying the junction, which the delay
        # term already does, and would double-count it. Converted mg -> g so the
        # weight is on the same scale as the delay term.
        if self.use_emissions and cfg.emissions_beta:
            co2_mg = sum(ts.sumo.lane.getCO2Emission(lane) for lane in ts.lanes)
            present = sum(ts.sumo.lane.getLastStepVehicleNumber(lane) for lane in ts.lanes)
            if present > 0:
                reward -= cfg.emissions_beta * (co2_mg / present) / 1000.0

        # ── Lagrangian fairness ──────────────────────────────────────────────
        # Convert each lane's summed accumulated wait into a per-vehicle mean, so the
        # cap means "seconds a vehicle waits" rather than "seconds summed over the
        # whole queue" (see the module docstring for why that distinction matters).
        max_lane_wait = 0.0
        for lane, summed_wait in zip(ts.lanes, lane_waits):
            occupancy = ts.sumo.lane.getLastStepVehicleNumber(lane)
            if occupancy > 0:
                max_lane_wait = max(max_lane_wait, summed_wait / occupancy)
        violation = max(0.0, max_lane_wait - cfg.max_wait_s)
        if self.use_fairness:
            # Two conversions, for two different reasons:
            #  - dividing by the cap makes the violation dimensionless, so the penalty
            #    does not change scale when the cap is retuned;
            #  - fairness_scale puts it in the delay term's per-step units. The delay
            #    term telescopes (its episode sum is just start-minus-end wait), while
            #    this penalty accumulates every step, so without the conversion the
            #    constraint dominates the return by two orders of magnitude.
            reward -= get_lambda() * cfg.fairness_scale * (violation / cfg.max_wait_s)

        # Stash per-junction fairness stats where the env can read them without
        # paying for a second get_accumulated_waiting_time_per_lane() call.
        stats = getattr(ts.env, "fairness_stats", None)
        if stats is None:
            stats = {}
            ts.env.fairness_stats = stats
        stats[ts.id] = {"max_lane_wait_s": float(max_lane_wait), "violation_s": float(violation)}

        return float(reward)


def make_reward_fn(name: str, weights: RewardWeights | None = None) -> Any:
    """Resolve a reward-function name to something sumo-rl accepts.

    Names are resolved inside each env runner, so only a string has to cross the
    process boundary in ``env_config``.

    Args:
        name: ``"diff-waiting-time"`` (Weeks 2-4 default), ``"shaped"`` (Week 5 full),
            ``"shaped-no-fairness"`` or ``"shaped-no-coordination"`` (Week 6 ablations).
        weights: reward weights for the shaped variants.

    Returns:
        Either a sumo-rl reward name (``str``) or a callable reward function.

    Raises:
        ValueError: if ``name`` is not a known reward function.
    """
    builtin = {"diff-waiting-time", "average-speed", "queue", "pressure"}
    if name in builtin:
        return name
    if name == "shaped":
        return ShapedReward(weights, use_fairness=True, use_coordination=True)
    if name == "shaped-no-fairness":
        return ShapedReward(weights, use_fairness=False, use_coordination=True)
    if name == "shaped-no-coordination":
        return ShapedReward(weights, use_fairness=True, use_coordination=False)
    raise ValueError(
        f"Unknown reward function '{name}'. "
        f"Expected one of {sorted(builtin)} or "
        "'shaped', 'shaped-no-fairness', 'shaped-no-coordination'."
    )
