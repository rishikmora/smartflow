"""Multi-agent corridor environment for SmartFlow Weeks 4-6.

Wraps sumo-rl's multi-agent ``SumoEnvironment`` in a PettingZoo ``ParallelEnv`` that
RLlib can consume through ``ParallelPettingZooEnv``. Going direct — rather than using
sumo-rl's AEC-to-parallel conversion — keeps ``env.sumo`` and ``env.traffic_signals``
reachable, which the evaluation harness (per-second metrics) and the Week 5 reward
shaping both need.

Observation alignment
---------------------
sumo-rl's default observation is laid out as::

    [ green-phase one-hot (P_i) | min-green flag (1) | density (L_i) | queue (L_i) ]

``P_i`` and ``L_i`` vary per junction — on this corridor the interior junctions have
4 controlled lanes (11 dims) and the edge junctions 3 (9 dims). Padding the flat
vector on the right would put one agent's *queue* values at the same indices as
another agent's *density* values, so a parameter-shared policy would be reading
different quantities from the same input unit.

:class:`ObservationAligner` therefore pads each **field** to the corridor-wide
maximum, so index *k* means the same thing for every agent. That is what makes the
Week 5 shared policy well-posed.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Sequence

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartflow_env import make_controlled_env_class, require_sumo_home

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


class ObservationAligner:
    """Pads per-agent observations field-by-field onto a common layout.

    Args:
        specs: ``{agent_id: (num_green_phases, num_lanes)}`` for every agent.
        include_agent_id: append a one-hot agent identity to each observation. Off by
            default so a shared policy has to generalise across junctions rather than
            memorise them; available as a Week 6 ablation.
    """

    def __init__(self, specs: dict[str, tuple[int, int]], include_agent_id: bool = False) -> None:
        if not specs:
            raise ValueError("specs must describe at least one agent.")
        self.specs = dict(specs)
        self.agents = sorted(specs)
        self.include_agent_id = include_agent_id
        self.max_phases = max(p for p, _ in specs.values())
        self.max_lanes = max(l for _, l in specs.values())
        self.dim = self.max_phases + 1 + 2 * self.max_lanes
        if include_agent_id:
            self.dim += len(self.agents)
        self._index = {agent: i for i, agent in enumerate(self.agents)}

    def align(self, agent: str, obs: Any) -> np.ndarray:
        """Re-pack one raw sumo-rl observation onto the common layout.

        Args:
            agent: the agent the observation belongs to.
            obs: raw observation from sumo-rl.

        Returns:
            A ``float32`` vector of length :attr:`dim`.

        Raises:
            ValueError: if the observation length does not match the agent's spec.
        """
        phases, lanes = self.specs[agent]
        raw = np.asarray(obs, dtype=np.float32).ravel()
        expected = phases + 1 + 2 * lanes
        if raw.shape[0] != expected:
            raise ValueError(
                f"Agent {agent}: expected observation of length {expected} "
                f"(phases={phases}, lanes={lanes}) but got {raw.shape[0]}."
            )

        out = np.zeros((self.dim,), dtype=np.float32)
        # green-phase one-hot
        out[:phases] = raw[:phases]
        # min-green flag always sits directly after the padded phase block
        out[self.max_phases] = raw[phases]
        # density, then queue, each padded to max_lanes
        d0 = self.max_phases + 1
        q0 = d0 + self.max_lanes
        out[d0 : d0 + lanes] = raw[phases + 1 : phases + 1 + lanes]
        out[q0 : q0 + lanes] = raw[phases + 1 + lanes : expected]
        if self.include_agent_id:
            out[self.max_phases + 1 + 2 * self.max_lanes + self._index[agent]] = 1.0
        return out

    def observation_space(self) -> spaces.Box:
        """Return the common observation space.

        Densities and queues are already normalised to ``[0, 1]`` by sumo-rl, so the
        aligned vector is bounded.
        """
        return spaces.Box(low=0.0, high=1.0, shape=(self.dim,), dtype=np.float32)


class CorridorParallelEnv:
    """PettingZoo-style parallel multi-agent corridor environment.

    Implements the subset of the ``ParallelEnv`` API that RLlib's
    ``ParallelPettingZooEnv`` uses: ``possible_agents``, ``agents``,
    ``observation_space(agent)``, ``action_space(agent)``, ``reset`` and ``step``.
    """

    metadata = {"name": "smartflow_corridor_v1", "is_parallelizable": True, "render_modes": []}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        *,
        controlled_ts: Sequence[str],
        num_seconds: int,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        seed: int = 0,
        reward_fn: Any = "diff-waiting-time",
        include_agent_id: bool = False,
        neighbor_context: bool = False,
        max_neighbors: int = 4,
        step_hook: Callable[[Any], None] | None = None,
        quiet: bool = True,
    ) -> None:
        """Create the multi-agent corridor environment.

        Args:
            net_file: SUMO ``.net.xml``.
            route_file: SUMO ``.rou.xml`` (Week 6 swaps this to change demand).
            controlled_ts: junction ids under RL control.
            num_seconds: simulated seconds per episode.
            delta_time: simulated seconds between agent decisions.
            yellow_time: yellow phase duration; must be < ``delta_time``.
            min_green: minimum green time per phase.
            max_green: maximum green time per phase.
            seed: SUMO RNG seed.
            reward_fn: sumo-rl reward name or ``callable(TrafficSignal) -> float``.
            include_agent_id: append a one-hot agent identity to observations.
            neighbor_context: append each junction's directly-downstream neighbours'
                observations (plus a presence flag per slot). Required by the
                graph-attention encoder, which needs neighbour features in the
                per-agent batch RLlib hands a policy.
            max_neighbors: neighbour slots per junction; this corridor's busiest
                junctions have four.
            step_hook: optional ``callable(sumo_connection)`` run once per simulated
                second (evaluation metrics).
            quiet: pass ``--no-step-log`` to SUMO.

        Raises:
            FileNotFoundError: if ``net_file`` or ``route_file`` is missing.
        """
        require_sumo_home()
        for label, path in (("net_file", net_file), ("route_file", route_file)):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        env_cls = make_controlled_env_class()
        self.env = env_cls(
            controlled_ts=list(controlled_ts),
            net_file=net_file,
            route_file=route_file,
            out_csv_name=None,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            single_agent=False,
            use_gui=False,
            sumo_warnings=False,
            sumo_seed=seed,
            reward_fn=reward_fn,
            add_system_info=False,
            step_hook=step_hook,
            additional_sumo_cmd="--no-step-log" if quiet else None,
        )

        self.possible_agents: list[str] = list(self.env.ts_ids)
        self.agents: list[str] = list(self.possible_agents)
        specs = {
            ts: (self.env.traffic_signals[ts].num_green_phases, len(self.env.traffic_signals[ts].lanes))
            for ts in self.possible_agents
        }
        self.aligner = ObservationAligner(specs, include_agent_id=include_agent_id)
        self._act_spaces = {ts: self.env.action_spaces(ts) for ts in self.possible_agents}
        self._seed = seed

        self.neighbor_context = neighbor_context
        self.max_neighbors = int(max_neighbors)
        self.node_dim = self.aligner.dim
        if neighbor_context:
            from smartflow_env import neighbor_map

            full = neighbor_map(net_file, self.possible_agents)
            # Fixed-width neighbour slots keep the observation space rectangular,
            # which is what RLlib's Box space and the batched GAT both require.
            self.neighbors = {
                ts: full.get(ts, [])[: self.max_neighbors] for ts in self.possible_agents
            }
            obs_dim = self.node_dim + self.max_neighbors * (self.node_dim + 1)
        else:
            self.neighbors = {ts: [] for ts in self.possible_agents}
            obs_dim = self.node_dim
        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Per-episode fairness telemetry, populated by rewards.ShapedReward via
        # env.fairness_stats. Stays at zero when the default reward is used.
        self.episode_max_lane_wait_s = 0.0
        self.episode_violation_s = 0.0
        self.episode_violating_steps = 0
        self.episode_steps = 0

    # ── spaces ───────────────────────────────────────────────────────────────
    def observation_space(self, agent: str) -> spaces.Box:
        """Return the (shared) observation space for an agent."""
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        """Return the action space for an agent."""
        return self._act_spaces[agent]

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Start a new episode.

        Args:
            seed: SUMO RNG seed for this episode; ``None`` keeps the previous seed.
            options: unused, accepted for API compatibility.

        Returns:
            ``(observations, infos)`` keyed by agent id.
        """
        raw = self.env.reset(seed=seed if seed is not None else self._seed)
        self.agents = list(self.possible_agents)
        self.episode_max_lane_wait_s = 0.0
        self.episode_violation_s = 0.0
        self.episode_violating_steps = 0
        self.episode_steps = 0
        self.env.fairness_stats = {}
        return self._align(raw), {agent: {} for agent in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        """Apply one joint action and advance ``delta_time`` simulated seconds.

        Args:
            actions: ``{agent_id: green phase index}``.

        Returns:
            ``(observations, rewards, terminations, truncations, infos)``.
        """
        raw_obs, raw_rewards, dones, _info = self.env.step({a: int(v) for a, v in actions.items()})
        done = bool(dones["__all__"])
        self.episode_steps += 1
        self._record_fairness()
        obs = self._align(raw_obs)
        rewards = {agent: float(value) for agent, value in raw_rewards.items()}
        terminations = {agent: False for agent in obs}
        truncations = {agent: done for agent in obs}
        infos: dict[str, dict] = {agent: {} for agent in obs}
        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        """Shut down the SUMO process."""
        self.env.close()

    def render(self, *args: Any, **kwargs: Any) -> None:
        """Rendering is not supported for headless corridor training."""
        return None

    def set_step_hook(self, step_hook: Callable[[Any], None] | None) -> None:
        """Install the per-simulated-second metrics hook."""
        self.env.set_step_hook(step_hook)

    @property
    def sumo(self) -> Any:
        """The live SUMO/TraCI connection, for metric collection."""
        return self.env.sumo

    @property
    def traffic_signals(self) -> dict[str, Any]:
        """The sumo-rl ``TrafficSignal`` objects, keyed by junction id."""
        return self.env.traffic_signals

    def _align(self, raw: dict[str, Any]) -> dict[str, np.ndarray]:
        """Align raw observations, optionally appending neighbour context.

        With ``neighbor_context`` the returned vector is::

            [ own node features | (neighbour features, present flag) x max_neighbors ]

        An absent or not-yet-reporting neighbour contributes zeros and a flag of 0,
        so the graph-attention encoder can mask it out.
        """
        aligned = {agent: self.aligner.align(agent, obs) for agent, obs in raw.items()}
        if not self.neighbor_context:
            return aligned

        out: dict[str, np.ndarray] = {}
        zeros = np.zeros((self.node_dim,), dtype=np.float32)
        for agent, own in aligned.items():
            parts: list[np.ndarray] = [own]
            for slot in range(self.max_neighbors):
                neighbours = self.neighbors.get(agent, [])
                if slot < len(neighbours) and neighbours[slot] in aligned:
                    parts.append(aligned[neighbours[slot]])
                    parts.append(np.ones((1,), dtype=np.float32))
                else:
                    parts.append(zeros)
                    parts.append(np.zeros((1,), dtype=np.float32))
            out[agent] = np.concatenate(parts).astype(np.float32)
        return out

    def _record_fairness(self) -> None:
        """Fold this step's per-junction fairness stats into episode aggregates.

        ``rewards.ShapedReward`` writes the stats while computing the reward, so this
        costs no extra TraCI calls. With the default reward nothing is written and the
        aggregates stay at zero.
        """
        stats = getattr(self.env, "fairness_stats", None)
        if not stats:
            return
        step_max = max(entry["max_lane_wait_s"] for entry in stats.values())
        step_violation = max(entry["violation_s"] for entry in stats.values())
        self.episode_max_lane_wait_s = max(self.episode_max_lane_wait_s, step_max)
        self.episode_violation_s += step_violation
        if step_violation > 0:
            self.episode_violating_steps += 1

    def fairness_summary(self) -> dict[str, float]:
        """Return this episode's fairness telemetry.

        Returns:
            ``max_lane_wait_s`` (worst per-lane accumulated wait seen this episode),
            ``mean_violation_s`` (average per-step constraint excess) and
            ``violation_rate`` (fraction of steps where any lane exceeded the cap).
        """
        steps = max(1, self.episode_steps)
        return {
            "max_lane_wait_s": float(self.episode_max_lane_wait_s),
            "mean_violation_s": float(self.episode_violation_s / steps),
            "violation_rate": float(self.episode_violating_steps / steps),
        }


def make_parallel_env(env_config: dict[str, Any] | None = None) -> CorridorParallelEnv:
    """Build :class:`CorridorParallelEnv` from a plain config dict.

    This is the single entry point RLlib registers, so every worker constructs the
    environment the same way. Defaults come from ``week4_config``.

    Args:
        env_config: overrides for any constructor argument, plus ``seed``.

    Returns:
        A configured :class:`CorridorParallelEnv`.
    """
    import week4_config as cfg
    from rewards import RewardWeights, make_reward_fn

    config = dict(env_config or {})

    # Only a reward *name* crosses the process boundary; the callable is built here,
    # inside the env runner, so env_config stays trivially serialisable.
    reward_name = config.get("reward_fn", "diff-waiting-time")
    if isinstance(reward_name, str):
        weight_overrides = config.get("reward_weights") or {}
        reward_fn = make_reward_fn(reward_name, RewardWeights(**weight_overrides))
    else:
        reward_fn = reward_name

    return CorridorParallelEnv(
        net_file=config.get("net_file", cfg.CORRIDOR_NET),
        route_file=config.get("route_file", cfg.CORRIDOR_ROUTE),
        controlled_ts=config.get("controlled_ts", cfg.CORRIDOR_TLS_IDS),
        num_seconds=int(config.get("num_seconds", cfg.SIM_SECONDS)),
        delta_time=int(config.get("delta_time", cfg.DELTA_TIME)),
        yellow_time=int(config.get("yellow_time", cfg.YELLOW_TIME)),
        min_green=int(config.get("min_green", cfg.MIN_GREEN)),
        max_green=int(config.get("max_green", cfg.MAX_GREEN)),
        seed=int(config.get("seed", 0)),
        reward_fn=reward_fn,
        include_agent_id=bool(config.get("include_agent_id", False)),
        neighbor_context=bool(config.get("neighbor_context", False)),
        max_neighbors=int(config.get("max_neighbors", 4)),
    )


def rllib_env_creator(env_config: dict[str, Any] | None = None) -> Any:
    """RLlib env creator: wrap :func:`make_parallel_env` for ``MultiAgentEnv``.

    Args:
        env_config: forwarded to :func:`make_parallel_env`.

    Returns:
        A ``ParallelPettingZooEnv`` around the corridor environment.
    """
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    return ParallelPettingZooEnv(make_parallel_env(env_config))
