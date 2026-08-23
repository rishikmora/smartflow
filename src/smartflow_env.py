"""Shared SUMO/sumo-rl environment layer for SmartFlow Weeks 3-6.

Why this module exists
----------------------
``sumo_rl.SumoEnvironment`` builds a :class:`~sumo_rl.environment.traffic_signal.TrafficSignal`
for *every* traffic light in the network, and ``TrafficSignal._build_phases`` **overwrites
that junction's signal program** with a synthetic green/yellow phase list.

On a 12-junction corridor where only one junction is RL-controlled that is a silent
experiment-breaking bug: the other 11 junctions lose their real fixed-time plan and
are left cycling a degenerate 60 s-green program that no agent ever advances. Measured
effect on ``data/corridor.net.xml``: throughput collapses from ~1520 vehicles to ~56,
and the environment slows from ~168 to ~8 agent-steps/s because every step still pays
the TraCI cost of polling all 12 junctions.

:class:`ControlledSumoEnvironment` restricts ``ts_ids`` to the junctions actually under
RL control, so untouched junctions keep the signal program defined in the ``.net.xml``.
That is what makes "1 RL junction + 11 fixed-time junctions" a meaningful comparison
against the fixed-time baseline.

Expects ``SUMO_HOME`` to be set (a Windows User environment variable for this project).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def require_sumo_home() -> str:
    """Return ``SUMO_HOME``, failing with an actionable message if it is unset.

    Returns:
        Absolute path to the SUMO installation root.

    Raises:
        EnvironmentError: if ``SUMO_HOME`` is not set in the environment.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME is not set. On Windows set it as a User environment variable "
            "pointing at the SUMO install root (e.g. C:\\Program Files (x86)\\Eclipse\\Sumo), "
            "then restart the shell."
        )
    if not os.path.isdir(sumo_home):
        raise EnvironmentError(f"SUMO_HOME points at a missing directory: {sumo_home}")
    return sumo_home


def sumo_binary(gui: bool = False) -> str:
    """Return the absolute path to the ``sumo`` or ``sumo-gui`` binary.

    SUMO's binaries are not on PATH on Windows, so they are always resolved
    through ``SUMO_HOME``.

    Args:
        gui: return ``sumo-gui.exe`` instead of ``sumo.exe``.

    Returns:
        Absolute path to the requested SUMO binary.

    Raises:
        FileNotFoundError: if the resolved binary does not exist.
    """
    name = "sumo-gui.exe" if gui else "sumo.exe"
    path = os.path.join(require_sumo_home(), "bin", name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SUMO binary not found at {path}. Check the SUMO installation.")
    return path


# sumo-rl always starts SUMO with these three options, so any baseline that does not
# pass them is being simulated under different physics than the RL controllers.
#
# SUMO's own defaults differ: --time-to-teleport is 300 s, meaning a vehicle stuck for
# five minutes is teleported forward. That silently clears gridlock and inflates a
# baseline's throughput, while sumo-rl (-1) lets jams persist. On base demand the
# distinction is measurably irrelevant — a fixed-time run records zero teleports and
# byte-identical metrics either way — but the Week 6 `peak` scenario is designed to
# gridlock, which is exactly where it would bite. Passing them everywhere removes the
# question instead of relying on it staying harmless.
SUMO_SIM_OPTIONS = [
    "--max-depart-delay", "-1",
    "--waiting-time-memory", "1000",
    "--time-to-teleport", "-1",
]


def baseline_sumo_cmd(
    *, config: str | None = None, net_file: str | None = None, route_file: str | None = None,
    seed: int = 0, gui: bool = False,
) -> list[str]:
    """Build the SUMO command line for a non-RL baseline run.

    Uses :data:`SUMO_SIM_OPTIONS` so baselines and sumo-rl-driven RL runs experience
    identical simulation behaviour.

    Args:
        config: a ``.sumocfg`` to run. Mutually exclusive with ``net_file``/``route_file``.
        net_file: SUMO network, when not using a config.
        route_file: SUMO routes, when not using a config.
        seed: SUMO RNG seed.
        gui: launch ``sumo-gui``.

    Returns:
        The argument list for ``traci.start``.

    Raises:
        ValueError: if neither a config nor a net/route pair is given.
        FileNotFoundError: if a given input file does not exist.
    """
    cmd = [sumo_binary(gui)]
    if config:
        if not os.path.isfile(config):
            raise FileNotFoundError(f"SUMO config not found: {config}")
        cmd += ["-c", config]
    elif net_file and route_file:
        _validate_inputs(net_file, route_file)
        cmd += ["-n", net_file, "-r", route_file]
    else:
        raise ValueError("Provide either config=... or both net_file=... and route_file=...")
    cmd += ["--seed", str(seed), "--no-step-log", "--no-warnings", *SUMO_SIM_OPTIONS]
    return cmd


def _validate_inputs(net_file: str, route_file: str) -> None:
    """Fail early with a clear message when a SUMO input file is missing."""
    for label, path in (("net_file", net_file), ("route_file", route_file)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")


def _import_sumo_environment() -> type:
    """Import ``SumoEnvironment`` after validating SUMO_HOME.

    ``sumo_rl`` raises ``ImportError`` at *import time* if SUMO_HOME is unset, so the
    check happens first to produce a message that says what to fix.

    Returns:
        The ``sumo_rl.SumoEnvironment`` class.
    """
    require_sumo_home()
    try:
        from sumo_rl import SumoEnvironment
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            f"Could not import sumo_rl.SumoEnvironment: {exc}. "
            "Check that the venv is active and 'pip install sumo-rl' succeeded."
        ) from exc
    return SumoEnvironment


def make_controlled_env_class() -> type:
    """Build :class:`ControlledSumoEnvironment` bound to the installed ``SumoEnvironment``.

    The base class is imported lazily (it hard-fails without SUMO_HOME), so the
    subclass is created on demand rather than at module import.

    Returns:
        A ``SumoEnvironment`` subclass that restricts control to selected junctions.
    """
    base = _import_sumo_environment()

    class ControlledSumoEnvironment(base):  # type: ignore[misc, valid-type]
        """A ``SumoEnvironment`` that only controls a chosen subset of junctions.

        Junctions outside ``controlled_ts`` keep the signal program defined in the
        ``.net.xml`` file, so they behave exactly as they do under the fixed-time
        baseline. Passing every junction id reproduces stock sumo-rl behaviour.
        """

        def __init__(
            self,
            controlled_ts: Sequence[str],
            reward_fn: Any = "diff-waiting-time",
            step_hook: Callable[[Any], None] | None = None,
            **kwargs: Any,
        ) -> None:
            """Create the environment.

            Args:
                controlled_ts: junction ids placed under RL control. Must be non-empty.
                reward_fn: a sumo-rl reward name or a ``callable(TrafficSignal) -> float``
                    applied to every controlled junction.
                step_hook: optional ``callable(sumo_connection)`` invoked after every
                    *simulated second*. Used to sample evaluation metrics at the same
                    rate as the raw-TraCI baselines. Leave ``None`` during training.
                **kwargs: forwarded to ``SumoEnvironment``.

            Raises:
                ValueError: if ``controlled_ts`` is empty or names an unknown junction.
            """
            controlled = list(dict.fromkeys(controlled_ts))
            if not controlled:
                raise ValueError("controlled_ts must name at least one traffic light.")
            # A dict-valued reward_fn is sumo-rl's supported way to build TrafficSignal
            # objects for a subset of junctions only.
            super().__init__(reward_fn={ts: reward_fn for ts in controlled}, **kwargs)

            unknown = [ts for ts in controlled if ts not in self.ts_ids]
            if unknown:
                raise ValueError(
                    f"Unknown traffic light id(s) {unknown}. "
                    f"Network defines: {sorted(self.ts_ids)}"
                )
            # Restrict the agent set. Everything downstream (observations, rewards,
            # dones, info) iterates ts_ids, so this is the single point of control.
            self.ts_ids = controlled
            self.observations = {ts: None for ts in self.ts_ids}
            self.rewards = {ts: None for ts in self.ts_ids}
            self.controlled_ts = controlled
            self._step_hook = step_hook

        def set_step_hook(self, step_hook: Callable[[Any], None] | None) -> None:
            """Install (or clear) the per-simulated-second hook.

            The hook can only be attached after ``reset()`` in practice, because the
            metrics collector needs a live SUMO connection to build itself.
            """
            self._step_hook = step_hook

        def _sumo_step(self) -> None:
            """Advance SUMO one second, then notify the metrics hook.

            ``SumoEnvironment`` funnels every simulated second through this method,
            so overriding it is enough to sample at 1 Hz regardless of ``delta_time``.
            """
            self.sumo.simulationStep()
            if self._step_hook is not None:
                self._step_hook(self.sumo)

    return ControlledSumoEnvironment


def make_single_agent_env(
    net_file: str,
    route_file: str,
    controlled_ts: str,
    *,
    num_seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int,
    seed: int = 0,
    reward_fn: Any = "diff-waiting-time",
    use_gui: bool = False,
    add_system_info: bool = False,
    step_hook: Callable[[Any], None] | None = None,
    quiet: bool = True,
) -> Any:
    """Create a single-agent corridor environment controlling exactly one junction.

    Args:
        net_file: path to the SUMO ``.net.xml``.
        route_file: path to the SUMO ``.rou.xml``.
        controlled_ts: the one junction id placed under RL control.
        num_seconds: simulated seconds per episode.
        delta_time: simulated seconds between agent decisions.
        yellow_time: yellow phase duration, must be < ``delta_time``.
        min_green: minimum green time per phase.
        max_green: maximum green time per phase.
        seed: SUMO RNG seed.
        reward_fn: sumo-rl reward name or ``callable(TrafficSignal) -> float``.
        use_gui: launch ``sumo-gui`` instead of headless ``sumo``.
        add_system_info: collect network-wide stats each step (slow; off by default).
        step_hook: optional ``callable(sumo_connection)`` run after every simulated
            second, used by the evaluation harness to sample metrics.
        quiet: pass ``--no-step-log`` so SUMO does not flood training logs.

    Returns:
        A Gymnasium-compatible environment whose single agent is ``controlled_ts``.
    """
    _validate_inputs(net_file, route_file)
    env_cls = make_controlled_env_class()
    return env_cls(
        controlled_ts=[controlled_ts],
        step_hook=step_hook,
        additional_sumo_cmd="--no-step-log" if quiet else None,
        net_file=net_file,
        route_file=route_file,
        out_csv_name=None,
        num_seconds=num_seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        max_green=max_green,
        single_agent=True,
        use_gui=use_gui,
        sumo_warnings=False,
        sumo_seed=seed,
        reward_fn=reward_fn,
        add_system_info=add_system_info,
    )


def list_traffic_lights(net_file: str) -> list[str]:
    """Return every traffic-light id defined in a SUMO network.

    Uses ``sumolib`` so no SUMO process has to be started.

    Args:
        net_file: path to the SUMO ``.net.xml``.

    Returns:
        Sorted traffic-light ids.
    """
    if not os.path.isfile(net_file):
        raise FileNotFoundError(f"net_file not found: {net_file}")
    try:
        import sumolib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"sumolib is required to read {net_file}: {exc}") from exc
    net = sumolib.net.readNet(net_file)
    return sorted(node.getID() for node in net.getNodes() if node.getType() == "traffic_light")


def neighbor_map(net_file: str, tls_ids: Iterable[str]) -> dict[str, list[str]]:
    """Map each traffic light to the traffic lights it feeds directly.

    An edge from junction ``u`` to junction ``v`` makes ``v`` a downstream neighbour
    of ``u``. This is the adjacency used for green-wave reward shaping (Week 5) and
    for the graph-attention encoder (Week 4).

    Args:
        net_file: path to the SUMO ``.net.xml``.
        tls_ids: the traffic-light ids to include as graph nodes.

    Returns:
        ``{tls_id: [downstream tls ids]}`` with deterministic ordering.
    """
    if not os.path.isfile(net_file):
        raise FileNotFoundError(f"net_file not found: {net_file}")
    try:
        import sumolib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"sumolib is required to read {net_file}: {exc}") from exc

    wanted = set(tls_ids)
    net = sumolib.net.readNet(net_file)
    out: dict[str, set[str]] = {ts: set() for ts in wanted}
    for edge in net.getEdges():
        src = edge.getFromNode().getID()
        dst = edge.getToNode().getID()
        if src in wanted and dst in wanted and src != dst:
            out[src].add(dst)
    return {ts: sorted(neigh) for ts, neigh in out.items()}


def count_route_vehicles(route_file: str) -> int:
    """Count vehicles declared in a SUMO route file.

    Args:
        route_file: path to the SUMO ``.rou.xml``.

    Returns:
        Number of ``<vehicle>`` / ``<trip>`` entries.
    """
    if not os.path.isfile(route_file):
        raise FileNotFoundError(f"route_file not found: {route_file}")
    import xml.etree.ElementTree as ET

    root = ET.parse(route_file).getroot()
    return len(root.findall("vehicle")) + len(root.findall("trip"))
