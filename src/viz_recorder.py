"""Record a real SUMO run into a compact replay file for the visualiser.

The frontend does not drive SUMO live. It replays what SUMO actually did, which
matters for three reasons: the same recording can be shown anywhere without a running
simulator, two controllers can be watched **on identical traffic** side by side, and a
demo cannot fail live in front of an examiner. Every frame in the output came out of
TraCI.

Encoding is deliberately terse because the whole point is to ship the replay inside a
single self-contained page:

* Coordinates are decimetres as integers (the corridor is 600 m square, so values fit
  in four digits) rather than floats.
* Each frame stores a flat ``[id, x, y, id, x, y, ...]`` array instead of a list of
  objects, which removes the repeated JSON keys that would otherwise dominate the file.
* Vehicle ids are kept so the player can match a car between frames and interpolate its
  motion. That is what lets the sample interval be coarse without the animation looking
  like a slideshow — heading and speed are then derived from consecutive positions
  rather than stored.

Traffic-light state is reduced from SUMO's per-link string (16 characters per junction)
to one character per *approach*, since all four movements from one incoming lane share a
signal head in this network.

Usage:
    python src/viz_recorder.py --controller fixed
    python src/viz_recorder.py --controller actuated
    python src/viz_recorder.py --controller marl --mode shared --tag w5
    python src/viz_recorder.py --controller marl --mode independent --scenario peak

Writes ``outputs/viz/<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricsCollector
from smartflow_env import baseline_sumo_cmd, require_sumo_home
from week3_config import CORRIDOR_TLS_ID
from week4_config import CORRIDOR_NET, CORRIDOR_TLS_IDS, OUTPUTS_DIR, SIM_SECONDS, checkpoint_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

VIZ_DIR = os.path.join(OUTPUTS_DIR, "viz")
DECIMETRES = 10.0          # coordinate scale factor
DEFAULT_STEP_S = 4         # simulated seconds between recorded frames


def _q(value: float) -> int:
    """Quantise a metre coordinate to integer decimetres."""
    return int(round(value * DECIMETRES))


def extract_network(net_file: str = CORRIDOR_NET) -> dict[str, Any]:
    """Read road geometry and signal-head placement straight from the SUMO network.

    Args:
        net_file: path to the ``.net.xml``.

    Returns:
        A dict with lane polylines, junction outlines and, per traffic light, one
        signal head per approach (stop-line position plus approach heading).
    """
    import sumolib

    net = sumolib.net.readNet(net_file)
    xmin, ymin, xmax, ymax = net.getBoundary()

    lanes = []
    for edge in net.getEdges():
        for lane in edge.getLanes():
            shape = lane.getShape()
            lanes.append({
                "p": [c for x, y in shape for c in (_q(x), _q(y))],
                "w": round(lane.getWidth(), 1),
            })

    junctions = []
    for node in net.getNodes():
        shape = node.getShape()
        junctions.append({
            "id": node.getID(),
            "tls": node.getType() == "traffic_light",
            "c": [_q(node.getCoord()[0]), _q(node.getCoord()[1])],
            "p": [c for x, y in shape for c in (_q(x), _q(y))],
        })

    return {
        "bounds": [_q(xmin), _q(ymin), _q(xmax), _q(ymax)],
        "lanes": lanes,
        "junctions": junctions,
    }


def signal_heads(conn: Any, net_file: str = CORRIDOR_NET) -> dict[str, Any]:
    """Locate one signal head per approach for every traffic light.

    SUMO's state string has one character per *link* (16 per junction here, four
    movements from each of four approaches). A driver sees one head per approach, so
    the links are grouped by incoming lane and the group's state is summarised.

    Args:
        conn: a live TraCI connection.
        net_file: path to the ``.net.xml``, for lane geometry.

    Returns:
        ``{"ids": [...], "heads": {tls_id: [{"x","y","dx","dy","links":[...]}, ...]}}``
    """
    import sumolib

    net = sumolib.net.readNet(net_file)
    heads: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    for tls_id in CORRIDOR_TLS_IDS:
        links = conn.trafficlight.getControlledLinks(tls_id)
        by_lane: dict[str, list[int]] = {}
        for index, link in enumerate(links):
            if not link:
                continue
            incoming = link[0][0]
            by_lane.setdefault(incoming, []).append(index)

        entries = []
        for lane_id, indices in by_lane.items():
            shape = net.getLane(lane_id).getShape()
            (x1, y1), (x2, y2) = shape[-2], shape[-1]
            length = math.hypot(x2 - x1, y2 - y1) or 1.0
            entries.append({
                # stop line sits at the downstream end of the incoming lane
                "x": _q(x2), "y": _q(y2),
                "dx": round((x2 - x1) / length, 3),
                "dy": round((y2 - y1) / length, 3),
                "links": indices,
            })
        heads[tls_id] = entries
        order.append(tls_id)

    return {"ids": order, "heads": heads}


class ReplayRecorder:
    """Samples vehicle positions, signal states and running metrics into frames."""

    def __init__(self, conn: Any, tls: dict[str, Any], step_s: int = DEFAULT_STEP_S) -> None:
        """Create the recorder.

        Args:
            conn: live TraCI connection.
            tls: output of :func:`signal_heads`.
            step_s: simulated seconds between recorded frames.
        """
        self.conn = conn
        self.tls = tls
        self.step_s = max(1, int(step_s))
        self.frames: list[dict[str, Any]] = []
        self._second = 0

    def sample(self) -> None:
        """Record one frame if this simulated second falls on the sample interval."""
        second = self._second
        self._second += 1
        if second % self.step_s:
            return

        conn = self.conn
        vehicles: list[int] = []
        for vid in conn.vehicle.getIDList():
            x, y = conn.vehicle.getPosition(vid)
            # Vehicle ids in this project's route files are plain integers; anything
            # else is hashed into a stable positive int so the player can still track it.
            try:
                numeric = int(vid)
            except ValueError:
                numeric = abs(hash(vid)) % 1_000_000
            vehicles.extend((numeric, _q(x), _q(y)))

        states = []
        for tls_id in self.tls["ids"]:
            raw = conn.trafficlight.getRedYellowGreenState(tls_id)
            # one character per approach: green if any of its links is green
            per_approach = []
            for head in self.tls["heads"][tls_id]:
                chars = [raw[i] for i in head["links"] if i < len(raw)]
                if any(c in "Gg" for c in chars):
                    per_approach.append("G")
                elif any(c in "yY" for c in chars):
                    per_approach.append("y")
                else:
                    per_approach.append("r")
            states.append("".join(per_approach))

        self.frames.append({"t": second, "v": vehicles, "s": states})

    def attach_metrics(self, collector: MetricsCollector) -> None:
        """Append the running metric snapshot to the frame just recorded.

        Args:
            collector: the collector being fed alongside this recorder.
        """
        if not self.frames:
            return
        row = collector.corridor_row()
        self.frames[-1]["m"] = [
            row["avg_wait_time_s"],
            row["max_queue_len"],
            row["throughput_veh"],
            round(float(row["total_co2_kg"]), 1),
        ]


def _write(payload: dict[str, Any], name: str) -> str:
    """Write a replay file and report its size.

    Args:
        payload: the replay document.
        name: output basename without extension.

    Returns:
        The path written.
    """
    os.makedirs(VIZ_DIR, exist_ok=True)
    path = os.path.join(VIZ_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    size = os.path.getsize(path) / 1024 / 1024
    log.info("Wrote %s (%d frames, %.2f MB)", path, len(payload["frames"]), size)
    return path


def record_baseline(controller: str, seed: int, scenario: str, step_s: int,
                    num_seconds: int) -> str:
    """Record a fixed-time or actuated run.

    Args:
        controller: ``"fixed"`` or ``"actuated"``.
        seed: SUMO seed.
        scenario: demand scenario name.
        step_s: seconds between frames.
        num_seconds: episode length.

    Returns:
        Path to the replay file.
    """
    require_sumo_home()
    import traci

    from scenarios import resolve_scenario

    net_file, route_file = resolve_scenario(scenario, actuated=(controller == "actuated"))
    traci.start(baseline_sumo_cmd(net_file=net_file, route_file=route_file, seed=seed))
    try:
        network = extract_network(net_file)
        tls = signal_heads(traci, net_file)
        recorder = ReplayRecorder(traci, tls, step_s)
        junction_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(CORRIDOR_TLS_ID)))
        collector = MetricsCollector(traci, junction_lanes=junction_lanes)
        while traci.simulation.getTime() < num_seconds:
            traci.simulationStep()
            collector.sample()
            recorder.sample()
            recorder.attach_metrics(collector)
    finally:
        traci.close()

    payload = {
        "meta": {
            "controller": controller, "seed": seed, "scenario": scenario,
            "step_s": step_s, "sim_seconds": num_seconds, "scale": DECIMETRES,
        },
        "network": network, "tls": tls, "frames": recorder.frames,
        "summary": collector.corridor_row(),
    }
    return _write(payload, f"{controller}_{scenario}_seed{seed}")


def record_marl(mode: str, tag: str, seed: int, scenario: str, step_s: int,
                num_seconds: int, reward: str, use_gnn: bool) -> str:
    """Record a trained multi-agent policy driving the corridor.

    Args:
        mode: ``"independent"`` or ``"shared"``.
        tag: checkpoint tag.
        seed: SUMO seed and checkpoint seed.
        scenario: demand scenario name.
        step_s: seconds between frames.
        num_seconds: episode length.
        reward: reward name used to build the env.
        use_gnn: whether the policy needs neighbour context.

    Returns:
        Path to the replay file.
    """
    require_sumo_home()
    import torch
    from ray.rllib.core.columns import Columns

    from eval_marl_corridor import load_multi_rl_module
    from marl_env import make_parallel_env
    from scenarios import resolve_scenario

    net_file, route_file = resolve_scenario(scenario)
    module = load_multi_rl_module(checkpoint_dir(mode, seed, tag))
    policy_for = (lambda a: a) if mode == "independent" else (lambda a: "shared_policy")

    env = make_parallel_env({
        "net_file": net_file, "route_file": route_file,
        "controlled_ts": CORRIDOR_TLS_IDS, "num_seconds": num_seconds,
        "seed": seed, "reward_fn": reward, "neighbor_context": use_gnn,
    })
    try:
        observations, _ = env.reset(seed=seed)
        network = extract_network(net_file)
        tls = signal_heads(env.sumo, net_file)
        recorder = ReplayRecorder(env.sumo, tls, step_s)
        junction_lanes = list(env.traffic_signals[CORRIDOR_TLS_ID].lanes)
        collector = MetricsCollector(env.sumo, junction_lanes=junction_lanes)

        def hook(_conn: Any) -> None:
            collector.sample()
            recorder.sample()
            recorder.attach_metrics(collector)

        env.set_step_hook(hook)
        while env.agents:
            actions: dict[str, int] = {}
            with torch.no_grad():
                for agent, observation in observations.items():
                    batch = {Columns.OBS: torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)}
                    out = module[policy_for(agent)].forward_inference(batch)
                    actions[agent] = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
            observations, _r, _t, _tr, _i = env.step(actions)
    finally:
        env.close()

    label = f"marl_{mode}" + (f"_{tag}" if tag else "")
    payload = {
        "meta": {
            "controller": label, "seed": seed, "scenario": scenario,
            "step_s": step_s, "sim_seconds": num_seconds, "scale": DECIMETRES,
        },
        "network": network, "tls": tls, "frames": recorder.frames,
        "summary": collector.corridor_row(),
    }
    return _write(payload, f"{label}_{scenario}_seed{seed}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Record a SUMO run for the visualiser.")
    parser.add_argument("--controller", choices=["fixed", "actuated", "marl"], required=True)
    parser.add_argument("--mode", choices=["independent", "shared"], default="shared")
    parser.add_argument("--tag", type=str, default="w5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", type=str, default="base")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_S)
    parser.add_argument("--num-seconds", type=int, default=SIM_SECONDS)
    parser.add_argument("--reward", type=str, default="diff-waiting-time")
    parser.add_argument("--gnn", action="store_true")
    args = parser.parse_args()

    if args.controller == "marl":
        record_marl(args.mode, args.tag, args.seed, args.scenario, args.step,
                    args.num_seconds, args.reward, args.gnn)
    else:
        record_baseline(args.controller, args.seed, args.scenario, args.step, args.num_seconds)


if __name__ == "__main__":
    main()
