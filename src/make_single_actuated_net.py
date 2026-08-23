"""Generate a corridor network where only ONE junction is SUMO-actuated.

Week 3 puts a single junction under RL control and leaves the other eleven on
their fixed-time program. Comparing that against ``corridor_actuated.net.xml``
(where *all twelve* junctions are actuated) would confound two different effects:
a better controller at one junction, versus a better controller everywhere.

This script produces the missing control condition — one actuated junction, eleven
fixed-time — so ``ppo`` and ``actuated_single`` differ in exactly one thing: what
algorithm drives that one junction.

Follows the same rule as ``make_actuated_net.py``: an actuated ``tlLogic`` only
behaves differently from a static one if its green phases carry explicit
``minDur``/``maxDur``. Without them SUMO defaults both to the static duration and
the actuated controller is indistinguishable from fixed-time.

Usage:
    python src/make_single_actuated_net.py                 # defaults to B1
    python src/make_single_actuated_net.py --tls-id C2

Writes ``data/corridor_actuated_<tls_id>.net.xml`` and a matching ``.sumocfg``.
Output is deterministic; re-running overwrites in place.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_config import CONFIGS_DIR, CORRIDOR_NET, CORRIDOR_TLS_ID, DATA_DIR, SIM_SECONDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

MIN_DUR = 5    # minimum green (s) before the actuated controller may switch
MAX_DUR = 90   # maximum green (s) before it must switch
GREEN_PHASE_MIN_DURATION = 10  # phases shorter than this are yellow/intergreen

SUMOCFG_TEMPLATE = """<configuration>
    <input>
        <net-file value="../data/{net_name}"/>
        <route-files value="../data/corridor.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_end}"/>
    </time>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""


def _patch_phases(block: str) -> tuple[str, int]:
    """Add ``minDur``/``maxDur`` to the green phases inside one ``tlLogic`` block.

    Args:
        block: the raw XML text of a single ``<tlLogic>...</tlLogic>`` element.

    Returns:
        Tuple of (patched block, number of phases patched).
    """

    patched = 0

    def patch(match: re.Match) -> str:
        nonlocal patched
        duration = int(match.group(1))
        state = match.group(2)
        if duration < GREEN_PHASE_MIN_DURATION:
            return match.group(0)  # yellow/intergreen: leave fixed
        patched += 1
        return f'<phase duration="{duration}" minDur="{MIN_DUR}" maxDur="{MAX_DUR}" state={state}/>'

    # re.subn's own count includes matches returned unchanged, so count explicitly.
    new_block = re.sub(r'<phase duration="(\d+)"\s+state=("[^"]*")/>', patch, block)
    return new_block, patched


def build(tls_id: str, src_net: str = CORRIDOR_NET) -> tuple[str, str]:
    """Write a network with exactly one actuated junction, plus its sumocfg.

    Args:
        tls_id: the junction to convert to ``type="actuated"``.
        src_net: source ``.net.xml`` (all junctions static).

    Returns:
        Tuple of (net path, sumocfg path).

    Raises:
        FileNotFoundError: if ``src_net`` does not exist.
        ValueError: if ``tls_id`` has no ``tlLogic`` in the source network.
    """
    if not os.path.isfile(src_net):
        raise FileNotFoundError(f"Source network not found: {src_net}")
    with open(src_net, "r", encoding="utf-8") as handle:
        content = handle.read()

    pattern = re.compile(
        r'(<tlLogic\b[^>]*\bid="' + re.escape(tls_id) + r'"[^>]*>)(.*?)(</tlLogic>)',
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        raise ValueError(
            f"No <tlLogic> with id='{tls_id}' found in {src_net}. "
            "Check the id against the network's traffic lights."
        )

    header, body, footer = match.group(1), match.group(2), match.group(3)
    new_header = re.sub(r'\btype="static"', 'type="actuated"', header)
    if new_header == header:
        log.warning("tlLogic id=%s was not type='static'; header left unchanged.", tls_id)
    new_body, patched = _patch_phases(body)
    if patched == 0:
        raise ValueError(
            f"tlLogic id='{tls_id}' has no phase with duration >= {GREEN_PHASE_MIN_DURATION}s. "
            "Without minDur/maxDur the actuated controller would behave as fixed-time."
        )

    content = content[: match.start()] + new_header + new_body + footer + content[match.end() :]

    net_name = f"corridor_actuated_{tls_id}.net.xml"
    net_path = os.path.join(DATA_DIR, net_name)
    with open(net_path, "w", encoding="utf-8") as handle:
        handle.write(content)

    cfg_path = os.path.join(CONFIGS_DIR, f"corridor_actuated_{tls_id}.sumocfg")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        handle.write(SUMOCFG_TEMPLATE.format(net_name=net_name, sim_end=SIM_SECONDS))

    log.info("Wrote %s (junction %s actuated, %d green phases patched)", net_path, tls_id, patched)
    log.info("Wrote %s", cfg_path)
    return net_path, cfg_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Make a corridor net with one actuated junction.")
    parser.add_argument("--tls-id", default=CORRIDOR_TLS_ID, help="Junction id to actuate.")
    args = parser.parse_args()
    build(args.tls_id)


if __name__ == "__main__":
    main()
