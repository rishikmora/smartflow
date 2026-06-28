"""
Day 1 smoketest: generates a tiny 2-node network, inserts one vehicle,
steps the simulation 20 times via TraCI, and prints speed each step.
Pass: prints SUCCESS with no errors.
"""
import os
import sys
import subprocess
import tempfile
import shutil

# ── sanity check ──────────────────────────────────────────────────────────────
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    print("ERROR: SUMO_HOME environment variable is not set.")
    sys.exit(1)

SUMO_BIN      = os.path.join(SUMO_HOME, "bin", "sumo.exe")
NETCONVERT    = os.path.join(SUMO_HOME, "bin", "netconvert.exe")

for path, label in [(SUMO_BIN, "sumo.exe"), (NETCONVERT, "netconvert.exe")]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at {path}")
        sys.exit(1)

import traci  # noqa: E402  (import after path check so error is clear)

# ── build tiny network in a temp dir ─────────────────────────────────────────
work_dir = tempfile.mkdtemp(prefix="smoketest_")

nodes_xml = os.path.join(work_dir, "nodes.nod.xml")
edges_xml  = os.path.join(work_dir, "edges.edg.xml")
net_xml    = os.path.join(work_dir, "tiny.net.xml")
routes_xml = os.path.join(work_dir, "tiny.rou.xml")
cfg_xml    = os.path.join(work_dir, "tiny.sumocfg")

with open(nodes_xml, "w") as f:
    f.write(
        '<nodes>\n'
        '    <node id="n0" x="0"   y="0"/>\n'
        '    <node id="n1" x="200" y="0"/>\n'
        '</nodes>\n'
    )

with open(edges_xml, "w") as f:
    f.write(
        '<edges>\n'
        '    <edge id="e0" from="n0" to="n1" numLanes="1" speed="13.9"/>\n'
        '</edges>\n'
    )

result = subprocess.run(
    [NETCONVERT,
     "--node-files", nodes_xml,
     "--edge-files", edges_xml,
     "--output-file", net_xml,
     "--no-warnings"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("ERROR: netconvert failed:\n", result.stderr)
    shutil.rmtree(work_dir, ignore_errors=True)
    sys.exit(1)

with open(routes_xml, "w") as f:
    f.write(
        '<routes>\n'
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.0"'
        ' maxSpeed="13.9" length="5"/>\n'
        '    <route id="r0" edges="e0"/>\n'
        '    <vehicle id="v0" type="car" route="r0" depart="0"/>\n'
        '</routes>\n'
    )

# use absolute paths in the cfg so SUMO can find the files regardless of cwd
with open(cfg_xml, "w") as f:
    f.write(
        '<configuration>\n'
        '    <input>\n'
        f'        <net-file value="{net_xml}"/>\n'
        f'        <route-files value="{routes_xml}"/>\n'
        '    </input>\n'
        '    <time><end value="100"/></time>\n'
        '</configuration>\n'
    )

# ── run via TraCI ─────────────────────────────────────────────────────────────
traci.start([SUMO_BIN, "-c", cfg_xml, "--no-step-log", "--no-warnings"])

print(f"{'Step':>4}  {'Vehicle':12}  {'Speed (m/s)':>12}")
print("-" * 34)

for step in range(20):
    traci.simulationStep()
    ids = traci.vehicle.getIDList()
    if ids:
        speed = traci.vehicle.getSpeed(ids[0])
        print(f"{step:>4}  {ids[0]:12}  {speed:>12.4f}")
    else:
        print(f"{step:>4}  {'(none)':12}  {'—':>12}")

traci.close()

# ── cleanup ───────────────────────────────────────────────────────────────────
shutil.rmtree(work_dir, ignore_errors=True)

print("\nSUCCESS")
