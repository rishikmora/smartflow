# SmartFlow

Multi-agent digital twin platform for adaptive traffic signal control.  
B.Tech CSE final-year major project, CMRCET.

## Setup

Requirements: Python 3.11+, SUMO 1.27.1 (with `SUMO_HOME` set as a User environment variable).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install sumolib traci pandas matplotlib
```

## Commands

```powershell
# Verify environment (TraCI smoke test)
python src\day1_smoketest.py

# Run a single benchmark (appends to outputs/metrics.csv)
python src\eval.py --controller fixed    --seed 0
python src\eval.py --controller actuated --seed 0

# Regenerate actuated network (if corridor.net.xml changes)
python src\make_actuated_net.py

# Produce comparison chart
python src\compare_baselines.py
```

## File Map

```
smartflow/
  configs/
    corridor.sumocfg           # fixed-time TLS, 1800 s
    corridor_actuated.sumocfg  # actuated TLS, same demand
  data/
    corridor.net.xml           # 4x4 synthetic grid, 12 signalled junctions
    corridor_actuated.net.xml  # same network, tlLogic type=actuated + minDur/maxDur
    corridor.rou.xml           # 1800 vehicles, ~1/s departure rate
  src/
    day1_smoketest.py          # TraCI connectivity proof
    eval.py                    # benchmark harness — central contract for all phases
    make_actuated_net.py       # converts fixed net to actuated (reproducible)
    compare_baselines.py       # reads metrics.csv, saves baseline_comparison.png
  outputs/
    metrics.csv                # append-only benchmark log
    baseline_comparison.png    # Week 1 results chart
```

## Week 1 Results (3-seed averages, 4x4 grid, 1800 vehicles/1800 s)

| Controller | Avg Wait (s) | Max Queue | Throughput | CO₂ (kg) |
|------------|-------------|-----------|------------|-----------|
| fixed      | 86.3        | 387       | 1294       | 731       |
| actuated   | **40.9**    | **76**    | **1647**   | **508**   |

Actuated beats fixed on all three primary metrics: −53 % wait, −80 % max queue, +27 % throughput.

The fixed controller showed high seed-to-seed variance (max queue: 195–573), indicating the
network operates near saturation under static timing. Actuated stabilises this dramatically
(max queue: 74–78 across seeds). This motivates the Week 2+ MARL investigation.

> Note: network is a synthetic 4x4 grid (OSM import was deferred — osmWebWizard requires
> browser interaction). This does not affect the validity of the baseline comparison.

## Architecture Note

`eval.py`'s CLI (`--controller`, `--seed`) and the `metrics.csv` column schema are fixed
contracts. Every later phase (RL, multi-agent, federated) is compared against these rows.
