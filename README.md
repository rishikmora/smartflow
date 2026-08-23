# 🚦 SmartFlow

> **Teaching traffic lights to think — so cars spend less time waiting and more time moving.**

SmartFlow is a research platform that uses **AI and simulation** to make traffic signals smarter.
Instead of lights that blindly switch every 42 seconds whether cars are there or not,
SmartFlow trains agents that *watch* traffic and decide in real time when to turn green.

Built as a B.Tech final-year project at CMRCET — but the results are real.

---

## 🧠 The Big Idea (explain it like I'm 10)

Imagine you're stuck at a red light. There are 20 cars behind you... and zero cars on the
green side. The light stays red for 40 more seconds anyway. Annoying, right?

That's **fixed-time** control — the city sets a timer and the lights just follow it forever,
no matter what's actually happening on the road.

SmartFlow replaces that timer with an **AI brain** that:
1. Watches how many cars are waiting at each intersection
2. Learns from millions of simulated cars which signal timing works best
3. Makes the green light last longer when there's a big queue, shorter when the road is empty

The result? **Fewer cars waiting, less fuel burned, cleaner air.**

---

## ✨ Week 1 Results

We ran both approaches on a **4×4 grid of 12 signalled intersections** with **1,800 vehicles**
over a 30-minute simulation — measured across 3 random seeds for fairness.

| What we measured | 🔴 Old (fixed timer) | 🟢 Smart (actuated) | Improvement |
|---|---|---|---|
| Average time a car waits | 86.3 seconds | **40.9 seconds** | **−53%** ✅ |
| Worst traffic jam (cars) | 387 cars | **76 cars** | **−80%** ✅ |
| Cars that reached their destination | 1,294 | **1,647** | **+27%** ✅ |
| CO₂ emitted | 731 kg | **508 kg** | **−31%** ✅ |

> The AI version got **27% more cars home** while making each car wait **half as long**.
> That's the power of watching traffic instead of ignoring it.

![Baseline comparison chart](outputs/baseline_comparison.png)

---

## 🗺️ How the Project is Structured

```
smartflow/
│
├── 📁 data/               ← The road network and vehicle routes
│   ├── corridor.net.xml          — the 4×4 grid map (12 traffic lights)
│   ├── corridor_actuated.net.xml — same map, but lights can think
│   └── corridor.rou.xml          — 1,800 trips for cars to take
│
├── 📁 configs/            ← Simulation settings
│   ├── corridor.sumocfg          — run with fixed-time lights
│   └── corridor_actuated.sumocfg — run with smart lights
│
├── 📁 src/                ← Python code
│   ├── day1_smoketest.py         — checks the simulation engine works
│   ├── eval.py                   — the benchmarking engine (most important file)
│   ├── make_actuated_net.py      — converts fixed lights → smart lights
│   └── compare_baselines.py      — draws the results chart
│
└── 📁 outputs/            ← Results (committed — this is the proof of work)
    ├── metrics.csv               — every benchmark run, one row each
    └── baseline_comparison.png   — the chart above
```

---

## ⚙️ Setup

You need **Python 3.11+** and **SUMO 1.27.1** installed, with `SUMO_HOME` set as a
Windows User environment variable pointing to your SUMO folder.

```powershell
# 1. Create an isolated Python environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install sumolib traci pandas matplotlib
```

---

## 🚀 Run It Yourself

```powershell
# Activate the environment first (once per terminal session)
.\venv\Scripts\Activate.ps1

# ✅ Check everything is working (should print SUCCESS)
python src\day1_smoketest.py

# 📊 Run a benchmark — results appear in outputs/metrics.csv
python src\eval.py --controller fixed    --seed 0
python src\eval.py --controller actuated --seed 0

# 📈 Draw the comparison chart — saved to outputs/baseline_comparison.png
python src\compare_baselines.py
```

---

## 🔬 How the Benchmark Works

`eval.py` is the heart of the project. Every time you run it, it:

1. Launches a full SUMO traffic simulation (1,800 simulated seconds)
2. Controls all 12 traffic lights using the chosen strategy
3. Watches every car — recording how long it waited, how far it got, how much CO₂ it emitted
4. Writes one row to `outputs/metrics.csv` when done

The `--seed` flag controls randomness in the simulation. Running three seeds (0, 1, 2)
and averaging the results ensures we're seeing a real pattern, not just luck.

---

## Current Implementation Status

The repository now contains the Week 3 corridor PPO pipeline scaffolding and later-week platform scaffolding:

- Week 3: corridor config, smoke test, PPO trainer, evaluator, comparison chart, and report writer.
- Week 4: multi-agent PettingZoo smoke test plus PyTorch Geometric graph/GAT scaffolds.
- Week 5/8/9: reusable reward shaping and anomaly detection utilities.
- Week 10: five FastAPI service skeletons, Dockerfiles, Docker Compose, production Compose, and CI health tests.
- Week 12: draft final report, viva prep, benchmark report, and citation/deferred-status files.

Important: long-running RL training, SUMO GUI recording, Auth0 setup, k3d/Helm deployment, citation verification, and GitHub push are not claimed as complete in this repo. They are documented in `outputs/week*_deferred.md` until run on a machine with SUMO configured, approved package installs, credentials, and compute time.

Week 3 entry points:

```powershell
python src\week3_smoketest.py
python src\train_ppo_corridor.py --timesteps 30000 --seed 0 --tag short
python src\eval_corridor.py --controller fixed --seeds 0 1 2
python src\eval_corridor.py --controller actuated --seeds 0 1 2
python src\eval_corridor.py --controller ppo --seeds 0 1 2
python src\compare_week3.py
```

Service health tests:

```powershell
pip install pytest fastapi httpx pydantic
python -m pytest tests\test_health.py
```

## 🗓️ 12-Week Roadmap

This is Week 1 of 12. Here's the journey ahead:

| Phase | Weeks | What gets built |
|---|---|---|
| **A — Foundation** | 1–3 | ✅ Simulation + baselines (you are here) → single-agent RL |
| **B — Multi-Agent** | 4–6 | AI agents at every intersection, talking to each other |
| **C — Features** | 7–9 | Vision (cameras), knowledge graph, federated learning |
| **D — Platform** | 10–11 | Microservices, Kubernetes, live dashboard |
| **E — Demo** | 12 | Final report, demo video, viva prep |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Traffic simulation | SUMO 1.27.1 + TraCI (Python API) |
| Reinforcement learning | Stable-Baselines3 → Ray RLlib + PyTorch |
| Backend services | FastAPI (5 domain services) |
| Frontend dashboard | Next.js 15 + Recharts |
| Database | SQLite → Supabase (Postgres) |
| Knowledge graph | Neo4j AuraDB + Chroma vector store |
| AI language model | Claude API |
| Containers | Docker → k3d (local Kubernetes) |
| Monitoring | Prometheus + Grafana + Loki |

---

## 👤 Author

**Rishik** — B.Tech CSE, CMRCET  
Solo build · 12-week window · 2026
