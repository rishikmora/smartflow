# Week 4 Report

Implemented:

- `src/week4_config.py` records all 12 corridor traffic-light IDs.
- `src/week4_smoketest.py` instantiates sumo-rl's parallel PettingZoo API and steps all agents with random actions.
- `src/build_corridor_graph.py` builds traffic-light adjacency from `data/corridor.net.xml`.
- `src/gnn_encoder.py` defines a two-layer GAT encoder scaffold.

Deferred:

- RLlib installation, 50,000-step smoke training, full 3-seed MARL training, and evaluation are deferred until package installation and compute approval.
