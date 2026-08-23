# Week 4 Non-Stationarity Notes

The project now has the multi-agent smoke-test and GNN encoder scaffold, but the RLlib training runs needed to observe non-stationarity were not executed in this local session.

Expected risks to inspect after training:

- Per-agent rewards may move in different directions because each signal changes the transition distribution seen by its neighbors.
- Independent PPO policies may show high seed variance compared with the Week 3 single-agent setup.
- Queue spillback can make local reward noisy when upstream or downstream agents behave poorly.

These are the reasons Week 5 moves to parameter sharing and cooperative reward shaping.
