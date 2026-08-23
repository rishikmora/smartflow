# Viva Prep Draft

1. Why PPO and not DQN or SAC?
PPO is a stable on-policy policy-gradient method that works naturally with discrete signal-phase actions and has reliable implementations in Stable-Baselines3 and RLlib. DQN can work for discrete actions but is often less stable with non-stationary traffic dynamics. SAC is strongest for continuous control, which is not the primary action space here.

2. What does the Lagrangian fairness constraint do mathematically?
It adds a penalty when a lane's average wait exceeds a chosen cap. In this project scaffold, the penalty is `lambda * max(0, avg_wait - MAX_WAIT_S)`, subtracted from reward.

3. Why is the LLM service read-only?
The LLM can summarize, explain, and flag anomalies, but it must never directly change simulation or RL actions. This prevents hallucinated or unsafe text output from entering the control loop.

4. How does the GNN encode neighbor state?
The corridor is represented as a graph where signalized junctions are nodes and road connectivity forms edges. A GAT encoder can enrich each local observation with learned neighbor context.

5. What is the sim-to-real gap and how do you address it?
The current corridor is synthetic, so real driver behavior, sensing noise, geometry, and enforcement constraints are missing. The correct mitigation is to validate on OSM-imported networks and real/surveyed demand before making deployment claims.

6. Why Flower for federated learning? What is FedAvg?
Flower provides a practical framework for simulating multiple clients. FedAvg averages client model updates into one global model after local training.

7. Why Chroma instead of Pinecone?
Chroma is local and free, which avoids billing and credential risk during a student project. It can later be swapped for a managed vector DB if needed.

8. What is the biggest limitation of your system?
The biggest limitation is that results currently come from synthetic simulation, not a calibrated real corridor.

9. What would you do differently given more time?
I would import a real OSM corridor earlier, calibrate demand from observed counts, and reserve more time for seed sweeps and ablation studies.

10. How does your result compare to Liu et al. (2025)?
This must be answered only after the citation and benchmark comparison are verified. The current repo should not claim superiority to papers without matched benchmarks.

11. Why sumo-rl? What is its default reward function?
sumo-rl provides Gymnasium and PettingZoo interfaces over SUMO traffic signals. The project uses `diff-waiting-time`, which rewards reductions in accumulated waiting time.

12. What is non-stationarity in MARL and did you observe it?
Non-stationarity means each agent's environment changes as other agents learn. The code scaffold notes this risk, but it has not been measured until multi-agent runs complete.

13. What was the actuated-controller bug and what caused it?
The Week 1 actuated controller initially lacked proper min/max duration settings. It was fixed by ensuring actuated signal phases have `minDur=5` and `maxDur=90`.

14. Why k3d instead of real Kubernetes?
k3d gives Kubernetes-compatible manifests locally with no cloud billing risk. It is appropriate for a zero-budget final-year project.

15. What does the Week 6 benchmark report prove that Week 3 did not?
Week 3 should prove single-agent PPO on one real corridor intersection. Week 6 is intended to prove corridor-wide multi-agent behavior, robustness, and fairness across stronger baselines.
