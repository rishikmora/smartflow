# Week 5 Report

The reward-shaping core is now represented by `src/reward_wrappers.py`.

Current formula:

`r_total = r_local + alpha * r_coordination - lambda * fairness_penalty - beta * co2_penalty`

Defaults:

- `alpha = 0.1`
- `lambda = 0.5`
- `MAX_WAIT_S = 120`
- `beta = 0.001`

The pure function was sanity-checked locally. Training and policy comparison remain deferred because RLlib and long-running SUMO jobs are not available in this session.
