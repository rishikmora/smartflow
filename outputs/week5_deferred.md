# Week 5 Deferred Items

- Parameter-shared RLlib PPO training was not run.
- Green-wave, fairness, and emissions reward terms are implemented as reusable pure functions in `src/reward_wrappers.py`, but not yet wired into a PettingZoo wrapper.
- Online learning mode was not added to an RLlib trainer because the RLlib trainer is deferred.
- Phase B 1,000,000-timestep 3-seed training was not executed.
