# Citation Verification

Every identifier below was resolved against the registry that issued it and the **registered title compared against the title this project cites**. A DOI that resolves is not the same as a DOI that resolves to the work you meant; only the second is worth anything in a bibliography.

Checked by `src/verify_citations.py` on 2026-08-28. **9 verified, 2 software without an identifier, 0 unresolved or mismatched.**

Publisher pages (IEEE Xplore, ScienceDirect) block automated fetching, so DOIs are checked against CrossRef — the registration agency's own record, which is what a DOI actually is — and preprints against the arXiv API.

## Verified

| Key | Cited title | Identifier | Registered title | Year | Match |
|---|---|---|---|---:|---:|
| `sumo` | Microscopic Traffic Simulation using SUMO | `10.1109/ITSC.2018.8569938` | Microscopic Traffic Simulation using SUMO | 2018 | 1.00 |
| `ppo` | Proximal Policy Optimization Algorithms | `arXiv:1707.06347` | Proximal Policy Optimization Algorithms | 2017 | 1.00 |
| `sb3` | Stable-Baselines3: Reliable Reinforcement Learning Implementations | `http://jmlr.org/papers/v22/20-1364.html` | Stable-Baselines3: Reliable Reinforcement Learning Implementations | ? | 1.00 |
| `rllib` | RLlib: Abstractions for Distributed Reinforcement Learning | `arXiv:1712.09381` | RLlib: Abstractions for Distributed Reinforcement Learning | 2017 | 1.00 |
| `gat` | Graph Attention Networks | `arXiv:1710.10903` | Graph Attention Networks | 2017 | 1.00 |
| `fedavg` | Communication-Efficient Learning of Deep Networks from Decentralized Data | `arXiv:1602.05629` | Communication-Efficient Learning of Deep Networks from Decentralized Data | 2016 | 1.00 |
| `flower` | Flower: A Friendly Federated Learning Research Framework | `arXiv:2007.14390` | Flower: A Friendly Federated Learning Research Framework | 2020 | 1.00 |
| `lora` | LoRA: Low-Rank Adaptation of Large Language Models | `arXiv:2106.09685` | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | 1.00 |
| `ua-detrac` | UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking | `10.1016/j.cviu.2020.102907` | UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking | 2020 | 1.00 |

### What each is used for

- **`sumo`** — The simulator every result in this project is measured in.
- **`ppo`** — The RL algorithm used in Weeks 2-6 and 9.
- **`sb3`** — The PPO implementation for single-agent work (Weeks 2-3, 9).
  - JMLR 22(268):1-8, 2021. JMLR assigns no DOIs, so this is verified against the journal's own article page. An arXiv identifier was guessed for this entry first and resolved to an unrelated paper - which is exactly what this script exists to catch.
- **`rllib`** — The multi-agent training stack for Weeks 4-6.
- **`gat`** — The graph-attention state encoder ablated in Weeks 4-6.
- **`fedavg`** — The averaging algorithm behind Week 9's federated experiment.
- **`flower`** — The federated learning framework used in Week 9.
- **`lora`** — The adapter method used in Week 9's fine-tune.
- **`ua-detrac`** — The vehicle-detection benchmark Week 8 was meant to use, and did not.
- **`sumo-rl`** — The Gymnasium/PettingZoo environment wrapper used from Week 2 onward.
  - Software with no DOI or preprint. Cited by repository, not fabricated.
- **`ultralytics`** — The detector architecture and training loop in Week 8.
  - Software with no DOI or accompanying paper for v8.

## Software cited without an identifier

These have no DOI and no accompanying paper. They are cited by repository rather than given a fabricated identifier.

| Key | Project | Repository |
|---|---|---|
| `sumo-rl` | SUMO-RL | https://github.com/LucasAlegre/sumo-rl |
| `ultralytics` | Ultralytics YOLOv8 | https://github.com/ultralytics/ultralytics |

## Unresolved or mismatched

None. Every identifier resolved to the work it is cited as.


## Reproducing this

```
python src/verify_citations.py
```

The script queries CrossRef and arXiv live, compares each registered title against the cited one, and rewrites this file. It is not a transcription of a manual check, so it cannot go stale silently.
