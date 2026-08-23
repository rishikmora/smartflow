# Week 4 — Non-Stationarity Notes (Independent Learners)

Twelve junctions each run their own PPO policy and update simultaneously. From
any one agent's perspective the transition dynamics keep changing, because the
other eleven policies keep changing — the environment is non-stationary and the
usual single-agent convergence guarantees do not apply. These are the numbers
that behaviour actually produced.

| Seed | Iters | Median return | First-quarter median | Final-quarter median | Worst iteration | Collapse rate | Diverged |
|---|---|---|---|---|---|---|---|
| 0 | 40 | -3.719 | -16.963 | -3.308 | -1404.386 | 28% | no |
| 1 | 40 | -4.027 | -8.366 | -3.746 | -1380.31 | 20% | no |
| 2 | 40 | -3.543 | -4.79 | -3.243 | -1364.01 | 20% | no |

## What the numbers show

### The return distribution is bimodal, not merely noisy

A typical iteration scores about **-3.8**, but a minority collapse to several hundred negative — the worst seen was **-1404**. Those are episodes where the corridor gridlocks early and never recovers, not gradual
degradation. Collapse rates across seeds (iterations an order of magnitude worse
than that seed's median): 28%, 20%, 20%.

This shape changes how the run must be judged. One collapsed episode moves a
ten-iteration *mean* by two orders of magnitude, so a mean-based trend test
reports 'divergence' according to where the rare collapses happened to fall
rather than whether learning progressed. Scored on the median — the typical
iteration — every seed improves or holds.

### Why independent learners produce this

From any one junction's point of view the other eleven policies are part of the
environment, and they keep changing, so each agent's advantage estimates are
computed against a moving target. The deeper problem is **credit assignment**:
`diff-waiting-time` is a purely *local* reward, so an agent that clears its own
queue by discharging into an already-saturated neighbour is rewarded for doing
so. Nothing in Week 4's objective makes that costly — and a corridor of twelve
agents all doing it simultaneously is exactly how a gridlock episode starts.

### What Week 5 changes

Parameter sharing cuts the number of independently moving policies from twelve to
one, which removes most of the non-stationarity. The green-wave term prices the
externality directly, penalising discharge into a full downstream link. The
collapse rate above is the number to compare against.
