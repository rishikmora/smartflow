# Week 8 — Vision, Anomaly Detection and the Scenario Planner

Three independent components, each with its own Definition of Done. Every number below is read from the recorded artifacts by `src/week8_report.py`; none of it is transcribed by hand.

## 1. Vehicle detection

> **DoD** — the vision model detects vehicles in a held-out clip.

**Verdict: MET.** `yolov8n.pt` trained for 40 epochs on CPU reaches **mAP50 0.988** and **mAP50-95 0.786** on junction C2, which it never saw in training.

| Split | Junctions | Images | Boxes | Cars | Trucks |
|---|---|---:|---:|---:|---:|
| train | B1, B2, C1 | 360 | 14156 | 11370 | 2786 |
| held out | C2 | 120 | 4958 | 3429 | 1529 |

| Metric | Value |
|---|---:|
| mAP50 | 0.9884 |
| mAP50_95 | 0.7864 |
| precision | 0.9864 |
| recall | 0.9820 |
| mAP50_car | 0.9822 |
| mAP50_truck | 0.9947 |

**The split holds out whole junctions, not random frames.** Frames of one junction sampled seconds apart are near-duplicates; a random split would put near-copies of every validation frame into training and report a score that means nothing. `src/test_week8.py` asserts that no held-out junction appears in the training split.

### What this does and does not demonstrate

The roadmap's substitution is "YOLOv8 trained on UA-DETRAC (public dataset) + rendered SUMO-GUI frames". Only the rendered half was possible here, for two separate reasons:

- **UA-DETRAC** is a ~5 GB download behind a registration wall and cannot be fetched unattended.
- **sumo-gui's own screenshot API** was tried first and proved unstable on this machine — it hung the GUI process partway through a capture run, twice, leaving orphaned processes. Frames are therefore rendered directly from the network geometry and the live vehicle state, which is headless, deterministic, and gives exact ground-truth boxes by construction.

So the honest reading of mAP50 ≈ 0.99 is **the pipeline works end to end**, not that the model is ready for real traffic cameras. The rendered task has no occlusion, no perspective, no lens weather, no motion blur and a clean background. A real-camera number would be substantially lower, and this project has not measured one.

Reproduce: `python src/vision_dataset.py` then `python src/train_yolo.py`. Frames come from 2 demand scenarios (base, peak).

## 2. Anomaly detection

> **DoD** — an injected anomaly is flagged automatically.

**Verdict: MET.** Across 3 seeds the detector flags **3 of 3** injected incidents (recall 1.00) at z=3.0 with 10 s persistence, with mean detection latency **30 s**.

| Measure | Value |
|---|---:|
| recall | 1.00 |
| precision | 0.47 |
| F1 | 0.64 |
| false alarms per episode | 9.3 |
| detection latency (s) | 30 |

Incidents are physical, not numbers poked into a CSV: `lane.setMaxSpeed` drops the lanes of `B1B2`, `C1C2`, `B2B3` to walking pace at known times, traffic backs up behind them, and the queue responds the way it would to a broken-down vehicle.

### Two things that had to be fixed to get here

- **Corridor-wide aggregation does not work.** The first version watched one summed queue signal and reached recall 0.33: a single obstructed lane is buried under the variance of twelve signal cycles. One detector per lane — which is what a real system with detector loops has — reaches recall 1.00.
- **A zero-variance window silenced the detector.** A lane that has been empty for the whole baseline window has σ=0, and the naive zero-guard scored every excursion as 0.0, so an always-empty lane that suddenly queued could never alarm. A floor of half a vehicle on σ fixes it. This was found by `test_detector_is_causal`, not by inspection.

### The precision/recall trade is reported, not hidden

Forty-eight independent detectors are a multiple-comparisons problem, so precision depends heavily on the threshold. The signals are recorded once and the detector replayed across a grid of thresholds and persistence values; the operating point above maximises F1 **among points with full recall**, because for a traffic controller a missed obstruction is a worse failure than an alert a human dismisses. The whole sweep is in `week8_anomaly_results.json` and charted in `week8_anomalies.png`.

A word on what precision around 0.5 means here, because it is easy to misread as the detector being wrong half the time. An alarm is scored false if it falls outside an injected incident window. But this corridor runs close to saturation and genuine congestion builds and clears on its own throughout the episode, so a flagged queue spike is often a real excursion that simply was not one of the three events this script injected. Without labels for naturally occurring congestion there is no way to separate those from true false alarms, so the figure is reported as measured and should be read as a lower bound on precision rather than an estimate of it.

## 3. Scenario planner

> **DoD** — one closure scenario completes end to end.

**Verdict: MET.** All 6 closure × weather combinations complete end to end over 3 seeds each.

| Closure | Weather | Avg wait (s) | Max queue | Throughput | CO2 (kg) |
|---|---|---:|---:|---:|---:|
| none | clear | 76.79 | 159 | 1598 | 625.4 |
| none | rain | 128.97 | 709 | 806 | 1047.5 |
| none | fog | 124.18 | 776 | 564 | 1262.5 |
| roadworks | clear | 92.28 | 277 | 1422 | 723.9 |
| roadworks | rain | 143.00 | 740 | 624 | 1174.4 |
| roadworks | fog | 123.82 | 824 | 450 | 1355.1 |

Restricting the central link **B1B2** costs **+20% average wait** and drops throughput from 1598 to 1422 vehicles.

**Weather dominates the closure.** Fog cuts throughput to 564 and rain to 806, against 1598 in the clear — a far larger effect than the roadworks. For a planner that is the useful finding: on this corridor, driving behaviour in bad weather matters more than losing one link.

One caveat on reading the table: fog shows a *lower* average wait than rain despite being the more severe profile. That is a metric interaction, not a contradiction — far fewer vehicles complete their trip under fog, and average wait is taken over completed trips. Throughput is the honest column there.

### How closures and weather are modelled

**Closure as roadworks, not a ban.** A total ban (`setDisallowed`) was tried first and rejected: many trips in `corridor.rou.xml` *end* on the closed edge, so banning it cancels them outright and SUMO aborts with "no valid route". With `--ignore-route-errors` it runs, but the throughput collapse (1520 → 520) then measures impossible trips rather than the cost of rerouting. Reducing the link to walking pace and making it expensive to route over degrades it without making any trip infeasible, which is the question a planner is actually asking.

**Weather as behaviour, not graphics.** Rain and fog are applied the way SUMO itself recommends — as changes to speed factor, headway (`tau`), deceleration and minimum gap. Values are in `week8_config.WEATHER`, and `test_week8.py` asserts fog is modelled at least as severe as rain on every axis.

Rerouting devices are enabled so traffic responds to the restriction the way a navigation system would, rather than driving into it regardless.

## Week 8 summary

| Component | DoD | Verdict |
|---|---|---|
| Vision | detects vehicles in a held-out clip | **MET** |
| Anomaly detection | injected anomaly flagged automatically | **MET** |
| Scenario planner | one closure scenario completes end to end | **MET** |

Verification: `python src/test_week8.py` — 10 checks covering the world-to-pixel projection, label validity, split integrity, detector causality, the persistence gate, scoring arithmetic and scenario consistency.
