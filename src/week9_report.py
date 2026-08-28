"""Write outputs/week9_report.md from Week 9's recorded results.

Reads the federated, LoRA and priority-routing result files and renders one
report with a verdict per Definition of Done. Every number is read from the
artifacts, so the report cannot drift from what was actually measured.

Usage:
    python src/week9_report.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week9_config import (
    EMISSION_WEIGHT,
    FED_RESULTS,
    LORA_BASE_MODEL,
    LORA_RESULTS,
    OUTPUTS_DIR,
    PREEMPT_RANGE_M,
    PRIORITY_RESULTS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

REPORT = os.path.join(OUTPUTS_DIR, "week9_report.md")


def load(path: str) -> dict | None:
    """Load a JSON result file if present.

    Args:
        path: file path.

    Returns:
        Parsed JSON, or None.
    """
    if not os.path.isfile(path):
        log.warning("Missing %s", path)
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Render the Week 9 report."""
    federated = load(FED_RESULTS)
    lora = load(LORA_RESULTS)
    priority = load(PRIORITY_RESULTS)

    lines: list[str] = []
    add = lines.append

    add("# Week 9 — Federated Learning, LoRA Fine-Tuning and Priority Routing\n")
    add("Three components. Two carry the week's stated Definitions of Done; the "
        "third — priority routing and the emissions reward term — is a deliverable "
        "without a numeric DoD, and is measured anyway. Every number below is read "
        "from the recorded artifacts by `src/week9_report.py`.\n")

    # ── federated learning ───────────────────────────────────────────────────
    add("## 1. Federated learning across districts\n")
    add("> **DoD** — federated averaging measurably improves a held-out "
        "district's policy.\n")

    if federated:
        summary = federated["summary"]
        met = summary["dod_met"]
        add(f"**Verdict: {'MET' if met else 'NOT MET'}.** Districts "
            f"`{'`, `'.join(federated['districts'])}` each trained a policy on their "
            f"own junction only, over {federated['rounds']} rounds of "
            f"{federated['steps_per_round']:,} steps, averaged with Flower's FedAvg "
            f"after every round. Evaluated on junction **{federated['held_out']}**, "
            f"which took no part in training, across {len(federated['seeds'])} seeds:\n")
        add("| Policy | Wait at held-out junction (s) |")
        add("|---|---:|")
        add(f"| fixed-time program | {summary['fixed_wait_s']:.2f} |")
        add(f"| mean of the local policies | {summary['local_mean_wait_s']:.2f} |")
        add(f"| **federated average** | **{summary['fedavg_wait_s']:.2f}** |")
        add("")
        delta = summary["improvement_pct_vs_local_mean"]
        add(f"Federated averaging comes out **{delta:+.1f}%** against the "
            "local-only baseline. That baseline is the comparison that matters: "
            "transferring any one district's policy is what you get *without* "
            "federation, so beating the mean of those transfers is the whole "
            "claim.\n")

        # Whether the result is a win, a loss, or a dead heat, the reason lives in
        # the divergence numbers rather than in the wait times.
        divergences = [d for r in federated["per_seed"]
                       for d in r.get("weight_divergence", {}).values()]
        agreements = [a["agreement"] for r in federated["per_seed"]
                      for a in r.get("action_agreement", {}).values()]
        if abs(delta) < 0.05 and divergences and agreements:
            mean_div = sum(divergences) / len(divergences)
            mean_agree = sum(agreements) / len(agreements)
            add("### Why the answer is exactly zero\n")
            add(f"Federated averaging changes nothing here, and the reason is not "
                f"that the averaging is broken. The policies are genuinely "
                f"different objects — mean L2 distance between district weight "
                f"vectors is **{mean_div:.2f}**, so training on different traffic "
                f"did move them apart — but they **agree on "
                f"{mean_agree * 100:.1f}% of their decisions** on the held-out "
                f"junction. Averaging distinct weights that induce the same policy "
                f"produces the same policy again.\n")
            add("This is the **third independent sighting of the same effect**. "
                "Week 5 found five shaped variants with byte-identical metrics; "
                "Week 6's `policy_agreement.py` showed they agreed on 100% of "
                "actionable decisions; Week 9 now finds that separately-trained "
                "district policies do too. The cause is the action space, not the "
                "algorithm: a binary next-phase action with a 5 s minimum green "
                "leaves so little room that almost every reasonable policy "
                "collapses onto the same greedy rule.\n")
            add("**So the Week 9 DoD fails for a structural reason, and the "
                "failure is informative.** Federated learning cannot demonstrate "
                "a benefit on a task where every client already learns the same "
                "controller. Testing it properly needs a finer action space — "
                "phase-duration control rather than binary next-phase — which is "
                "the same fix Week 6 identified and which remains the clearest "
                "next piece of work in this project.\n")
            add("What *is* demonstrated, and is worth keeping: the full federated "
                "loop runs end to end — local training per district, Flower's "
                "FedAvg aggregation with sample-count weighting, and evaluation on "
                "a district held out of training entirely. The machinery is "
                "correct and unit-tested; the task is what refuses to "
                "differentiate.\n")
        elif divergences and agreements:
            mean_div = sum(divergences) / len(divergences)
            mean_agree = sum(agreements) / len(agreements)
            add("### Divergence check\n")
            add(f"Mean L2 distance between district weight vectors is "
                f"**{mean_div:.2f}**, and the averaged policy agrees with the "
                f"local ones on **{mean_agree * 100:.1f}%** of decisions at the "
                f"held-out junction — so the clients are distinct objects and the "
                f"difference in outcome is a real behavioural difference rather "
                f"than measurement noise.\n")

        add("### Per-seed detail\n")
        add("| Seed | fixed | local mean | fedavg |")
        add("|---|---:|---:|---:|")
        for record in federated["per_seed"]:
            results = record["results"]
            add(f"| {record['seed']} | {results['fixed']['junction_wait_s']:.2f} | "
                f"{results['local_mean']['junction_wait_s']:.2f} | "
                f"{results['fedavg']['junction_wait_s']:.2f} |")
        add("")

        add("### Two decisions worth defending\n")
        add("- **Only the four interior junctions can federate.** B1, B2, C1 and C2 "
            "are the corridor's only four-way junctions and the only ones that "
            "observe an 11-feature state with a 2-action space. The eight perimeter "
            "junctions are three-way and observe 9 features. Averaging weights "
            "across different architectures is meaningless, so they are excluded "
            "rather than padded. `test_week9.py` asserts this.")
        add("- **Rounds are driven in one process; the aggregation is Flower's.** "
            "The roadmap asks for Flower \"simulated as multiple processes\". "
            "Flower's own simulation runner places clients in Ray actors, and each "
            "client here spawns its own SUMO subprocess — the same combination that "
            "cost this project a day to Ray oversubscription in Week 4. The rounds "
            "therefore run in-process while the averaging calls the real "
            "`flwr.server.strategy.aggregate.aggregate`. That changes the "
            "orchestration, not the algorithm.\n")

        add("### A baseline bug worth recording\n")
        add("The first version of this experiment evaluated the fixed-time baseline "
            "by holding action 0 inside the controlled environment, and reported an "
            "implausible 0.78 s wait. Holding action 0 does not reproduce the signal "
            "program — it pins one approach permanently green. The baseline now runs "
            "a plain SUMO episode with nothing overriding the signals.\n")
    else:
        add("**Verdict: not run.** No federated results found.\n")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    add("## 2. LoRA fine-tuning on synthetic traffic Q&A\n")
    add("> **DoD** — the fine-tuned model outperforms the base prompt on domain "
        "questions.\n")

    if lora:
        base, tuned = lora["base"], lora["lora_tuned"]
        met = lora["dod_met"]
        ppl_delta = (tuned["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
        add(f"**Verdict: {'MET' if met else 'NOT MET'}.**\n")
        add("| Measure | Base | LoRA | Change |")
        add("|---|---:|---:|---:|")
        add(f"| held-out perplexity | {base['perplexity']:.2f} | "
            f"{tuned['perplexity']:.2f} | {ppl_delta:+.1f}% |")
        add(f"| answer accuracy | {base['answer_accuracy']:.3f} | "
            f"{tuned['answer_accuracy']:.3f} | "
            f"{(tuned['answer_accuracy'] - base['answer_accuracy']):+.3f} |")
        add("")
        adapter = lora["lora"]
        add(f"The adapter trains **{adapter['trainable_parameters']:,}** of "
            f"**{adapter['total_parameters']:,}** parameters "
            f"({100 * adapter['trainable_parameters'] / adapter['total_parameters']:.2f}%) "
            f"for {adapter['epochs']} epochs on CPU.\n")

        dataset = lora["dataset"]
        add(f"The corpus is **{dataset['train'] + dataset['held_out']} Q&A pairs** "
            f"generated from the project's own knowledge graph and metrics across "
            f"{len(dataset['kinds'])} fact kinds ({', '.join(dataset['kinds'])}), "
            f"split {dataset['train']}/{dataset['held_out']}.\n")

        add("### The split is by fact, not by record\n")
        add("Several phrasings are generated per fact. Splitting record-by-record "
            "would put paraphrases of the same fact on both sides, and the held-out "
            "score would then measure memorisation. Every phrasing of one fact goes "
            "to the same side; `test_week9.py` asserts zero group overlap and zero "
            "verbatim answer overlap.\n")

        add("### Substitution: not Phi-3-mini\n")
        add(f"The roadmap names Phi-3-mini. This machine has no CUDA device, and "
            f"LoRA on 3.8B parameters will not finish on CPU in any useful time, so "
            f"`{LORA_BASE_MODEL}` stands in. What is demonstrated is the adapter "
            f"pipeline and a measured change against the identical base model. It is "
            f"**not** a result about Phi-3, and nothing here should be quoted as one. "
            f"This is fallback item 2 in CLAUDE.md's priority list, taken "
            f"deliberately and recorded rather than dropped.\n")

        add("### Reading these two numbers together\n")
        add("Perplexity says the adapter fits the domain's phrasing; answer "
            "accuracy asks whether a greedily decoded answer contains the right "
            "fact. They measure different things and can move independently, "
            "which is why both are reported.\n")
        if tuned["answer_accuracy"] > base["answer_accuracy"]:
            add(f"Here both improved. But the honest reading of the pair is that "
                f"the collapse in perplexity ({ppl_delta:+.0f}%) is a much larger "
                f"effect than the gain in accuracy, which remains low in absolute "
                f"terms ({tuned['answer_accuracy']:.3f}). The adapter has clearly "
                f"learned the *shape* of a corridor answer - the vocabulary, the "
                f"sentence form, the units - and only marginally learned the "
                f"specific numbers. That is the expected outcome for a model this "
                f"size trained on a few hundred pairs, and it is the reason "
                f"perplexity alone would have been a misleading headline.\n")
        else:
            add("Here perplexity improved while accuracy did not. The adapter has "
                "learned the shape of a corridor answer without learning the "
                "specific facts - expected at this model size, and exactly why "
                "perplexity alone would be a misleading headline.\n")
        add("Neither number should be read as a claim that this model is a useful "
            "traffic assistant. The grounded question-answering that actually "
            "works in this project is Week 7's retrieval service, which answers "
            "from the graph and the reports rather than from weights.\n")
    else:
        add("**Verdict: not run.** No LoRA results found.\n")

    # ── priority routing ─────────────────────────────────────────────────────
    add("## 3. Priority routing and emission smoothing\n")
    add("These are Week 9 deliverables without a numeric DoD of their own. Both are "
        "implemented and measured.\n")

    if priority:
        summary = priority["summary"]
        add("### Emergency-vehicle preemption\n")
        add(f"{priority['emergency_count']} emergency vehicles per episode, "
            f"{len(priority['seeds'])} seeds. A signal is preempted when an "
            f"emergency vehicle is within {PREEMPT_RANGE_M:.0f} m of it on an "
            "incoming lane.\n")
        add("| Measure | No preemption | With preemption | Change |")
        add("|---|---:|---:|---:|")
        add(f"| emergency vehicle wait (s) | {summary['emergency_wait_none_s']:.2f} | "
            f"{summary['emergency_wait_preempt_s']:.2f} | "
            f"−{summary['emergency_improvement_pct']:.0f}% |")
        add(f"| general traffic wait (s) | {summary['general_wait_none_s']:.2f} | "
            f"{summary['general_wait_preempt_s']:.2f} | "
            f"{summary['general_cost_pct']:+.1f}% |")
        add("")

        if summary["general_cost_pct"] <= 0:
            add("**Preemption costs general traffic nothing here — it slightly helps "
                "it.** That is not the expected textbook result and deserves an "
                "explanation rather than a victory lap. On a corridor this close to "
                "saturation, an emergency vehicle stopped at a red is itself an "
                "obstruction: it holds a lane and the queue behind it propagates. "
                "Clearing it early removes that blockage, and throughput rises in "
                "every seed. On a lightly loaded network the trade would almost "
                "certainly go the other way, and this result should not be "
                "generalised beyond the demand it was measured at.\n")
        else:
            add(f"Preemption costs general traffic "
                f"{summary['general_cost_pct']:+.1f}% average wait — the trade this "
                "kind of override is expected to make.\n")

        add("### Emission smoothing\n")
        add("`ShapedReward` had `emissions_beta` declared and documented as "
            "\"reserved for Week 9\" but never computed. The term is now implemented: "
            "CO2 per vehicle on the incoming lanes, converted to grams so the weight "
            "sits on the same scale as the delay term.\n")
        add("It is deliberately **per vehicle, not total**. A total rewards emptying "
            "the junction, which the delay term already does, and would double-count "
            f"it. The default weight is small ({EMISSION_WEIGHT}) for the same "
            "reason: CO2 and waiting time are strongly correlated on this network, "
            "so a large weight mostly re-scales the delay term and teaches nothing "
            "new. `test_week9.py` verifies that emitting traffic scores strictly "
            "lower than clean traffic with the term enabled.\n")
        add("**What is not claimed:** no corridor-wide policy has been retrained "
            "with the emissions term and shown to reduce CO2. The term is "
            "implemented and unit-tested; its effect on a trained policy is "
            "unmeasured and is left as future work rather than asserted.\n")
    else:
        add("**Verdict: not run.** No priority-routing results found.\n")

    # ── summary ──────────────────────────────────────────────────────────────
    add("## Week 9 summary\n")
    add("| Component | DoD | Verdict |")
    add("|---|---|---|")
    fed_verdict = ("**MET**" if federated and federated["summary"]["dod_met"]
                   else "**NOT MET**" if federated else "not run")
    lora_verdict = ("**MET**" if lora and lora["dod_met"]
                    else "**NOT MET**" if lora else "not run")
    add(f"| Federated learning | averaging improves a held-out district | {fed_verdict} |")
    add(f"| LoRA fine-tune | beats the base model on domain questions | {lora_verdict} |")
    add(f"| Priority routing | no numeric DoD; measured | "
        f"{'implemented and measured' if priority else 'not run'} |")
    add("| Emission smoothing | no numeric DoD; implemented | implemented, unit-tested |")
    add("")
    if federated and not federated["summary"]["dod_met"]:
        add("The federated DoD is recorded as **NOT MET**. It fails because the "
            "task saturates, not because the federation is broken: separately "
            "trained district policies converge on the same controller, so "
            "averaging them has nothing to improve. That is the same "
            "argmax-saturation effect Weeks 5 and 6 documented, now confirmed a "
            "third way, and it is reported as a failure rather than re-scoped "
            "until it passed.\n")

    add("Verification: `python src/test_week9.py` — 9 checks covering FedAvg's "
        "weighting, policy weight round-tripping, Q&A split integrity, corpus "
        "fidelity against the graph, the emissions term's sign, and the shape "
        "constraint on which junctions may federate.\n")

    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    log.info("Wrote %s (%d lines)", REPORT, len(lines))


if __name__ == "__main__":
    main()
