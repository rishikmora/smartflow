"""Week 9: LoRA fine-tune a small language model on synthetic traffic Q&A.

The Q&A pairs are generated from the project's own knowledge graph and metrics,
so every answer in the training set is a fact about this corridor rather than
invented text. Questions are split by *instance*: the held-out set asks the same
kinds of question about junctions and controllers the model never saw answered,
so a model that simply memorised the training strings scores no better than the
base model.

**Substitution.** The roadmap names Phi-3-mini. This machine has no CUDA device,
and LoRA on 3.8B parameters will not finish on CPU in any useful time, so a much
smaller causal model stands in (see ``week9_config.LORA_BASE_MODEL``). What is
demonstrated is the adapter pipeline and a measured gain over the same base
model; it is not a result about Phi-3, and ``outputs/week9_report.md`` says so.

Two measures, both against the identical base model:

* **Held-out perplexity** - how well the model predicts unseen domain answers.
  Standard, unambiguous, and impossible to game with formatting.
* **Answer accuracy** - whether a greedily generated answer contains the correct
  fact. Perplexity can improve while output stays useless, so both are reported.

Usage:
    python src/lora_finetune.py
    python src/lora_finetune.py --epochs 2      # quick pipeline check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week9_config import (
    LORA_ALPHA,
    LORA_BASE_MODEL,
    LORA_BATCH,
    LORA_DATASET,
    LORA_DIR,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_LR,
    LORA_MAX_LEN,
    LORA_RANK,
    LORA_RESULTS,
    LORA_SEED,
    LORA_TRAIN_FRACTION,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

PROMPT = "Question: {q}\nAnswer:"


def build_qa() -> list[dict[str, str]]:
    """Generate the Q&A corpus from the corridor graph and metrics.

    Several phrasings are produced per fact. That is not padding: with one
    phrasing each, the corpus is 81 pairs and the adapter has nothing to
    generalise from. Paraphrases force the model to learn the fact rather than
    one surface string, which is the whole point of the held-out split.

    Returns:
        Records of ``{"question", "answer", "fact", "kind"}``, where ``fact`` is
        the substring a generated answer must contain to count as correct.
    """
    from knowledge_graph import EmbeddedGraph, load_document

    document = load_document()
    graph = EmbeddedGraph(document)
    records: list[dict[str, str]] = []

    def add(kind: str, fact: str, pairs: list[tuple[str, str]],
            group: str = "") -> None:
        for question, answer in pairs:
            records.append({"question": question, "answer": answer,
                            "fact": str(fact), "kind": kind,
                            "group": group or f"{kind}:{fact}"})

    junctions = sorted({n["id"] for n in document["nodes"]
                        if n.get("label") == "Junction"})

    for junction in junctions:
        neighbours = graph.neighbours(junction)
        if neighbours:
            joined = ", ".join(neighbours)
            add("topology", neighbours[0], group=f"topology:{junction}", pairs=[
                (f"Which junctions does {junction} feed into?",
                 f"{junction} feeds into {joined}."),
                (f"What are the downstream junctions of {junction}?",
                 f"The downstream junctions of {junction} are {joined}."),
                (f"Where can traffic from {junction} go next?",
                 f"Traffic from {junction} can go to {joined}."),
            ])
            add("degree", len(neighbours), group=f"degree:{junction}", pairs=[
                (f"How many junctions does {junction} feed into?",
                 f"{junction} feeds into {len(neighbours)} junctions."),
                (f"What is the out-degree of junction {junction}?",
                 f"Junction {junction} has an out-degree of {len(neighbours)}."),
            ])

        lanes = graph.lanes_of(junction)
        if lanes:
            add("lanes", len(lanes), group=f"lanes:{junction}", pairs=[
                (f"How many incoming lanes does junction {junction} have?",
                 f"Junction {junction} has {len(lanes)} incoming lanes."),
                (f"What is the incoming lane count at {junction}?",
                 f"The incoming lane count at {junction} is {len(lanes)}."),
            ])

        sensors = graph.sensors_of(junction)
        if sensors:
            add("sensors", len(sensors), group=f"sensors:{junction}", pairs=[
                (f"How many sensors does junction {junction} have?",
                 f"Junction {junction} has {len(sensors)} sensors."),
                (f"How many detector loops are installed at {junction}?",
                 f"There are {len(sensors)} detector loops at {junction}."),
            ])

        program = graph.program_of(junction)
        if program and program.get("phases"):
            phases = program["phases"]
            cycle = sum(int(p.get("duration_s", 0)) for p in phases)
            add("program", cycle, group=f"program:{junction}", pairs=[
                (f"How long is the signal cycle at junction {junction}?",
                 f"The signal cycle at {junction} is {cycle} seconds."),
                (f"What is the cycle time of {junction}?",
                 f"The cycle time of {junction} is {cycle} seconds."),
            ])
            add("phases", len(phases), group=f"phases:{junction}", pairs=[
                (f"How many phases does junction {junction} run?",
                 f"Junction {junction} runs {len(phases)} phases."),
                (f"What is the phase count of the program at {junction}?",
                 f"The program at {junction} has {len(phases)} phases."),
            ])

        node = graph.junction(junction)
        if node and node.get("x") is not None:
            add("position", str(int(float(node["x"]))), group=f"position:{junction}", pairs=[
                (f"Where is junction {junction} located?",
                 f"Junction {junction} is at x={int(float(node['x']))}, "
                 f"y={int(float(node['y']))}."),
            ])

    for rule in graph.rules():
        value = rule.get("value", rule.get("mean"))
        add("rule", value, group=f"rule:{rule['id']}", pairs=[
            (f"What is the value of the {rule['id']} rule?",
             f"The {rule['id']} rule is set to {value}."),
            (f"What does {rule['id']} control and what is it set to?",
             f"{rule['id']} is a signal-timing constant set to {value}."),
        ])

    # Result rows key their number as "mean", not "value". An earlier version of
    # this function read "value", found None every time, and silently dropped all
    # 69 measured results from the corpus.
    for row in graph.results():
        controller = row.get("controller")
        scenario = row.get("scenario")
        metric = row.get("metric")
        mean = row.get("mean")
        if not all((controller, scenario, metric)) or mean is None:
            continue
        std = row.get("std")
        seeds = row.get("seeds")
        add("result", mean, group=f"result:{controller}|{scenario}|{metric}", pairs=[
            (f"What {metric} did {controller} achieve on {scenario} demand?",
             f"{controller} achieved {metric} of {mean} on {scenario} demand."),
            (f"On {scenario} demand, what was the {metric} of {controller}?",
             f"On {scenario} demand {controller} recorded {metric} of {mean}."),
            (f"How many seeds back the {controller} {metric} result on {scenario}?",
             f"The {controller} {metric} result on {scenario} is a {seeds}-seed "
             f"mean of {mean} with standard deviation {std}."),
        ])

    log.info("Generated %d Q&A pairs from the graph across %d fact kinds",
             len(records), len({r["kind"] for r in records}))
    return records


def split(records: list[dict[str, str]], seed: int) -> tuple[list, list]:
    """Split records into train and held-out sets **by fact group**.

    Splitting record-by-record would put paraphrases of the same fact on both
    sides, and the held-out score would then measure memorisation rather than
    generalisation. Every phrasing of one fact goes to the same side.

    Args:
        records: the corpus.
        seed: shuffle seed.

    Returns:
        ``(train, held_out)``.
    """
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    cut = int(len(keys) * LORA_TRAIN_FRACTION)

    train = [r for key in keys[:cut] for r in groups[key]]
    held_out = [r for key in keys[cut:] for r in groups[key]]
    log.info("Split %d fact groups -> %d train / %d held-out records",
             len(keys), len(train), len(held_out))
    return train, held_out


def perplexity(model: Any, tokenizer: Any, records: list[dict[str, str]]) -> float:
    """Mean perplexity of the answers, conditioned on their questions.

    Only answer tokens are scored; the question is context, so a model is not
    rewarded for predicting text it was given.

    Args:
        model: a causal LM.
        tokenizer: its tokenizer.
        records: records to score.

    Returns:
        Mean perplexity across records.
    """
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for record in records:
            prompt = PROMPT.format(q=record["question"])
            full = prompt + " " + record["answer"]
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            full_ids = tokenizer(full, return_tensors="pt",
                                 truncation=True, max_length=LORA_MAX_LEN).input_ids
            if full_ids.shape[1] <= prompt_ids.shape[1]:
                continue
            labels = full_ids.clone()
            labels[:, :prompt_ids.shape[1]] = -100      # score the answer only
            out = model(input_ids=full_ids, labels=labels)
            losses.append(float(out.loss))
    if not losses:
        return float("nan")
    import math
    return math.exp(sum(losses) / len(losses))


def answer_accuracy(model: Any, tokenizer: Any,
                    records: list[dict[str, str]]) -> float:
    """Fraction of greedily generated answers containing the correct fact.

    Args:
        model: a causal LM.
        tokenizer: its tokenizer.
        records: records to score.

    Returns:
        Accuracy in 0-1.
    """
    import torch

    model.eval()
    hits = 0
    with torch.no_grad():
        for record in records:
            prompt = PROMPT.format(q=record["question"])
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            out = model.generate(
                ids, max_new_tokens=24, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if record["fact"] and record["fact"] in text:
                hits += 1
    return hits / len(records) if records else 0.0


def main() -> None:
    """Generate the corpus, fine-tune with LoRA, and compare against the base."""
    parser = argparse.ArgumentParser(description="LoRA fine-tune on traffic Q&A.")
    parser.add_argument("--epochs", type=int, default=LORA_EPOCHS)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    torch.manual_seed(LORA_SEED)
    random.seed(LORA_SEED)

    records = build_qa()
    train_records, held_out = split(records, LORA_SEED)
    os.makedirs(os.path.dirname(LORA_DATASET), exist_ok=True)
    with open(LORA_DATASET, "w", encoding="utf-8") as handle:
        json.dump({"train": train_records, "held_out": held_out}, handle, indent=2)
    log.info("Train %d / held out %d", len(train_records), len(held_out))

    tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Scoring the base model...")
    base = AutoModelForCausalLM.from_pretrained(LORA_BASE_MODEL)
    base_ppl = perplexity(base, tokenizer, held_out)
    base_acc = answer_accuracy(base, tokenizer, held_out)
    log.info("  base: perplexity %.2f, answer accuracy %.3f", base_ppl, base_acc)

    log.info("Fine-tuning with LoRA (r=%d, alpha=%d) for %d epochs on CPU...",
             LORA_RANK, LORA_ALPHA, args.epochs)
    model = AutoModelForCausalLM.from_pretrained(LORA_BASE_MODEL)
    config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=["c_attn"],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("  trainable %s of %s parameters (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        texts = [PROMPT.format(q=q) + " " + a + tokenizer.eos_token
                 for q, a in zip(batch["question"], batch["answer"])]
        return tokenizer(texts, truncation=True, max_length=LORA_MAX_LEN,
                         padding="max_length")

    dataset = Dataset.from_list(train_records).map(
        tokenize, batched=True, remove_columns=["question", "answer", "fact", "kind", "group"])

    os.makedirs(LORA_DIR, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(LORA_DIR, "run"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=LORA_BATCH,
            learning_rate=LORA_LR,
            logging_steps=25,
            save_strategy="no",
            report_to=[],
            seed=LORA_SEED,
            use_cpu=True,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    log.info("Scoring the fine-tuned model...")
    tuned_ppl = perplexity(model, tokenizer, held_out)
    tuned_acc = answer_accuracy(model, tokenizer, held_out)
    log.info("  LoRA: perplexity %.2f, answer accuracy %.3f", tuned_ppl, tuned_acc)

    model.save_pretrained(os.path.join(LORA_DIR, "adapter"))

    improved = tuned_ppl < base_ppl and tuned_acc >= base_acc
    log.info("")
    log.info("Held-out comparison (%d questions the model never saw answered):",
             len(held_out))
    log.info("  perplexity      base %8.2f -> LoRA %8.2f  (%+.1f%%)",
             base_ppl, tuned_ppl, (tuned_ppl - base_ppl) / base_ppl * 100)
    log.info("  answer accuracy base %8.3f -> LoRA %8.3f", base_acc, tuned_acc)
    log.info("DoD - fine-tuned beats base on domain questions: %s",
             "MET" if improved else "NOT MET")

    payload = {
        "base_model": LORA_BASE_MODEL,
        "roadmap_model": "Phi-3-mini (substituted: CPU-only machine)",
        "lora": {"r": LORA_RANK, "alpha": LORA_ALPHA, "epochs": args.epochs,
                 "trainable_parameters": trainable, "total_parameters": total},
        "dataset": {"train": len(train_records), "held_out": len(held_out),
                    "kinds": sorted({r["kind"] for r in records})},
        "base": {"perplexity": base_ppl, "answer_accuracy": base_acc},
        "lora_tuned": {"perplexity": tuned_ppl, "answer_accuracy": tuned_acc},
        "dod_met": bool(improved),
    }
    with open(LORA_RESULTS, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("Wrote %s", LORA_RESULTS)


if __name__ == "__main__":
    main()
