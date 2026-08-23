"""Orchestrate SmartFlow's Week 4-6 training and evaluation runs.

One entry point for every run behind the Phase B results, so the experiment matrix
is a reviewable artifact rather than a shell history someone has to reconstruct.

Concurrency is bounded on purpose. Ray sizes each cluster to the whole machine by
default, so launching training jobs without a cap spawns hundreds of worker processes
that fight over the same cores — measured on this machine, that turned ~50 env-steps/s
per job into ~13. ``--max-parallel`` times the per-job env-runner count should stay at
or below the core count.

Stages:

``week4``       independent policies, 3 seeds (Week 4 DoD)
``week5``       shared policy + green wave + Lagrangian fairness, 3 seeds, and the
                no-fairness ablation the Week 5 DoD is judged against
``week6``       the remaining ablations: no green-wave term, and the GAT encoder
``evals``       every controller across every demand scenario, sharded then merged
``online``      the Week 5 online-learning loop

Usage:
    python src/run_experiments.py --stage week4
    python src/run_experiments.py --stage evals --max-parallel 8
    python src/run_experiments.py --stage all
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week4_config import FULL_TIMESTEPS, LOGS_DIR, NUM_ENV_RUNNERS, OUTPUTS_DIR, ROOT, TRAIN_SEEDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
SRC = os.path.join(ROOT, "src")
EVAL_SHARD_DIR = os.path.join(OUTPUTS_DIR, "tmp_marl")
MARL_CSV = os.path.join(OUTPUTS_DIR, "marl_metrics.csv")

SCENARIOS = ["base", "light", "peak", "asymmetric"]


@dataclass
class Job:
    """One subprocess to run.

    Attributes:
        name: short identifier, also used for the log filename.
        args: arguments appended to ``python <script>``.
        script: script under ``src/`` to run.
        env: extra environment variables.
    """

    name: str
    args: list[str]
    script: str
    env: dict[str, str] = field(default_factory=dict)


def _python() -> str:
    """Return the interpreter to launch jobs with.

    Returns:
        The project venv's python if present, otherwise the current interpreter.
    """
    return PYTHON if os.path.isfile(PYTHON) else sys.executable


def training_jobs(stage: str, timesteps: int, runners: int) -> list[Job]:
    """Build the training-job list for a stage.

    Args:
        stage: ``"week4"``, ``"week5"`` or ``"week6"``.
        timesteps: environment-step budget per run.
        runners: env runners per run.

    Returns:
        The jobs to execute.
    """
    common = ["--timesteps", str(timesteps), "--num-env-runners", str(runners)]
    jobs: list[Job] = []

    if stage == "week4":
        for seed in TRAIN_SEEDS:
            jobs.append(Job(
                name=f"w4_independent_seed{seed}",
                script="train_marl_corridor.py",
                args=["--mode", "independent", "--seed", str(seed),
                      "--reward", "diff-waiting-time", *common],
            ))

    elif stage == "week5":
        for seed in TRAIN_SEEDS:
            jobs.append(Job(
                name=f"w5_shared_seed{seed}",
                script="train_marl_corridor.py",
                args=["--mode", "shared", "--seed", str(seed), "--reward", "shaped",
                      "--fairness", "--tag", "w5", *common],
            ))
        for seed in TRAIN_SEEDS:
            # The counterfactual the Week 5 fairness claim is judged against.
            jobs.append(Job(
                name=f"w5_nofair_seed{seed}",
                script="train_marl_corridor.py",
                args=["--mode", "shared", "--seed", str(seed),
                      "--reward", "shaped-no-fairness", "--tag", "w5nofair", *common],
            ))

    elif stage == "week6":
        for seed in TRAIN_SEEDS:
            jobs.append(Job(
                name=f"w6_nocoord_seed{seed}",
                script="train_marl_corridor.py",
                args=["--mode", "shared", "--seed", str(seed),
                      "--reward", "shaped-no-coordination", "--fairness",
                      "--tag", "w5nocoord", *common],
            ))
        for seed in TRAIN_SEEDS:
            jobs.append(Job(
                name=f"w6_gnn_seed{seed}",
                script="train_marl_corridor.py",
                args=["--mode", "shared", "--seed", str(seed), "--reward", "shaped",
                      "--fairness", "--gnn", "--tag", "w5gnn", *common],
            ))

    else:
        raise ValueError(f"Unknown training stage '{stage}'.")

    return jobs


def eval_jobs() -> list[Job]:
    """Build the full evaluation matrix.

    Baselines and the two headline RL controllers run on every demand scenario; the
    reward/architecture ablations run on base demand only, since their purpose is to
    isolate a reward term rather than to test generalisation.

    Returns:
        One job per (controller, scenario, seed), each writing its own CSV shard.
    """
    os.makedirs(EVAL_SHARD_DIR, exist_ok=True)
    jobs: list[Job] = []

    def shard(name: str) -> list[str]:
        return ["--out", os.path.join(EVAL_SHARD_DIR, f"{name}.csv")]

    for scenario in SCENARIOS:
        for seed in TRAIN_SEEDS:
            for controller in ("fixed", "actuated"):
                name = f"{controller}_{scenario}_s{seed}"
                jobs.append(Job(
                    name=name, script="eval_marl_corridor.py",
                    args=["--controller", controller, "--seeds", str(seed),
                          "--scenario", scenario, *shard(name)],
                ))
            name = f"marl_independent_{scenario}_s{seed}"
            jobs.append(Job(
                name=name, script="eval_marl_corridor.py",
                args=["--controller", "marl", "--mode", "independent", "--seeds", str(seed),
                      "--scenario", scenario, *shard(name)],
            ))
            name = f"marl_shared_w5_{scenario}_s{seed}"
            jobs.append(Job(
                name=name, script="eval_marl_corridor.py",
                args=["--controller", "marl", "--mode", "shared", "--tag", "w5",
                      "--reward", "shaped", "--seeds", str(seed),
                      "--scenario", scenario, *shard(name)],
            ))

    # Ablations: base demand only.
    ablations = [
        ("w5nofair", "shaped-no-fairness", False),
        ("w5nocoord", "shaped-no-coordination", False),
        ("w5gnn", "shaped", True),
    ]
    for tag, reward, use_gnn in ablations:
        for seed in TRAIN_SEEDS:
            name = f"marl_shared_{tag}_base_s{seed}"
            args = ["--controller", "marl", "--mode", "shared", "--tag", tag,
                    "--reward", reward, "--seeds", str(seed), "--scenario", "base", *shard(name)]
            if use_gnn:
                args.append("--gnn")
            jobs.append(Job(name=name, script="eval_marl_corridor.py", args=args))

    return jobs


def run_jobs(jobs: list[Job], max_parallel: int) -> list[str]:
    """Run jobs with bounded concurrency, streaming each one's output to a log file.

    Args:
        jobs: jobs to run.
        max_parallel: how many may run at once.

    Returns:
        Names of jobs that exited non-zero.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    pending = list(jobs)
    running: list[tuple[Job, subprocess.Popen, object]] = []
    failures: list[str] = []
    total = len(jobs)
    done = 0
    started_at = time.perf_counter()

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            log_path = os.path.join(LOGS_DIR, f"{job.name}.log")
            handle = open(log_path, "w", encoding="utf-8")
            env = {**os.environ, **job.env}
            process = subprocess.Popen(
                [_python(), os.path.join(SRC, job.script), *job.args],
                stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
            )
            running.append((job, process, handle))
            log.info("start  %-34s (%d running, %d queued)", job.name, len(running), len(pending))

        time.sleep(2.0)

        still: list[tuple[Job, subprocess.Popen, object]] = []
        for job, process, handle in running:
            code = process.poll()
            if code is None:
                still.append((job, process, handle))
                continue
            handle.close()
            done += 1
            if code == 0:
                log.info("done   %-34s [%d/%d, %.0f s elapsed]",
                         job.name, done, total, time.perf_counter() - started_at)
            else:
                failures.append(job.name)
                log.error("FAILED %-34s exit=%d — see outputs/logs/%s.log",
                          job.name, code, job.name)
        running = still

    log.info("Stage finished in %.0f s: %d/%d succeeded",
             time.perf_counter() - started_at, total - len(failures), total)
    return failures


def merge_eval_shards() -> None:
    """Merge the per-process evaluation CSVs into the tracked metrics file."""
    from merge_metrics import merge

    merge(os.path.join(EVAL_SHARD_DIR, "*.csv"), MARL_CSV)


def run_online() -> list[str]:
    """Run the Week 5 online-learning loop."""
    return run_jobs(
        [Job(name="w5_online", script="online_learning.py",
             args=["--seed", "0", "--tag", "w5", "--scenario", "asymmetric",
                   "--rounds", "8", "--num-env-runners", "5"])],
        max_parallel=1,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the SmartFlow Week 4-6 experiment matrix.")
    parser.add_argument("--stage", required=True,
                        choices=["week4", "week5", "week6", "evals", "online", "all"])
    parser.add_argument("--timesteps", type=int, default=FULL_TIMESTEPS)
    parser.add_argument("--runners", type=int, default=NUM_ENV_RUNNERS)
    parser.add_argument("--max-parallel", type=int, default=3,
                        help="Concurrent jobs. Keep max_parallel x runners <= core count.")
    parser.add_argument("--clean-shards", action="store_true",
                        help="Delete existing evaluation shards before running the eval stage.")
    args = parser.parse_args()

    stages = ["week4", "week5", "week6", "evals", "online"] if args.stage == "all" else [args.stage]
    all_failures: list[str] = []

    for stage in stages:
        log.info("=== stage: %s ===", stage)
        if stage == "evals":
            if args.clean_shards and os.path.isdir(EVAL_SHARD_DIR):
                shutil.rmtree(EVAL_SHARD_DIR)
            failures = run_jobs(eval_jobs(), max_parallel=max(args.max_parallel, 4))
            merge_eval_shards()
        elif stage == "online":
            failures = run_online()
        else:
            failures = run_jobs(
                training_jobs(stage, args.timesteps, args.runners),
                max_parallel=args.max_parallel,
            )
        all_failures.extend(failures)

    if all_failures:
        log.error("Failed jobs: %s", all_failures)
        raise SystemExit(1)
    log.info("All requested stages completed.")


if __name__ == "__main__":
    main()
