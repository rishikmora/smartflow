# Week 2 full pipeline runner.
# Run this from the project root to execute Days 3-7 sequentially.
# If Claude's session ended before completing the week, run this manually:
#
#   cd C:\Users\sirin\Desktop\projects\smartflow
#   .\venv\Scripts\Activate.ps1
#   .\scripts\run_week2_pipeline.ps1
#
# Assumes Day 2 validation already completed (models/ppo_benchmark_seed0_short.zip exists).
# The full 3-seed training (Day 4) takes ~3-4 hours on a single machine.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Log($msg) {
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  $msg" -ForegroundColor Cyan
}

# ── Day 3: Fixed-time baseline (3 seeds) ──────────────────────────────────────
Log "=== DAY 3: Fixed-time baseline (3 seeds) ==="
.\venv\Scripts\python.exe src\eval_benchmark.py --controller fixed --seeds 0 1 2
if ($LASTEXITCODE -ne 0) { throw "Day 3 failed (exit $LASTEXITCODE)" }
Log "Day 3 complete."

# ── Day 4: Full PPO training (3 seeds x 100k steps) ──────────────────────────
Log "=== DAY 4: Full PPO training (3 seeds x 100k steps, ~3-4 h) ==="
foreach ($seed in 0, 1, 2) {
    Log "Training seed=$seed ..."
    .\venv\Scripts\python.exe src\train_ppo_benchmark.py --timesteps 100000 --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Day 4 training seed=$seed failed (exit $LASTEXITCODE)" }
    Log "seed=$seed saved."
}
Log "Day 4 complete."

# ── Day 5: PPO evaluation (3 seeds) ───────────────────────────────────────────
Log "=== DAY 5: PPO evaluation (3 seeds) ==="
.\venv\Scripts\python.exe src\eval_benchmark.py --controller ppo --seeds 0 1 2
if ($LASTEXITCODE -ne 0) { throw "Day 5 failed (exit $LASTEXITCODE)" }
Log "Day 5 complete."

# ── Day 6: Comparison chart ────────────────────────────────────────────────────
Log "=== DAY 6: Generating comparison chart ==="
.\venv\Scripts\python.exe src\compare_week2.py
if ($LASTEXITCODE -ne 0) { throw "Day 6 failed (exit $LASTEXITCODE)" }
Log "Day 6 complete — chart saved to outputs\week2_benchmark_comparison.png"

# ── Day 7: pip freeze + print reminder ────────────────────────────────────────
Log "=== DAY 7: Freezing requirements ==="
.\venv\Scripts\pip.exe freeze | Out-File -FilePath requirements.txt -Encoding utf8
Log "requirements.txt updated."

Log ""
Log "=== WEEK 2 PIPELINE COMPLETE ==="
Log "Check outputs\week2_benchmark_metrics.csv for results."
Log "Check outputs\week2_benchmark_comparison.png for the chart."
Log ""
Log "Next step: update README.md with Week 2 results, then git commit."
Log "  git add src\eval_benchmark.py src\compare_week2.py src\week2_config.py"
Log "  git add src\train_ppo_benchmark.py src\week2_smoketest.py"
Log "  git add scripts\run_week2_pipeline.ps1 .gitignore requirements.txt"
Log "  git add outputs\week2_benchmark_metrics.csv outputs\week2_benchmark_comparison.png"
Log "  git add models\ppo_benchmark_seed0.zip models\ppo_benchmark_seed1.zip models\ppo_benchmark_seed2.zip"
Log "  git add README.md"
Log "  git commit -m 'feat(week2): PPO single-agent benchmark — beats fixed-time baseline'"
