# run_week2_full.ps1
# Full Week 2 pipeline: fixed-time baseline, PPO training (3 seeds), PPO eval, chart.
# Run from repo root with venv active:
#   .\venv\Scripts\Activate.ps1
#   .\scripts\run_week2_full.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function LogStep([string]$msg) {
    $ts = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[$ts] $msg" -ForegroundColor Cyan
}

function Die([string]$msg) {
    Write-Host "FATAL: $msg" -ForegroundColor Red
    exit 1
}

Set-Location $Root
LogStep "Working directory: $Root"

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { Die "python not found - activate the venv first." }
LogStep "Python: $($python.Source)"

# Remove stale CSV and old model files to ensure a clean run
$csv = Join-Path $Root "outputs\week2_benchmark_metrics.csv"
if (Test-Path $csv) {
    LogStep "Removing stale $csv"
    Remove-Item $csv
}
$modelsDir = Join-Path $Root "models"
if (Test-Path $modelsDir) {
    LogStep "Removing stale model files from $modelsDir"
    Remove-Item "$modelsDir\ppo_benchmark_seed*.zip" -ErrorAction SilentlyContinue
    Remove-Item "$modelsDir\ppo_benchmark_seed*_hparams.json" -ErrorAction SilentlyContinue
}
$ckptDir = Join-Path $Root "outputs\checkpoints"
if (Test-Path $ckptDir) {
    LogStep "Removing stale checkpoints"
    Remove-Item $ckptDir -Recurse -Force
}

# --- DAY 3: fixed-time baseline (3 seeds) ------------------------------------
LogStep "DAY 3 - Fixed-time baseline seeds 0 1 2"
python src\eval_benchmark.py --controller fixed --seeds 0 1 2
if ($LASTEXITCODE -ne 0) { Die "Day 3 eval failed (exit $LASTEXITCODE)" }
LogStep "DAY 3 complete."

# --- DAY 4: PPO training (3 seeds, chunked to avoid SUMO crash) --------------
LogStep "DAY 4 - Training seed 0 (chunked, 20k steps per chunk)"
python src\train_ppo_benchmark.py --seed 0
if ($LASTEXITCODE -ne 0) { Die "Training seed 0 failed." }
LogStep "Seed 0 done."

LogStep "DAY 4 - Training seed 1"
python src\train_ppo_benchmark.py --seed 1
if ($LASTEXITCODE -ne 0) { Die "Training seed 1 failed." }
LogStep "Seed 1 done."

LogStep "DAY 4 - Training seed 2"
python src\train_ppo_benchmark.py --seed 2
if ($LASTEXITCODE -ne 0) { Die "Training seed 2 failed." }
LogStep "Seed 2 done."

LogStep "DAY 4 complete - models saved to models/"

# --- DAY 5: PPO evaluation (3 seeds) -----------------------------------------
LogStep "DAY 5 - PPO evaluation seeds 0 1 2"
python src\eval_benchmark.py --controller ppo --seeds 0 1 2
if ($LASTEXITCODE -ne 0) { Die "Day 5 PPO eval failed." }
LogStep "DAY 5 complete."

# --- DAY 6: comparison chart --------------------------------------------------
LogStep "DAY 6 - Generating comparison chart"
python src\compare_week2.py
if ($LASTEXITCODE -ne 0) { Die "Day 6 chart failed." }
LogStep "DAY 6 complete."

# --- print result summary -----------------------------------------------------
LogStep "PIPELINE COMPLETE - result summary:"
python -c @"
import csv, sys
sys.path.insert(0, 'src')
from week2_config import BENCHMARK_CSV
rows = list(csv.DictReader(open(BENCHMARK_CSV)))
from collections import defaultdict
groups = defaultdict(list)
for r in rows:
    groups[r['controller']].append(float(r['avg_wait_time_s']))
print()
print('=== WEEK 2 RESULT SUMMARY ===')
for ctrl in ['fixed','ppo']:
    vals = groups[ctrl]
    if vals:
        avg = sum(vals)/len(vals)
        print(f'{ctrl:8s}  avg_wait={avg:.2f}s  n={len(vals)}')
if groups['fixed'] and groups['ppo']:
    f = sum(groups['fixed'])/len(groups['fixed'])
    p = sum(groups['ppo'])/len(groups['ppo'])
    pct = (p-f)/f*100
    verdict = 'PASS' if p < f else 'FAIL - diagnose before proceeding'
    print(f'PPO vs fixed: {pct:+.1f}%  [{verdict}]')
print('=============================')
"@
