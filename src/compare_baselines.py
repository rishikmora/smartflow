"""
Reads outputs/metrics.csv, averages by controller, and saves a 3-panel
bar chart to outputs/baseline_comparison.png.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "outputs", "metrics.csv")
OUT  = os.path.join(ROOT, "outputs", "baseline_comparison.png")

df   = pd.read_csv(CSV)
mean = df.groupby("controller")[["avg_wait_time_s", "max_queue_len", "throughput_veh"]].mean()
std  = df.groupby("controller")[["avg_wait_time_s", "max_queue_len", "throughput_veh"]].std()

controllers = ["fixed", "actuated"]
colors      = ["#4C72B0", "#DD8452"]

panels = [
    ("avg_wait_time_s", "Avg Wait Time (s)",    "lower is better"),
    ("max_queue_len",   "Max Queue Length (veh)", "lower is better"),
    ("throughput_veh",  "Throughput (veh/1800s)", "higher is better"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 5))
fig.suptitle("SmartFlow Week 1 Baselines — 4x4 Grid, 1800 vehicles, 3 seeds",
             fontsize=12, fontweight="bold", y=1.01)

for ax, (col, ylabel, note) in zip(axes, panels):
    vals = [mean.loc[c, col] for c in controllers]
    errs = [std.loc[c, col]  for c in controllers]
    bars = ax.bar(controllers, vals, yerr=errs, capsize=5,
                  color=colors, width=0.5, error_kw={"elinewidth": 1.5})
    ax.set_title(f"{ylabel}\n({note})", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis="x", labelsize=9)
    # annotate bar tops with mean value
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errs) * 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Chart saved -> {OUT}")
print("\n3-seed averages:")
print(mean.round(2).to_string())
