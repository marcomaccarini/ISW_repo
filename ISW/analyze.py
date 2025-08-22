#!/usr/bin/env python3
"""
Analysis of multi-family LLM evaluation results.
- Loads all *_zero.csv and *_few.csv result files.
- Aggregates per-model metrics.
- Produces useful charts in PNG format inside ./analysis_charts.

Run:
  python analyze_results.py
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
RESULTS_DIR = Path("/home/mmaccarini/Blerina/ISW_repo/results_multimodel")
OUT_DIR = Path("/home/mmaccarini/Blerina/ISW_repo/analysis_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics we want to plot
METRICS = ["exact_match", "edit_sim", "jaccard_sim", "rouge_l", "bleu", "bert_cosine"]


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def parse_filename(fname: str):
    """
    Expected naming: family_size_mode.csv
    Example: llama_small_few.csv
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    family, size, mode = parts[0], parts[1], parts[2]
    return family, size, mode


# ────────────────────────────────────────────────
# Load all results
# ────────────────────────────────────────────────
all_rows = []
for csvf in RESULTS_DIR.glob("*.csv"):
    parsed = parse_filename(csvf.name)
    if not parsed:
        continue
    family, size, mode = parsed
    df = pd.read_csv(csvf)
    # compute average metrics
    avg = {m: df[m].mean() for m in METRICS if m in df}
    avg["family"] = family
    avg["size"] = size
    avg["mode"] = mode
    all_rows.append(avg)

summary = pd.DataFrame(all_rows)
summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
print("Saved summary →", OUT_DIR / "summary_metrics.csv")


# ────────────────────────────────────────────────
# Charts
# ────────────────────────────────────────────────

def barplot_metric(metric):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="size", y=metric,
        hue="family",
        ci="sd"
    )
    plt.title(f"Average {metric} by size and family")
    plt.ylabel(metric)
    plt.xlabel("Model size tier")
    plt.ylim(0, 1)
    plt.legend(title="Family")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"bar_{metric}.png")
    plt.close()


def barplot_mode(metric):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="mode", y=metric,
        hue="family",
        ci="sd"
    )
    plt.title(f"Effect of prompt mode on {metric}")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"mode_{metric}.png")
    plt.close()


# produce barplots per metric
for m in METRICS:
    if m in summary:
        barplot_metric(m)
        barplot_mode(m)

# Scatter of BERT cosine vs BLEU
plt.figure(figsize=(7, 6))
sns.scatterplot(data=summary, x="bleu", y="bert_cosine", hue="family", style="size", s=120)
plt.title("BLEU vs BERT cosine (avg per model)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_bleu_vs_bert.png")
plt.close()

print("Charts saved in", OUT_DIR)
