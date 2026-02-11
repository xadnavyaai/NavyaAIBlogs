#!/usr/bin/env python3
"""Generate publication-quality comparison plots for Python 3.13 vs 3.14 vs 3.14t benchmarks."""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def compute_means(rows):
    """Group by scenario, compute means across document scales."""
    groups = defaultdict(lambda: {"docs_per_s": [], "cpu_mean": [], "mem_peak_mb": []})
    for r in rows:
        s = r["scenario"]
        groups[s]["docs_per_s"].append(float(r["docs_per_s"]))
        groups[s]["cpu_mean"].append(float(r["cpu_mean"]))
        groups[s]["mem_peak_mb"].append(float(r["mem_peak_mb"]))

    means = {}
    for s, v in groups.items():
        means[s] = {
            "docs_per_s": sum(v["docs_per_s"]) / len(v["docs_per_s"]),
            "cpu_mean": sum(v["cpu_mean"]) / len(v["cpu_mean"]),
            "mem_peak_mb": sum(v["mem_peak_mb"]) / len(v["mem_peak_mb"]),
        }
    return means


# Ordered scenario labels for plotting
SCENARIOS = [
    "py313-threads",
    "py314-threads",
    "py314t-threads",
    "py313-multiprocessing",
    "py314-multiprocessing",
    "py314t-multiprocessing",
]

LABELS = [
    "3.13\nThreads",
    "3.14\nThreads",
    "3.14t (no-GIL)\nThreads",
    "3.13\nMultiprocessing",
    "3.14\nMultiprocessing",
    "3.14t (no-GIL)\nMultiprocessing",
]

COLORS_THREADS = ["#2196F3", "#1565C0", "#0D47A1"]
COLORS_MP = ["#FF9800", "#E65100", "#BF360C"]
COLORS = COLORS_THREADS + COLORS_MP


def plot_throughput(means, output_dir):
    """Bar chart: mean docs/s by scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))

    vals = [means.get(s, {}).get("docs_per_s", 0) for s in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    bars = ax.bar(x, vals, color=COLORS, edgecolor="white", linewidth=0.5, width=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Documents / Second (mean)", fontsize=12)
    ax.set_title(
        "RAG Ingestion Throughput: Python 3.13 vs 3.14 vs 3.14t (no-GIL)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add separator line between threads and MP groups
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "throughput_314_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {output_dir}/throughput_314_comparison.png")


def plot_cpu(means, output_dir):
    """Bar chart: mean CPU % by scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))

    vals = [means.get(s, {}).get("cpu_mean", 0) for s in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    bars = ax.bar(x, vals, color=COLORS, edgecolor="white", linewidth=0.5, width=0.7)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Mean CPU Utilization (%)", fontsize=12)
    ax.set_title(
        "CPU Utilization: Python 3.13 vs 3.14 vs 3.14t (no-GIL)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cpu_314_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {output_dir}/cpu_314_comparison.png")


def plot_memory(means, output_dir):
    """Bar chart: peak memory by scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))

    vals = [means.get(s, {}).get("mem_peak_mb", 0) for s in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    bars = ax.bar(x, vals, color=COLORS, edgecolor="white", linewidth=0.5, width=0.7)

    for bar, val in zip(bars, vals):
        label = f"{val / 1000:.1f} GB" if val >= 1000 else f"{val:.0f} MB"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title(
        "Peak Memory Usage: Python 3.13 vs 3.14 vs 3.14t (no-GIL)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "memory_314_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {output_dir}/memory_314_comparison.png")


def plot_efficiency(means, output_dir):
    """Scatter: throughput vs CPU — shows efficiency (docs/s per CPU%)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, s in enumerate(SCENARIOS):
        m = means.get(s, {})
        docs = m.get("docs_per_s", 0)
        cpu = m.get("cpu_mean", 0)
        mem = m.get("mem_peak_mb", 0)
        ax.scatter(
            cpu,
            docs,
            s=mem / 8,  # size proportional to memory
            c=COLORS[i],
            edgecolors="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=5,
        )
        ax.annotate(
            LABELS[i].replace("\n", " "),
            (cpu, docs),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_xlabel("Mean CPU Utilization (%)", fontsize=12)
    ax.set_ylabel("Documents / Second", fontsize=12)
    ax.set_title(
        "Efficiency Map: Throughput vs CPU (bubble size = memory)",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "efficiency_314_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {output_dir}/efficiency_314_comparison.png")


def main():
    csv_path = "results/results_combined.csv"
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    rows = load_csv(csv_path)
    means = compute_means(rows)

    print("Generating comparison plots...")
    plot_throughput(means, output_dir)
    plot_cpu(means, output_dir)
    plot_memory(means, output_dir)
    plot_efficiency(means, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
