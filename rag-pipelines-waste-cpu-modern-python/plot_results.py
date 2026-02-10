"""
Plot benchmark results produced by rag_bench.runner.

Generates simple, publication-ready charts:
- documents_per_second_by_scenario.png
- cpu_mean_by_scenario.png
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RAG ingestion benchmark results.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/results.csv",
        help="Input CSV file with benchmark results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to write plot images.",
    )
    return parser.parse_args()


def _load_results(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _group_by_scenario(rows: List[Dict[str, str]]):
    by_scenario: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    return by_scenario


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_documents_per_second(by_scenario, output_dir: str) -> None:
    scenarios: List[str] = sorted(by_scenario.keys())
    docs_per_s_means: List[float] = []

    for scenario in scenarios:
        rows = by_scenario[scenario]
        docs_vals = [float(r["docs_per_s"]) for r in rows]
        docs_per_s_means.append(_mean(docs_vals))

    plt.figure(figsize=(6, 4))
    plt.bar(scenarios, docs_per_s_means)
    plt.ylabel("Documents / second (mean)")
    plt.xlabel("Scenario")
    plt.title("RAG Ingestion Throughput by Scenario")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "documents_per_second_by_scenario.png")
    plt.savefig(out_path)
    plt.close()


def plot_cpu_mean(by_scenario, output_dir: str) -> None:
    scenarios: List[str] = sorted(by_scenario.keys())
    cpu_means: List[float] = []

    for scenario in scenarios:
        rows = by_scenario[scenario]
        cpu_vals = [float(r["cpu_mean"]) for r in rows]
        cpu_means.append(_mean(cpu_vals))

    plt.figure(figsize=(6, 4))
    plt.bar(scenarios, cpu_means)
    plt.ylabel("CPU utilization (%) (mean)")
    plt.xlabel("Scenario")
    plt.title("CPU Utilization by Scenario")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "cpu_mean_by_scenario.png")
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = _parse_args()
    _ensure_output_dir(args.output_dir)
    rows = _load_results(args.input)
    by_scenario = _group_by_scenario(rows)

    plot_documents_per_second(by_scenario, args.output_dir)
    plot_cpu_mean(by_scenario, args.output_dir)


if __name__ == "__main__":
    main()

