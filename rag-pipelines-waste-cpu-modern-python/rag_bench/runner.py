"""
CLI entrypoint for running RAG ingestion benchmarks.

Examples:

    python -m rag_bench.runner --scenario baseline-threads --num-docs 1000
    python -m rag_bench.runner --all-scenarios --num-docs 5000
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable, List

from . import data_generation
from . import metrics
from .baseline_multiprocessing import ingest_documents_multiprocessing
from .baseline_threads import ingest_documents_threads
from .modern_interpreters import ingest_documents_modern
from .pipeline_base import PipelineConfig


SCENARIO_BASELINE_THREADS = "baseline-threads"
SCENARIO_BASELINE_MP = "baseline-mp"
SCENARIO_MODERN_INTERPRETERS = "modern-interpreters"

ALL_SCENARIOS = [
    SCENARIO_BASELINE_THREADS,
    SCENARIO_BASELINE_MP,
    SCENARIO_MODERN_INTERPRETERS,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG ingestion benchmarks.")
    parser.add_argument(
        "--scenario",
        choices=ALL_SCENARIOS,
        help="Single scenario to run.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all scenarios.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=1000,
        help="Number of synthetic documents to generate.",
    )
    parser.add_argument(
        "--avg-paragraphs",
        type=int,
        default=4,
        help="Average paragraphs per document.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Approximate characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Characters of overlap between chunks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override number of workers (defaults to CPU count).",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/results.csv",
        help="Path to write CSV results.",
    )
    return parser.parse_args()


def _ensure_results_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


def _select_scenarios(args: argparse.Namespace) -> List[str]:
    if args.all_scenarios:
        return ALL_SCENARIOS
    if args.scenario:
        return [args.scenario]
    # Default: run all scenarios if none specified explicitly.
    return ALL_SCENARIOS


def _run_single_scenario(
    scenario: str,
    documents: List[str],
    config: PipelineConfig,
) -> metrics.BenchmarkResult:
    if scenario == SCENARIO_BASELINE_THREADS:
        ingest_fn = lambda: ingest_documents_threads(documents, config)
    elif scenario == SCENARIO_BASELINE_MP:
        ingest_fn = lambda: ingest_documents_multiprocessing(documents, config)
    elif scenario == SCENARIO_MODERN_INTERPRETERS:
        ingest_fn = lambda: ingest_documents_modern(documents, config)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return metrics.run_benchmark(
        scenario=scenario,
        ingest_fn=ingest_fn,
        num_docs=len(documents),
        chunk_size=config.chunk_size,
        num_workers=config.num_workers,
    )


def _write_results(path: str, results: Iterable[metrics.BenchmarkResult]) -> None:
    _ensure_results_dir(path)
    results = list(results)
    if not results:
        return

    fieldnames = list(results[0].to_dict().keys())
    # Append to existing file if present, otherwise write header.
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def main() -> None:
    args = _parse_args()

    cfg = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if args.num_workers > 0:
        cfg.num_workers = args.num_workers

    doc_cfg = data_generation.DocumentGenerationConfig(
        num_documents=args.num_docs,
        avg_paragraphs_per_doc=args.avg_paragraphs,
    )
    documents = data_generation.generate_documents(doc_cfg)

    scenarios = _select_scenarios(args)
    results: List[metrics.BenchmarkResult] = []

    for scenario in scenarios:
        result = _run_single_scenario(scenario, documents, cfg)
        results.append(result)

    _write_results(args.results_path, results)


if __name__ == "__main__":
    main()

