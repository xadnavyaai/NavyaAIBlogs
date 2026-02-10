"""
Metrics and benchmarking utilities for RAG ingestion pipelines.

This module focuses on:
- Wall-clock time
- Documents per second
- Approximate CPU utilization
- Approximate peak memory usage
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import psutil


@dataclass
class BenchmarkResult:
    scenario: str
    num_docs: int
    chunk_size: int
    num_workers: int
    duration_s: float
    docs_per_s: float
    cpu_mean: float
    cpu_max: float
    mem_peak_mb: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


def _sample_process(
    stop_event: threading.Event, interval_s: float, cpu_samples: List[float], mem_samples: List[float]
) -> None:
    proc = psutil.Process(os.getpid())

    # Prime the CPU measurement to get a useful first value.
    proc.cpu_percent(interval=None)

    while not stop_event.is_set():
        cpu = proc.cpu_percent(interval=interval_s)
        mem = proc.memory_info().rss / (1024 * 1024)  # MB
        cpu_samples.append(cpu)
        mem_samples.append(mem)


def run_benchmark(
    scenario: str,
    ingest_fn: Callable[[], object],
    *,
    num_docs: int,
    chunk_size: int,
    num_workers: int,
    sample_interval_s: float = 0.1,
) -> BenchmarkResult:
    """
    Run a single benchmark scenario and collect metrics.

    ingest_fn is a callable that performs the full ingestion (including
    index construction). Its return value is ignored here.
    """
    stop_event = threading.Event()
    cpu_samples: List[float] = []
    mem_samples: List[float] = []

    sampler = threading.Thread(
        target=_sample_process,
        args=(stop_event, sample_interval_s, cpu_samples, mem_samples),
        daemon=True,
    )

    start = time.perf_counter()
    sampler.start()
    try:
        ingest_fn()
    finally:
        duration_s = time.perf_counter() - start
        stop_event.set()
        sampler.join(timeout=2.0)

    if duration_s <= 0:
        duration_s = 1e-9

    docs_per_s = num_docs / duration_s

    cpu_mean = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    cpu_max = max(cpu_samples) if cpu_samples else 0.0
    mem_peak_mb = max(mem_samples) if mem_samples else 0.0

    return BenchmarkResult(
        scenario=scenario,
        num_docs=num_docs,
        chunk_size=chunk_size,
        num_workers=num_workers,
        duration_s=duration_s,
        docs_per_s=docs_per_s,
        cpu_mean=cpu_mean,
        cpu_max=cpu_max,
        mem_peak_mb=mem_peak_mb,
    )


