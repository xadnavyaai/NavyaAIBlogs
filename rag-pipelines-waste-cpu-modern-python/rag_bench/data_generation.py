"""
Utilities for generating synthetic-but-realistic documents for RAG ingestion.

We avoid any external datasets and instead generate technical-style text with
configurable document counts and approximate lengths.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List


_BASE_PARAGRAPHS: List[str] = [
    (
        "Retrieval-Augmented Generation (RAG) pipelines typically ingest large volumes "
        "of heterogeneous documents. Each document is chunked, embedded into a vector "
        "space, and stored in an index for low-latency retrieval at query time."
    ),
    (
        "Modern Python runtimes introduce support for multiple interpreters with "
        "per-interpreter global interpreter locks. This enables true CPU-bound "
        "parallelism within a single process when tasks are independent."
    ),
    (
        "Naively scaling RAG systems often means adding more containers, pods, or "
        "instances. This increases cost and operational complexity without necessarily "
        "improving per-core utilization."
    ),
    (
        "Efficient ingestion focuses on maximizing throughput per machine by saturating "
        "available CPU cores, minimizing orchestration overhead, and avoiding "
        "unnecessary inter-process communication."
    ),
]


@dataclass
class DocumentGenerationConfig:
    """Configuration for synthetic document generation."""

    num_documents: int = 1_000
    avg_paragraphs_per_doc: int = 4
    std_paragraphs_per_doc: int = 1
    seed: int | None = 42


def _random_paragraphs(
    rng: random.Random,
    avg_paragraphs: int,
    std_paragraphs: int,
) -> List[str]:
    count = max(1, int(rng.normalvariate(avg_paragraphs, std_paragraphs)))
    return [rng.choice(_BASE_PARAGRAPHS) for _ in range(count)]


def generate_documents(config: DocumentGenerationConfig) -> List[str]:
    """
    Generate a list of synthetic documents.

    Each document is formed by sampling a small number of base paragraphs. This keeps
    content realistic enough for embedding workloads while remaining deterministic and
    fast to generate.
    """
    rng = random.Random(config.seed)
    docs: List[str] = []
    for _ in range(config.num_documents):
        paragraphs = _random_paragraphs(
            rng,
            config.avg_paragraphs_per_doc,
            config.std_paragraphs_per_doc,
        )
        docs.append("\n\n".join(paragraphs))
    return docs


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield items in fixed-size batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


