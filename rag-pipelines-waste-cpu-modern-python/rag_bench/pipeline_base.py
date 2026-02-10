"""
Common pieces for RAG ingestion pipelines.

This module provides:
- A shared PipelineConfig.
- Simple text chunking utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class PipelineConfig:
    """Configuration shared across ingestion pipelines."""

    num_workers: int = max(1, (getattr(os, "process_cpu_count", os.cpu_count)() or 1))
    chunk_size: int = 1000  # approximate characters per chunk
    chunk_overlap: int = 200  # characters of overlap between chunks
    batch_size: int = 16  # used for batching inside some pipelines

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Naive character-based chunking with overlap.

    This approximates token-based chunking well enough for performance
    comparisons without pulling in extra dependencies.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - chunk_overlap
    return chunks


def chunk_documents(documents: List[str], config: PipelineConfig) -> List[str]:
    """Apply `chunk_document` to all documents."""
    all_chunks: List[str] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, config.chunk_size, config.chunk_overlap)
        )
    return all_chunks


