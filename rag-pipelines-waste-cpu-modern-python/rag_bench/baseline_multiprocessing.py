"""
Baseline ingestion pipeline using ProcessPoolExecutor.

This improves CPU utilization for embedding-heavy workloads at the cost
of higher startup and memory overhead per worker process.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np

from .index import InMemoryIndex
from .pipeline_base import PipelineConfig, chunk_document


def _process_document_mp(doc: str, config_dict: dict) -> np.ndarray:
    """
    Worker function executed in a separate process.

    We import heavy modules inside the worker so that each process
    initializes its own model instance.
    """
    from rag_bench import embeddings  # imported in worker process

    cfg = PipelineConfig(**config_dict)
    chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings.embed_documents(chunks)


def ingest_documents_multiprocessing(
    documents: List[str], config: PipelineConfig
) -> InMemoryIndex:
    """
    Ingest documents using a pool of worker processes.
    """
    index = InMemoryIndex()

    if not documents:
        return index

    config_dict = config.as_dict()

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(_process_document_mp, doc, config_dict)
            for doc in documents
        ]

        for fut in as_completed(futures):
            vectors = fut.result()
            if vectors.size:
                index.add(vectors)

    return index


