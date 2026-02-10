"""
Baseline ingestion pipeline using ThreadPoolExecutor.

This intentionally demonstrates the limits of threads for CPU-bound
embedding work due to the GIL.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np

from . import embeddings
from .index import InMemoryIndex
from .pipeline_base import PipelineConfig, chunk_document


def _process_document_thread(doc: str, config: PipelineConfig) -> np.ndarray:
    chunks = chunk_document(doc, config.chunk_size, config.chunk_overlap)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings.embed_documents(chunks)


def ingest_documents_threads(
    documents: List[str], config: PipelineConfig
) -> InMemoryIndex:
    """
    Ingest documents using a thread pool.

    Each document is processed independently in a worker thread:
    - Chunk text
    - Generate embeddings per chunk
    - Append embeddings to a shared in-memory index
    """
    index = InMemoryIndex()

    if not documents:
        return index

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(_process_document_thread, doc, config) for doc in documents
        ]

        for fut in as_completed(futures):
            vectors = fut.result()
            if vectors.size:
                index.add(vectors)

    return index


