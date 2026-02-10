"""
Ingestion pipeline using modern Python multi-interpreter execution.

This uses concurrent.futures.InterpreterPoolExecutor when available
(Python 3.14+). On older Python versions it gracefully falls back to a
ProcessPoolExecutor while keeping the same API.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .index import InMemoryIndex
from .pipeline_base import PipelineConfig, chunk_document

try:  # Python 3.14+
    from concurrent.futures import InterpreterPoolExecutor as _PoolExecutor  # type: ignore[attr-defined]

    HAS_INTERPRETER_POOL = True
except ImportError:  # Older Python: fall back to processes
    from concurrent.futures import ProcessPoolExecutor as _PoolExecutor

    HAS_INTERPRETER_POOL = False


def _process_document_interpreter(doc: str, config_dict: dict) -> np.ndarray:
    """
    Worker function executed in an isolated interpreter (or process).

    Imports are done inside the function so that each interpreter has
    its own runtime state and model instance.
    """
    from rag_bench import embeddings  # imported per interpreter
    from rag_bench.pipeline_base import PipelineConfig, chunk_document

    cfg = PipelineConfig(**config_dict)
    chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings.embed_documents(chunks)


def ingest_documents_modern(
    documents: List[str], config: PipelineConfig
) -> InMemoryIndex:
    """
    Ingest documents using a pool of isolated interpreters.

    When running on a Python version with InterpreterPoolExecutor,
    each worker has its own interpreter and GIL, enabling true
    multi-core parallelism for CPU-bound embedding work.
    """
    index = InMemoryIndex()

    if not documents:
        return index

    config_dict = config.as_dict()

    with _PoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(_process_document_interpreter, doc, config_dict)
            for doc in documents
        ]

        for fut in futures:
            vectors = fut.result()
            if vectors.size:
                index.add(vectors)

    return index


