"""
Ingestion pipeline using modern Python multi-interpreter execution.

This uses concurrent.futures.InterpreterPoolExecutor when available
(Python 3.14+). On older Python versions it gracefully falls back to a
ProcessPoolExecutor while keeping the same API.

When using the real InterpreterPoolExecutor, all function arguments and
return values must be "shareable" types (str, int, float, bytes, tuple,
etc.).  We therefore serialise config as individual int args and return
embedding vectors as raw bytes with a struct header for the shape.
"""

from __future__ import annotations

import struct
from typing import List

import numpy as np

from .index import InMemoryIndex
from .pipeline_base import PipelineConfig, chunk_document

try:  # Python 3.14+
    from concurrent.futures import InterpreterPoolExecutor  # type: ignore[attr-defined]

    HAS_INTERPRETER_POOL = True
except ImportError:  # Older Python: fall back to processes
    HAS_INTERPRETER_POOL = False


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _process_doc_interp(doc: str, chunk_size: int, chunk_overlap: int) -> bytes:
    """Worker for InterpreterPoolExecutor (shareable args and return)."""
    import struct as _struct

    import numpy as _np

    from rag_bench import embeddings as _emb
    from rag_bench.pipeline_base import chunk_document as _chunk

    chunks = _chunk(doc, chunk_size, chunk_overlap)
    if not chunks:
        return b""
    vectors = _emb.embed_documents(chunks).astype(_np.float32)
    # Header: two int64s for (rows, cols), followed by raw float32 data.
    header = _struct.pack("!qq", vectors.shape[0], vectors.shape[1])
    return header + vectors.tobytes()


def _process_doc_process(doc: str, config_dict: dict) -> np.ndarray:
    """Worker for ProcessPoolExecutor (pickle-based, no shareability constraint)."""
    from rag_bench import embeddings
    from rag_bench.pipeline_base import PipelineConfig, chunk_document

    cfg = PipelineConfig(**config_dict)
    chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings.embed_documents(chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    if HAS_INTERPRETER_POOL:
        # Use the real InterpreterPoolExecutor with shareable types.
        with InterpreterPoolExecutor(max_workers=config.num_workers) as executor:
            futures = [
                executor.submit(
                    _process_doc_interp,
                    doc,
                    config.chunk_size,
                    config.chunk_overlap,
                )
                for doc in documents
            ]

            for fut in futures:
                raw = fut.result()
                if raw:
                    rows, cols = struct.unpack("!qq", raw[:16])
                    vectors = np.frombuffer(raw[16:], dtype=np.float32).reshape(
                        rows, cols
                    )
                    index.add(vectors)
    else:
        # Fallback: ProcessPoolExecutor (older Python).
        from concurrent.futures import ProcessPoolExecutor

        config_dict = config.as_dict()
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = [
                executor.submit(_process_doc_process, doc, config_dict)
                for doc in documents
            ]

            for fut in futures:
                vectors = fut.result()
                if vectors.size:
                    index.add(vectors)

    return index


