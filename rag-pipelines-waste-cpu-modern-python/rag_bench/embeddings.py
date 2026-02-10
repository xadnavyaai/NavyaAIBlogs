"""
Embedding utilities built on top of a real sentence-transformers model.

The default model is chosen to be small and CPU-friendly so it runs comfortably
inside a Docker container while still behaving like a realistic embedding workload.
"""

from __future__ import annotations

import os
import threading
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_NAME_ENV = "RAG_BENCH_MODEL_NAME"
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model_name() -> str:
    return os.environ.get(_MODEL_NAME_ENV, _DEFAULT_MODEL_NAME)


def get_model() -> SentenceTransformer:
    """
    Lazily load and cache the sentence-transformers model.

    Thread-safe so it can be shared across worker threads in the process.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(_get_model_name())
    return _model  # type: ignore[return-value]


def embed_documents(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of documents into a dense vector space.

    Returns a NumPy array of shape (len(texts), dim). If `texts` is empty, this
    returns a (0, dim) array once the model is known.
    """
    if not texts:
        # We still want a well-shaped array; query the model dimension.
        model = get_model()
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Ensure dtype is consistent.
    return np.asarray(embeddings, dtype=np.float32)


