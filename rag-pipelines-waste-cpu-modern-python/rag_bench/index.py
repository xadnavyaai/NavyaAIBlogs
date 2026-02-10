"""
Lightweight in-memory vector index used for ingestion benchmarks.

The goal here is to model the cost of building/updating an index rather
than to provide a full-featured ANN implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InMemoryIndex:
    """Simple append-only index storing embeddings in-memory."""

    dim: Optional[int] = None
    _embeddings: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def add(self, vectors: np.ndarray) -> None:
        """Append a batch of vectors to the index."""
        if vectors.size == 0:
            return

        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        vectors = np.asarray(vectors, dtype=np.float32)

        if self._embeddings is None:
            self.dim = vectors.shape[1]
            self._embeddings = vectors
        else:
            if self.dim is None:
                self.dim = vectors.shape[1]
            if vectors.shape[1] != self.dim:
                raise ValueError(
                    f"Dimension mismatch: existing dim={self.dim}, new dim={vectors.shape[1]}"
                )
            self._embeddings = np.vstack([self._embeddings, vectors])

    @property
    def size(self) -> int:
        """Number of vectors stored."""
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    @property
    def embeddings(self) -> np.ndarray:
        """Return the underlying embeddings array (read-only use)."""
        if self._embeddings is None:
            if self.dim is None:
                return np.zeros((0, 0), dtype=np.float32)
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embeddings


