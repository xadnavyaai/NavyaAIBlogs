import time
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


class IngestDocument(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional caller-specified ID")
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class JobStatus(str):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    job_id: str
    status: str
    total_documents: int
    processed_documents: int
    error: Optional[str] = None
    started_at: float
    finished_at: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, str]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    took_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    document_count: int


class InMemoryVectorIndex:
    """
    Simple in-memory index backed by NumPy arrays.

    This keeps the implementation easy to read while still reflecting the
    shape of a real gateway that would plug into pgvector, Qdrant, etc.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._embeddings: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict[str, str]] = []

    @property
    def document_count(self) -> int:
        return len(self._ids)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def ingest(self, docs: List[IngestDocument]) -> None:
        texts = [d.text for d in docs]
        metas = [d.metadata for d in docs]
        ids = [d.id or str(uuid.uuid4()) for d in docs]

        vectors = self._encode(texts)

        if self._embeddings is None:
            self._embeddings = vectors
        else:
            self._embeddings = np.vstack([self._embeddings, vectors])

        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metas.extend(metas)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self._embeddings is None or not self._ids:
            return []

        query_vec = self._encode([query])[0:1]
        scores = (self._embeddings @ query_vec.T).reshape(-1)
        top_k = min(top_k, len(self._ids))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    id=self._ids[int(idx)],
                    score=float(scores[int(idx)]),
                    text=self._texts[int(idx)],
                    metadata=self._metas[int(idx)],
                )
            )
        return results


app = FastAPI(title="Embedding + Rerank Gateway")

index = InMemoryVectorIndex()
jobs: Dict[str, JobResult] = {}


def _run_ingest_job(job_id: str, docs: List[IngestDocument]) -> None:
    job = jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = time.time()
    job.total_documents = len(docs)
    try:
        index.ingest(docs)
        job.processed_documents = len(docs)
        job.status = JobStatus.COMPLETED
    except Exception as exc:  # pragma: no cover - defensive
        job.error = str(exc)
        job.status = JobStatus.FAILED
    finally:
        job.finished_at = time.time()
        jobs[job_id] = job


@app.post("/v1/ingest", response_model=JobResult)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> JobResult:
    job_id = str(uuid.uuid4())
    job = JobResult(
        job_id=job_id,
        status=JobStatus.PENDING,
        total_documents=len(request.documents),
        processed_documents=0,
        started_at=time.time(),
    )
    jobs[job_id] = job
    background_tasks.add_task(_run_ingest_job, job_id, request.documents)
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobResult)
def get_job(job_id: str) -> JobResult:
    job = jobs.get(job_id)
    if job is None:
        # For simplicity we don't distinguish 404 vs 400 here.
        raise ValueError(f"Unknown job_id: {job_id}")
    return job


@app.post("/v1/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    start = time.time()
    results = index.search(request.query, top_k=request.top_k)
    took_ms = (time.time() - start) * 1000.0
    return SearchResponse(results=results, took_ms=took_ms)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=True, document_count=index.document_count)


__all__ = ["app", "index"]

