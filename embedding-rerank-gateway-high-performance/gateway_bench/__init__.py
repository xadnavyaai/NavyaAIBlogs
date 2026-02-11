"""
gateway_bench
==============

Minimal Embedding + Rerank gateway implementation plus benchmarking helpers.

This is intentionally small and focused:

- FastAPI app with:
  - POST /v1/ingest
  - POST /v1/search
  - GET /v1/jobs/{id}
  - GET /healthz
- In-memory vector index with cosine similarity
- Simple reranker using the same embedding model
- Bench harness for measuring latency and throughput
"""

