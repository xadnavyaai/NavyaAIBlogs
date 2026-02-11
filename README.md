# NavyaAIBlogs

Reproducible opensource codes and cookbooks.

## Projects

- **RAG ingestion performance (modern Python)** – [`rag-pipelines-waste-cpu-modern-python/`](./rag-pipelines-waste-cpu-modern-python/)  
  Benchmarking threads vs multiprocessing vs modern multi-interpreter execution for RAG ingestion, with:
  - Synthetic document generation
  - Real `sentence-transformers` embedding model
  - Metrics (docs/sec, CPU, memory)
  - Dockerized runner and plotting scripts for crisp, plottable outcomes

- **Embedding + Rerank gateway performance** – [`embedding-rerank-gateway-high-performance/`](./embedding-rerank-gateway-high-performance/)  
  Same gateway in Python (FastAPI + sentence-transformers), Go (thin layer), and Rust + ONNX (same model). Benchmark harness for p95, RPS, and RSS; single-node GCP comparison and reproducible steps. Run `scripts/export_onnx.py` for the Rust/ONNX model.

