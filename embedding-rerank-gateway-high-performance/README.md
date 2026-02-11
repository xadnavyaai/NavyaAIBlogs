Embedding + Rerank Gateway Benchmarks
=====================================

This project backs a blog post about **high-performance Embedding + Rerank gateways** for RAG and search — focusing on how a single service can:

- Accept documents/chunks for ingestion
- Store vectors in a simple vector index
- Serve `/v1/search` with retrieval + rerank + citations
- Support asynchronous ingestion jobs
- Expose **measurable performance metrics**:
  - p95 latency for `/v1/search`
  - Throughput (RPS) at a fixed p95 target
  - Approximate RSS memory at different concurrency levels
  - Cold start time

The code here is intentionally **small but realistic**, similar in spirit to `rag-pipelines-waste-cpu-modern-python/`.

Project layout
--------------

- `gateway_bench/` – Python package with:
  - Minimal Embedding + Rerank gateway implementation (FastAPI)
  - In-memory vector index and simple reranker
  - Benchmark runner for load-testing `/v1/search`
- `go-gateway/` – Go implementation of the same API (smaller image, lower footprint); same endpoints so you can run the Python bench against it.
- `rust-gateway/` – Rust + ONNX implementation running **the same model** (all-MiniLM-L6-v2) for apples-to-apples comparison with Python (see `rust-gateway/README.md`).
- `results/` – Raw benchmark results (CSV) for different concurrency levels.
- `blog/` – Blog post content (also deployed on navya.ai).

Quickstart (local)
------------------

1. Create and activate a virtual environment (Python 3.10+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the gateway locally:

```bash
uvicorn gateway_bench.service:app --host 0.0.0.0 --port 8000
```

3. Run a simple benchmark against `/v1/search`:

```bash
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 --concurrency 64 --num-requests 2000 --output results/search_c64_n2000.csv
```

This will:

- Warm up the model
- Fire concurrent `/v1/search` requests
- Record per-request latencies
- Emit summary stats (p50/p95/p99, throughput) and write a CSV of raw timings

Results at a glance
-------------------

Once you have run the benchmarks, you will be able to summarize:

- **p95 latency vs concurrency** (e.g., 16, 64, 256 workers)
- **Throughput (RPS)** at a target p95 (e.g., keep p95 < 150 ms)
- **Approximate memory usage** for the gateway process under each load level
- **Cold start time** from `docker run` to a healthy `/healthz` probe

The blog in `blog/blog.md` shows how to turn these into plots and tables for a production-ready writeup.

Go gateway (optional)
----------------------

Same API as the Python gateway, built for lower memory and smaller image:

```bash
cd go-gateway
docker build -t hprag-gateway-go .
docker run -p 8000:8000 hprag-gateway-go
```

Then run the **same** Python benchmark against it (from project root):

```bash
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 --concurrency 64 --num-requests 2000 --output results/go_search_c64.csv
```

The Go service uses deterministic in-memory “embeddings” (no ML runtime), so results are not comparable to Python’s sentence-transformers; use it to compare **throughput, latency, and RSS** between the two stacks.

Rust gateway (ONNX, same model — apples-to-apples)
---------------------------------------------------

To compare Python vs a native stack on **the same embedding model**:

1. Export the model to ONNX once: `pip install 'optimum[onnx]' && python scripts/export_onnx.py -o rust-gateway/model`
2. Build and run the Rust gateway: see `rust-gateway/README.md`
3. Run the same benchmark: `python -m gateway_bench.bench --base-url http://127.0.0.1:8000 ...`

Then compare Python (sentence-transformers) vs Rust (ONNX) on RPS, p95, and RSS.

Run the split (Architecture C)
------------------------------

To run **embedding service + thin gateway** (two processes, same Rust binary):

1. Export the model and build the Rust image (see rust-gateway/README.md):  
   `cd rust-gateway && docker build -t embedding-gateway-rust .`
2. From the project root:  
   `docker-compose up -d`
3. Benchmark against the gateway (port 8000):  
   `python -m gateway_bench.bench http://localhost:8000 -c 16 -n 2000`

See [docs/CASE_STUDY.md](docs/CASE_STUDY.md) for the full case study and comparison table.

Run benchmarks on GCP (generate numbers)
-----------------------------------------

To run all architectures on a GCP instance and collect RPS, p95, and RSS: see **[docs/RUN_ON_GCP.md](docs/RUN_ON_GCP.md)**. Copy the repo to the VM, install Docker and Python, then run `./scripts/run_benchmarks_gcp.sh`. Results go to `results/GCP_BENCHMARK_RESULTS.md`.

Related
-------

- **RAG ingestion performance (modern Python)** – `rag-pipelines-waste-cpu-modern-python/`
- **NavyaAI engineering blogs** – see the main `README.md` in this repo.

