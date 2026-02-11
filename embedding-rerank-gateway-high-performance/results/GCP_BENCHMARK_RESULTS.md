# GCP benchmark results (run via gcloud ssh)

**Date:** 2026-02-11  
**Bench:** `gateway_bench.bench` — concurrency 16, 2000 requests.

## Single-node summary (true apples-to-apples)

**All four gateways ran on the same GCP VM:** `instance-20260211-183413` (us-central1-a). Same bench (2000 req, c=16). Same session.

| Gateway           | Throughput (RPS) | p50 (ms) | p95 (ms) | p99 (ms) | Container RSS   |
|-------------------|------------------|----------|----------|----------|-----------------|
| **Python (A)**    | **355.9**        | 19.49    | 150.39   | 257.13   | 386.5 MiB       |
| **Go (pseudo)**   | **450.5**        | 14.83    | 113.57   | 198.06   | 6.5 MiB         |
| **Rust (B)**      | **456.1**        | 14.48    | 122.93   | 197.90   | 125.7 MiB       |
| **Split (C)**     | **450.3**        | 14.48    | 116.80   | 211.75   | ~114 + ~114 MiB |

- **Python (A):** CPU-only image, real embeddings (all-MiniLM-L6-v2).
- **Go:** Pseudo-embeddings (no ML); compare on footprint/API only.
- **Rust (B):** Same ONNX model (all-MiniLM-L6-v2), one process.
- **Split (C):** Embed service + thin gateway (same ONNX model).

## Apples-to-apples: what is and what isn’t

| Comparison | Same model? | Same workload? | Same API? | Same VM? |
|------------|-------------|----------------|-----------|----------|
| **Python vs Rust (B) vs Split (C)** | **Yes** — all use all-MiniLM-L6-v2 (Python: sentence-transformers; Rust/Split: ONNX export of same model) | **Yes** — same benchmark: 2000 requests, c=16, same query text; bench only hits `/v1/search` (embed query + search) | **Yes** | **Yes** — all on instance-20260211-183413 |
| **Rust (B) vs Split (C)** | **Yes** — same ONNX model | **Yes** | **Yes** | **Yes** — same VM |
| **Python vs Go** | **No** — Python: real model; Go: pseudo-embeddings (hash-based, no ML) | Same API and request shape | **Yes** | **Yes** |

**Conclusion:**  
- **Apples-to-apples for throughput/latency:** Python, Rust (B), and Split (C) — same embedding model, same workload, **same VM**.  
- **Rust (B) vs Split (C)** are fully apples-to-apples (same VM, same model, same bench).  
- **Go** is **not** apples-to-apples with Python or Rust on RPS/latency; compare Go only on **footprint** (image size, RSS) and API shape.

## Benchmark diff (c=16, 2000 req) — single node

**Baseline: Python (A)** — real embeddings, same API. All numbers from **instance-20260211-183413**. Deltas are *vs Python* unless noted.

| Gateway        | RPS    | Δ RPS   | p50 (ms) | Δ p50  | p95 (ms) | Δ p95   | p99 (ms) | Δ p99   | RSS      | Δ RSS    |
|----------------|--------|---------|----------|--------|----------|---------|----------|---------|----------|----------|
| **Python (A)** | 355.9  | —       | 19.49    | —      | 150.39   | —       | 257.13   | —       | 386.5 MiB | —       |
| **Go**         | 450.5  | +94.6   | 14.83    | −4.66  | 113.57   | −36.82  | 198.06   | −59.07  | 6.5 MiB  | −380    |
| **Rust (B)**   | 456.1  | **+100.2** | 14.48 | **−5.01** | 122.93 | **−27.46** | 197.90 | **−59.23** | 125.7 MiB | **−261** |
| **Split (C)**  | 450.3  | **+94.4**  | 14.48 | **−5.01** | 116.80 | **−33.59** | 211.75 | −45.38  | ~228 MiB | **−159** |

**Apples-to-apples (real embeddings):** Python vs Rust (B) vs Split (C) all use the same model (all-MiniLM-L6-v2) on the **same node**. Rust (B): **+28% RPS**, **−18% p95**, **−67% RSS** vs Python. Split (C): **+27% RPS**, **−22% p95**, **−41% RSS** vs Python.

**Go** is pseudo-embeddings only (no ML); compare to Python on footprint (image/RSS), not raw RPS.

## Improvement summary (Python vs Rust vs Split only)

| Metric | Python | Rust (B) | Split (C) |
|--------|--------|----------|-----------|
| RPS | 356 | **456** (+28%) | **450** (+27%) |
| p95 (ms) | 150 | **123** (−18%) | **117** (−22%) |
| RSS | 387 MiB | **126 MiB** (−67%) | **228 MiB** (−41%) |

**Projected savings (same traffic, same semantics):**

- **Replicas for 500 RPS:** Python 2, Rust 2, Split 2 → total RSS: **774 MiB** vs **252 MiB** (Rust) vs **456 MiB** (Split). Rust **−67%**, Split **−41%** memory.
- **Replicas for 1,000 RPS:** Python 3 (~1.16 GiB), Rust 3 (~378 MiB) → **~68% less memory** with Rust; same API and model.

## Python monolith (A)

- **Image:** `embedding-gateway-python` (built on VM with CPU-only PyTorch).
- **Build:** `sudo docker build -t embedding-gateway-python .` in project root (succeeded from cache after CPU-only Dockerfile sync).

## Go gateway

- **Image:** `hprag-gateway-go` (pre-built or built on VM).
- **Container RSS:** ~6.5 MiB (Docker `stats --no-stream`).

## Rust (B) / Split (C)

**Run on VM `instance-20260211-183413` (us-central1-a)** with larger disk.

- **Rust image:** Dockerfile uses **Ubuntu 24.04** as builder (glibc 2.38+ required by ort/ONNX Runtime). Install Rust via rustup; `cargo build --release` in `rust-gateway/`. Runtime image: `ubuntu:24.04` with ca-certificates + libgcc-s1.
- **B (Rust monolith):** `sudo docker run -d -p 8000:8000 -v /home/vikas/HPRAG/embedding-rerank-gateway-high-performance/rust-gateway/model:/app/model:ro --name hprag-rust embedding-gateway-rust`. **456.1 RPS**, p95 122.93 ms, **~126 MiB** RSS (single-node run).
- **C (Split):** No docker-compose on VM; run manually:
  ```bash
  sudo docker network create hprag-net
  sudo docker run -d --name hprag-embed --network hprag-net -e SERVE_EMBED_ONLY=true -e MODEL_DIR=/app/model -e PORT=8080 -v /home/vikas/HPRAG/embedding-rerank-gateway-high-performance/rust-gateway/model:/app/model:ro embedding-gateway-rust
  sudo docker run -d --name hprag-gateway --network hprag-net -p 8000:8000 -e EMBEDDING_API_URL=http://hprag-embed:8080/embed embedding-gateway-rust
  ```
  **450.3 RPS**, p95 116.80 ms; embed **~114 MiB** + gateway **~114 MiB** (single-node run).

## How to reproduce

From your machine:

```bash
# Copy project to VM
cd /path/to/HPRAG
tar czf - embedding-rerank-gateway-high-performance | gcloud compute ssh INSTANCE --zone=ZONE -- "mkdir -p ~/HPRAG && tar xzf - -C ~/HPRAG"

# On VM: venv for bench client
gcloud compute ssh INSTANCE --zone=ZONE
cd ~/HPRAG/embedding-rerank-gateway-high-performance
rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -r requirements-bench.txt

# Python (A)
sudo docker build -t embedding-gateway-python .
sudo docker run -d -p 8000:8000 --name hprag-python embedding-gateway-python
# wait for ready (curl http://127.0.0.1:8000/healthz)
.venv/bin/python gateway_bench/bench.py --base-url http://127.0.0.1:8000 --concurrency 16 --num-requests 2000
sudo docker stats hprag-python --no-stream
sudo docker stop hprag-python && sudo docker rm hprag-python

# Rust (B) — need rust-gateway image (Ubuntu 24.04 builder, see above)
# cd rust-gateway && sudo docker build -t embedding-gateway-rust .
# sudo docker run -d -p 8000:8000 -e MODEL_DIR=/app/model --name hprag-rust embedding-gateway-rust
# .venv/bin/python gateway_bench/bench.py --base-url http://127.0.0.1:8000 --concurrency 16 --num-requests 2000
# sudo docker stats hprag-rust --no-stream

# Split (C) — two containers on hprag-net (see above)

# Go
sudo docker run -d -p 8000:8000 --name hprag-go hprag-gateway-go
.venv/bin/python gateway_bench/bench.py --base-url http://127.0.0.1:8000 --concurrency 16 --num-requests 2000
sudo docker stats hprag-go --no-stream
sudo docker stop hprag-go && sudo docker rm hprag-go
```

See [docs/RUN_ON_GCP.md](../docs/RUN_ON_GCP.md) for full steps.
