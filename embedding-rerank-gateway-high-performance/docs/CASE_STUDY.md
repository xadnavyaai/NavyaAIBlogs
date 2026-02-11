# Production case study: beat Python (even when split)

One-page checklist to compare three production architectures and reproduce the benchmarks. Same client API; same benchmark harness; different deployment shapes.

## Three architectures

| Architecture | What it is | When to use |
|--------------|------------|-------------|
| **A. Python monolith** | One Python service (FastAPI + sentence-transformers): embed + index + search. | Baseline; teams already on Python; single service OK. |
| **B. Native monolith** | One Rust service (ONNX): same model, same API. | Beat Python with one deployment; same semantics, smaller footprint. |
| **C. Split (native)** | Embedding service (Rust, ONNX, `POST /embed` only) + thin gateway (Rust, calls embed service). Client hits gateway only. | Scale embed and search independently; still beat Python on footprint and latency. |

## Comparison table (single-node GCP, c=16, 2000 req)

Same benchmark: `python -m gateway_bench.bench http://localhost:8000 -c 16 -n 2000`. All from **instance-20260211-183413** (apples-to-apples).

| Architecture | RPS (c=16) | p95 (ms) | Total RSS | Improvement vs Python |
|--------------|------------|----------|-----------|------------------------|
| A. Python monolith | 356 | 150 | 387 MiB | baseline |
| B. Rust monolith | **456** | **123** | **126 MiB** | +28% RPS, −18% p95, **−67% RSS** |
| C. Split (Rust) | **450** | **117** | **228 MiB** (embed+gateway) | +27% RPS, −22% p95, **−41% RSS** |

**Projected savings:** At 1,000 RPS, Python needs 3 replicas (~1.16 GiB); Rust needs 3 (~378 MiB) → **~68% less memory**. Same model and API. See [results/GCP_BENCHMARK_RESULTS.md](../results/GCP_BENCHMARK_RESULTS.md).

## Reproduce steps

### A. Python monolith

```bash
pip install -r requirements.txt
uvicorn gateway_bench.service:app --host 0.0.0.0 --port 8000
# In another terminal:
python -m gateway_bench.bench http://localhost:8000 -c 16 -n 2000
```

### B. Rust monolith

```bash
python scripts/export_onnx.py -o rust-gateway/model
cd rust-gateway && cargo build --release
MODEL_DIR=model ./target/release/embedding-gateway
# In another terminal:
python -m gateway_bench.bench http://localhost:8000 -c 16 -n 2000
```

### C. Split (Rust)

```bash
cd rust-gateway && docker build -t embedding-gateway-rust .
cd .. && docker-compose up -d
python -m gateway_bench.bench http://localhost:8000 -c 16 -n 2000
```

## Decision flow

- **Scale embed and search independently or minimize blast radius?** → **Split (C)**.
- **Otherwise one service?** → **Python (A)** or **Rust (B)**.
- **Beat Python on footprint and latency?** → **Rust monolith (B)** or **Split (C)**.

## Links

- Blog: `blog/blog.md` (full narrative and numbers).
- Rust gateway (modes, embed-only, contract): `rust-gateway/README.md`.
- Main README: `README.md`.
