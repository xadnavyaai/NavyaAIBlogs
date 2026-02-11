# Rust super gateway (one binary, three modes)

One Rust binary that covers both the **Go** and **Rust (ONNX)** use cases, plus a **pseudo** mode for dev/bench. Same API as Python/Go.

| Mode   | Env / condition           | Use case |
|--------|---------------------------|----------|
| **ONNX**  | `MODEL_DIR` set, model exists | Same model as Python → apples-to-apples. |
| **Remote**| `EMBEDDING_API_URL` set   | Thin layer: gateway calls your embedding API (replaces Go thin). |
| **Pseudo**| neither set              | Deterministic vectors, no ML (like Go’s no-ML mode) for dev/bench. |
| **Embed-only** | `SERVE_EMBED_ONLY=true` | Embedding service only (Architecture C): exposes `POST /embed` for the thin gateway to call. |

## Embed-only service (Architecture C: split)

When `SERVE_EMBED_ONLY=true`, the binary runs as an **embedding service** only: it loads the ONNX model and exposes **`POST /embed`** and nothing else (no `/v1/ingest`, `/v1/search`, or index). Use it with a second process running as the thin gateway with `EMBEDDING_API_URL` pointing at this service.

- **Contract:** `POST /embed` — body `{"texts": ["...", ...]}`, response `{"embeddings": [[f32, ...], ...]}` (384-dim, L2-normalized).
- **Env:** `MODEL_DIR` (required), `PORT` (default 8080).

```bash
# Terminal 1: embedding service
SERVE_EMBED_ONLY=true PORT=8080 MODEL_DIR=model ./target/release/embedding-gateway

# Terminal 2: thin gateway
EMBEDDING_API_URL=http://localhost:8080/embed ./target/release/embedding-gateway
```

Then run the same benchmark against the gateway (port 8000). See the repo root for docker-compose (split setup).

## Why merge Rust and Go into one?

- **One codebase**: same API, tests, and docs for “with model” and “thin” and “pseudo.”
- **One binary**: choose behavior with env vars; no need to ship both a Go and a Rust gateway.
- **Advantages**: You keep a single gateway to maintain; you can still build a “slim” image (e.g. no model, `EMBEDDING_API_URL` only) or a “full” image (with ONNX + model). Go remains useful if you want the smallest possible thin binary with zero C/FFI deps; the Rust super gateway gives you one stack that can do everything.

## 1. ONNX mode (same model as Python)

Export the model once:

```bash
pip install 'optimum[onnx]' transformers
python scripts/export_onnx.py -o rust-gateway/model
```

Then run with the model:

```bash
cd rust-gateway
cargo build --release
MODEL_DIR=model ./target/release/embedding-gateway
```

## 2. Remote mode (thin layer, like Go)

Set `EMBEDDING_API_URL` to an HTTP endpoint that accepts `POST` with JSON `{"texts": ["...", ...]}` and returns `{"embeddings": [[...], ...]}` (384-dim vectors). The gateway will call it for ingest and search.

```bash
EMBEDDING_API_URL=http://your-embedding-service/embed ./target/release/embedding-gateway
```

## 3. Pseudo mode (no model, no API)

If neither `MODEL_DIR` nor `EMBEDDING_API_URL` is set, the gateway runs with deterministic pseudo-embeddings (no ML, same idea as the Go gateway’s no-ML mode). Good for local dev or benchmarking the gateway path without a real model.

```bash
./target/release/embedding-gateway
```

## Docker

Build with the model for ONNX mode (export the model first):

```bash
docker build -t embedding-gateway-rust .
docker run -p 8000:8000 embedding-gateway-rust
```

For a slim image (Remote or Pseudo only), you can build without copying the model and run with `EMBEDDING_API_URL` or nothing.

## Benchmark

Same harness as Python/Go:

```bash
python -m gateway_bench.bench http://localhost:8000 -c 16 -n 2000
```

## API

Same as Python/Go: `POST /v1/ingest`, `GET /v1/jobs/{id}`, `POST /v1/search`, `GET /healthz`.
