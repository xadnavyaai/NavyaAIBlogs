# Go Embedding Gateway

Same HTTP API as the Python gateway (`POST /v1/ingest`, `POST /v1/search`, `GET /v1/jobs/{id}`, `GET /healthz`) for apples-to-apples benchmarking.

- **No ML runtime**: uses deterministic pseudo-embeddings (hash-based 384-d vectors) so the binary stays small and startup is instant.
- **Single static binary**: built with `CGO_ENABLED=0` for a tiny Docker image (Alpine + binary).

## Build and run

```bash
go build -o gateway .
./gateway   # listens on :8000
```

## Docker

```bash
docker build -t hprag-gateway-go .
docker run -p 8000:8000 hprag-gateway-go
```

## Benchmark with Python harness

From the parent directory (where `gateway_bench` lives):

```bash
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 --concurrency 64 --num-requests 2000 --output results/go_c64.csv
```

Compare RSS, p95, and RPS vs the Python gateway.
