# GCP instance benchmark results: Go gateway

**Date:** 2026-02-11  
**Instance:** GCP VM (us-central1-c), project glassy-proton-480109-d6  
**Bench:** Python `gateway_bench.bench` (httpx) against Go gateway; 2000 requests per run.

## Go gateway (hprag-gateway-go)

| Concurrency | Throughput (RPS) | p50 (ms) | p95 (ms) | p99 (ms) | RSS (MB) |
|-------------|------------------|----------|----------|----------|----------|
| 1           | 224.8            | 4.21     | 8.26     | 11.00    | 35.0     |
| 16          | 203.8            | 35.13    | 259.49   | 450.37   | 36.5     |
| 64          | 175.7            | 225.66   | 992.50   | 1631.14  | 38.2     |

- **Docker image size:** ~13 MB  
- **Note:** Go gateway uses deterministic pseudo-embeddings (no ML runtime). For Python gateway (sentence-transformers) run the same bench against the Python container to compare RSS and image size; throughput/latency are not directly comparable because workload differs.
