#!/usr/bin/env bash
# Run on a GCP (or any Linux) instance to benchmark A (Python), B (Rust), C (Split), and Go.
# Prereqs: Docker, Python 3 with venv + pip install -r requirements-bench.txt (for bench client).
# For B/C: export ONNX model first: pip install 'optimum[onnx]' && python scripts/export_onnx.py -o rust-gateway/model
# Usage: from repo root, ./scripts/run_benchmarks_gcp.sh [--python] [--go] [--rust] [--split] (default: all that are available)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "$RESULTS_DIR"
CONCURRENCY=16
NUM_REQUESTS=2000
BASE_URL="http://127.0.0.1:8000"

run_bench() {
  local out
  out=$(python -m gateway_bench.bench --base-url "$BASE_URL" --concurrency "$CONCURRENCY" --num-requests "$NUM_REQUESTS" --output "$RESULTS_DIR/bench_$$.csv" 2>&1)
  echo "$out" >&2
  echo "$out"
  local rps p95 rss
  rps=$(echo "$out" | sed -n 's/Throughput: \([0-9.]*\) RPS/\1/p')
  p95=$(echo "$out" | sed -n 's/.*p95: \([0-9.]*\) ms.*/\1/p')
  rss=$(echo "$out" | sed -n 's/.*RSS: \([0-9.]*\) MB/\1/p')
  echo "PARSED|rps=$rps|p95=$p95|rss=$rss"
}

container_rss_mb() {
  local name="$1"
  docker stats --no-stream --format "{{.MemUsage}}" "$name" 2>/dev/null | awk '{print $1}' | sed 's/MiB//' || echo "0"
}

do_python() {
  echo "=== A. Python monolith ==="
  docker build -t embedding-gateway-python . >/dev/null 2>&1
  cid=$(docker run -d -p 8000:8000 --name hprag-python embedding-gateway-python)
  sleep 5
  until curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/healthz" | grep -q 200; do sleep 2; done
  run_bench
  rss=$(container_rss_mb hprag-python)
  echo "CONTAINER_RSS_MB=$rss"
  docker stop $cid
  docker rm $cid
}

do_go() {
  echo "=== Go gateway (pseudo-embeddings) ==="
  docker build -t hprag-gateway-go ./go-gateway >/dev/null 2>&1
  cid=$(docker run -d -p 8000:8000 --name hprag-go hprag-gateway-go)
  sleep 2
  until curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/healthz" | grep -q 200; do sleep 1; done
  run_bench
  rss=$(container_rss_mb hprag-go)
  echo "CONTAINER_RSS_MB=$rss"
  docker stop $cid
  docker rm $cid
}

do_rust() {
  echo "=== B. Rust monolith ==="
  if [[ ! -f rust-gateway/model/model.onnx ]]; then
    echo "Skip Rust: rust-gateway/model/model.onnx not found. Run: python scripts/export_onnx.py -o rust-gateway/model"
    return 0
  fi
  docker build -t embedding-gateway-rust ./rust-gateway >/dev/null 2>&1
  cid=$(docker run -d -p 8000:8000 -e MODEL_DIR=/app/model --name hprag-rust embedding-gateway-rust)
  sleep 5
  until curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/healthz" | grep -q 200; do sleep 2; done
  run_bench
  rss=$(container_rss_mb hprag-rust)
  echo "CONTAINER_RSS_MB=$rss"
  docker stop $cid
  docker rm $cid
}

do_split() {
  echo "=== C. Split (embed + gateway) ==="
  if [[ ! -f rust-gateway/model/model.onnx ]]; then
    echo "Skip Split: rust-gateway/model/model.onnx not found."
    return 0
  fi
  docker-compose down 2>/dev/null || true
  docker-compose up -d
  sleep 10
  until curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/healthz" | grep -q 200; do sleep 2; done
  run_bench
  rss_embed=$(container_rss_mb $(docker ps -q -f name=embed) 2>/dev/null || echo 0)
  rss_gw=$(container_rss_mb $(docker ps -q -f name=gateway) 2>/dev/null || echo 0)
  echo "CONTAINER_RSS_MB_EMBED=$rss_embed CONTAINER_RSS_MB_GATEWAY=$rss_gw"
  docker-compose down
}

RUN_PYTHON=false
RUN_GO=false
RUN_RUST=false
RUN_SPLIT=false
for arg in "$@"; do
  case "$arg" in
    --python) RUN_PYTHON=true ;;
    --go)     RUN_GO=true ;;
    --rust)   RUN_RUST=true ;;
    --split)  RUN_SPLIT=true ;;
  esac
done
if ! $RUN_PYTHON && ! $RUN_GO && ! $RUN_RUST && ! $RUN_SPLIT; then
  RUN_PYTHON=true
  RUN_GO=true
  RUN_RUST=true
  RUN_SPLIT=true
fi

OUT="$RESULTS_DIR/GCP_BENCHMARK_RESULTS.md"
echo -e "# GCP benchmark results\n\n**Date:** $(date -I)\n**Concurrency:** $CONCURRENCY **Requests:** $NUM_REQUESTS\n" > "$OUT"

if $RUN_PYTHON; then do_python >> "$OUT" 2>&1; fi
if $RUN_GO; then do_go >> "$OUT" 2>&1; fi
if $RUN_RUST; then do_rust >> "$OUT" 2>&1; fi
if $RUN_SPLIT; then do_split >> "$OUT" 2>&1; fi

echo "Results appended to $OUT"
