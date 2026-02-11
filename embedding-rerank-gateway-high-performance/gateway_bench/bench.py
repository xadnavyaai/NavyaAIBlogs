import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
import psutil


@dataclass
class RequestTiming:
    start_ts: float
    end_ts: float

    @property
    def latency_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000.0


def _measure_memory_mb(pid: int) -> float:
    """RSS of the given process (this bench client). For gateway RSS use e.g. docker stats."""
    proc = psutil.Process(pid)
    try:
        rss = proc.memory_info().rss
    except psutil.Error:
        rss = 0
    return rss / (1024 * 1024)


async def _run_worker(client: httpx.AsyncClient, base_url: str, num_requests: int, timings: List[RequestTiming]) -> None:
    for _ in range(num_requests):
        start = time.time()
        try:
            resp = await client.post(
                f"{base_url}/v1/search",
                json={"query": "example query about RAG gateways", "top_k": 5},
                timeout=30.0,
            )
            resp.raise_for_status()
        finally:
            end = time.time()
            timings.append(RequestTiming(start_ts=start, end_ts=end))


async def run_benchmark(base_url: str, concurrency: int, num_requests: int, output: str) -> None:
    """
    Fire num_requests total across concurrency workers and record per-request timings.

    The total number of requests is num_requests; each worker sends roughly
    num_requests / concurrency requests.
    """
    import asyncio

    per_worker = num_requests // concurrency
    remainder = num_requests % concurrency

    timings: List[RequestTiming] = []
    pid = psutil.Process().pid

    async with httpx.AsyncClient() as client:
        # Warmup
        for _ in range(5):
            await client.post(f"{base_url}/v1/search", json={"query": "warmup", "top_k": 3})

        start_wall = time.time()
        tasks = []
        for i in range(concurrency):
            count = per_worker + (1 if i < remainder else 0)
            tasks.append(_run_worker(client, base_url, count, timings))

        await asyncio.gather(*tasks)
        end_wall = time.time()

    # Compute summary stats
    latencies = [t.latency_ms for t in timings]
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies)) - 1]
    p99 = latencies[int(0.99 * len(latencies)) - 1]

    duration = end_wall - start_wall
    throughput = len(latencies) / duration if duration > 0 else 0.0
    mem_mb = _measure_memory_mb(pid)

    print(f"Completed {len(latencies)} requests in {duration:.2f}s")
    print(f"Throughput: {throughput:.1f} RPS")
    print(f"p50: {p50:.2f} ms, p95: {p95:.2f} ms, p99: {p99:.2f} ms")
    print(f"Approx. RSS (bench client): {mem_mb:.1f} MB")

    # Write raw timings so plots can be generated later.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_ts", "end_ts", "latency_ms", "concurrency"])
        for t in timings:
            writer.writerow([t.start_ts, t.end_ts, t.latency_ms, concurrency])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /v1/search latency and throughput.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Gateway base URL")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent workers")
    parser.add_argument("--num-requests", type=int, default=1000, help="Total number of requests")
    parser.add_argument("--output", type=str, default="results/search_bench.csv", help="CSV file to write timings")

    args = parser.parse_args()

    import asyncio

    asyncio.run(run_benchmark(args.base_url, args.concurrency, args.num_requests, args.output))


if __name__ == "__main__":
    main()

