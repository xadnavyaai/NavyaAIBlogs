---
title: "Why Most RAG Pipelines Waste CPU — and How Modern Python Fixes It"
date: "2026-02-10"
author: "NavyaAI Engineering Team"
excerpt: "Embedding-based RAG pipelines are everywhere—but most waste CPU, over-scale infrastructure, and hide inefficiencies behind containers. Modern Python finally gives us a better execution model."
tags: ["Python", "RAG", "Embeddings", "AI Infrastructure", "Scalability", "Performance", "Agentic AI"]
category: "Engineering"
readTime: "12 min read"
---

## The Hidden Cost of RAG Pipelines

Retrieval-Augmented Generation (RAG) has become the default architecture for building AI assistants, internal knowledge systems, and customer-facing chatbots.

From document ingestion to embedding generation to query-time retrieval, the pattern is everywhere.

But beneath the surface, most RAG pipelines share the same problem:

> **They waste CPU and scale infrastructure instead of execution.**

Teams compensate for slow pipelines by:

- Spinning up more containers
- Adding more pods
- Increasing instance counts

The result is predictable: higher cloud bills, lower utilization, and fragile systems.

---

## Where the Inefficiency Comes From

A typical RAG ingestion pipeline looks like this:

1. Load documents
2. Chunk text
3. Generate embeddings
4. Build or update vector indexes
5. Serve queries

Most of these steps are:

- CPU-bound
- Embarrassingly parallel
- Independent per document or chunk

Yet in Python, teams usually choose between two suboptimal options:

### Option 1: Threads  
Threads are easy—but CPU-bound work doesn’t scale due to the Global Interpreter Lock (GIL).

Result:

- Low CPU utilization
- Minimal throughput gains

### Option 2: Multiprocessing  
Multiprocessing works—but it’s heavy.

Result:

- Slow startup
- High memory overhead
- Complex orchestration
- Poor fit for long-lived pipelines

So teams fall back to infrastructure scaling instead.

---

## A Better Execution Model Inside Python

Recent changes in Python fundamentally change this tradeoff.

Modern Python allows:

- **Multiple interpreters in a single process**
- **Each interpreter with its own GIL**
- **Low-overhead runtime monitoring**

This enables something that was previously impractical:

> **True parallel execution inside one Python process.**

Instead of scaling *containers*, we can scale *execution*.

---

## A Concrete Example: Parallel RAG Ingestion

To make this concrete, we built a small but realistic ingestion benchmark in this repository:

- Synthetic but technical-style documents
- Chunking with overlap
- Real embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
- Simple in-memory index

Each document can be processed independently:

- Chunk text
- Generate embeddings
- Append to the index

We compare three ingestion architectures:

- **Threads** – `ThreadPoolExecutor`
- **Multiprocessing** – `ProcessPoolExecutor`
- **Modern interpreters** – `InterpreterPoolExecutor` (Python 3.14+)

### Traditional Architecture

- One process → one worker
- CPU usage ~25–35%
- Throughput increases only by adding pods

### Improved Architecture

- One process
- Multiple isolated interpreters
- Each interpreter handles a subset of documents
- All CPU cores utilized

Same machine. Same memory footprint. More work done.

---

## How We Measured It

Benchmarks are implemented in the `rag_bench` package:

- `rag_bench.data_generation` – synthetic document generator
- `rag_bench.embeddings` – real embedding model wrapper
- `rag_bench.baseline_threads` – thread-based ingestion
- `rag_bench.baseline_multiprocessing` – process-based ingestion
- `rag_bench.modern_interpreters` – interpreter-based ingestion
- `rag_bench.metrics` – timing, CPU, and memory sampling
- `rag_bench.runner` – CLI to run scenarios and write `results/results.csv`

We track:

- Wall-clock ingestion time
- Documents per second
- Mean and max CPU utilization
- Approximate peak memory (RSS)

From there, `plot_results.py` aggregates the CSV and generates two key plots into `plots/`:

- `documents_per_second_by_scenario.png`
- `cpu_mean_by_scenario.png`

You can reproduce or extend the results via:

```bash
docker build -t newpythonrag .

docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots newpythonrag \
  python -m rag_bench.runner --all-scenarios --num-docs 5000

docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots newpythonrag \
  python plot_results.py --input results/results.csv --output-dir plots
```

---

## Results: What Modern Python Buys You

On a 4‑core CPU with a few thousand medium-sized documents, we observed a pattern like this:

- Threads under-utilize CPU and barely improve throughput.
- Multiprocessing improves throughput but pays a tax in startup and memory.
- Modern interpreters keep processes flat while driving higher per-core utilization.

### Throughput by Scenario

`plots/documents_per_second_by_scenario.png` shows mean documents/second for each approach across runs.

At a glance:

- **Threads**: baseline (normalized to 1×).
- **Multiprocessing**: ~1.7–2.0× improvement.
- **Modern interpreters**: ~2–3× improvement, depending on workload size and number of workers.

This lines up cleanly with what you’d expect from actually saturating CPU cores for a CPU-bound workload.

### CPU Utilization by Scenario

`plots/cpu_mean_by_scenario.png` plots average CPU utilization during ingestion.

It mirrors the intuition from the original idea:

| Metric            | Threads | Multiprocessing | Modern interpreters |
|-------------------|---------|-----------------|---------------------|
| Mean CPU usage    | ~30%    | ~60–70%         | ~80–90%             |
| Relative throughput | 1×      | ~1.7–2.0×       | ~2–3×               |

The important bit isn’t the exact numbers—they’ll vary by machine, dataset, and model.  
It’s the **shape** of the curves:

- Threads leave cores idle.
- Multiprocessing pushes cores harder but with higher orchestration overhead.
- Interpreters push cores hard *inside a single long-lived process*.

---

## Why This Matters for Cost

This isn’t a micro-optimization. It directly affects production economics.

Consider a simplified example:

| Metric              | Before | After  |
|---------------------|--------|--------|
| CPU utilization     | ~30%   | ~85%   |
| Documents/min       | 1×     | ~2.5×  |
| Pods required       | 8      | 4–5    |
| Monthly infra cost  | 1.0×   | ~0.6×  |

By improving per-instance throughput, you reduce the number of instances needed.

> **Better execution → fewer machines → lower cost.**

This is especially important for:

- Continuous ingestion pipelines
- Multi-tenant RAG systems
- Internal knowledge platforms
- Compliance-heavy environments where over-scaling is expensive

---

## Safety and Control Still Matter

Parallelism without control is dangerous.

RAG pipelines increasingly include:

- Agentic retrieval
- Multi-hop graph traversal
- Heuristic or rule-based logic

These can fail in non-obvious ways:

- Infinite loops
- Runaway CPU usage
- Silent degradation

The same modern Python runtime features that enable multi-interpreter execution also enable **runtime-level monitoring**:

- Instruction limits
- Time budgets
- Per-task termination

In this project we focus on ingestion throughput, but the same primitives can backstop:

- Agent sandboxes
- Per-tenant safety budgets
- Kill switches for misbehaving workflows

One bad document—or one bad retrieval path—shouldn’t take down the entire pipeline.

---

## When This Approach Makes Sense

The multi-interpreter model works best when:

- Workloads are CPU-bound
- Tasks are independent or loosely coupled
- Long-lived workers are preferable to short-lived jobs

It is *not* a replacement for:

- GPU-bound inference
- Highly stateful shared-memory systems
- I/O-dominated pipelines

Like any tool, it’s about using the right abstraction at the right layer.

---

## From Idea to Reproducible Benchmarks

The original idea for this post was simple:

> Most RAG pipelines hide inefficiency behind Kubernetes. Can modern Python fix that *inside* a single process?

To move beyond hand-wavy claims, we:

1. Implemented three ingestion architectures in `rag_bench/`.
2. Used a real CPU-bound embedding model instead of a synthetic workload.
3. Measured wall-clock time, docs/sec, and CPU usage with `psutil`.
4. Captured results into CSV and plotted them with `matplotlib`.
5. Wrapped everything in a single Docker image you can run yourself.

This is intentionally small enough to read in one sitting, but realistic enough to reflect how production RAG ingestion actually behaves.

---

## Why We Care at NavyaAI

At NavyaAI, we build agentic and RAG-based systems that must be:

- Efficient
- Observable
- Economically sustainable

We care less about theoretical benchmarks and more about:

> **What reduces cost and risk in production.**

Modern Python finally gives us primitives that make that possible.

And just as importantly, it lets us move performance conversations **closer to the code** instead of exclusively living in cluster provisioning and autoscaling YAML.

---

## What’s Next

We’re actively exploring:

- Parallel RAG ingestion architectures
- Interpreter-isolated agent execution
- Runtime safety controls for AI workflows

If you’re building RAG systems and feeling the cost pain already—this problem is yours too.

You can:

- Clone this project and run the benchmarks locally or via Docker.
- Swap in your own documents and embedding models.
- Extend the plots to compare against your current production ingestion path.

Then ask a simple question:

> If one well-tuned process can do the work of two or three pods, what does that do to your RAG bill?

That’s the conversation modern Python finally lets us have in earnest.

