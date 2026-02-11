---
title: "Why Most RAG Pipelines Waste CPU — and How Modern Python Fixes It"
date: "2026-02-10"
author: "NavyaAI Engineering Team"
excerpt: "We benchmarked RAG ingestion across Python 3.13, 3.14, and 3.14t (free-threaded). The results surprised us: threads already win for embedding workloads, InterpreterPoolExecutor can't load NumPy yet, and the no-GIL build adds overhead. Here's what actually matters for your infra bill."
coverImage: "/blog/rag-pipelines-waste-cpu-modern-python/hero.png"
tags:
  - "Python"
  - "RAG"
  - "Embeddings"
  - "AI Infrastructure"
  - "Scalability"
  - "Performance"
  - "Agentic AI"
category: "Engineering"
readTime: "12 min read"
featured: true
---

# Why Most RAG Pipelines Waste CPU — and How Modern Python Fixes It

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

Yet in Python, teams usually choose between two suboptimal options.

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

To make this concrete, we built a small but realistic ingestion benchmark:

- Synthetic but technical-style documents  
- Chunking with overlap  
- Real embedding model (`sentence-transformers/all-MiniLM-L6-v2`)  
- Simple in-memory index  

Each document can be processed independently:

- Chunk text  
- Generate embeddings  
- Append to the index  

We tested two execution models across **three Python builds**:

- **Threads** – `ThreadPoolExecutor`  
- **Multiprocessing** – `ProcessPoolExecutor`  

On:

- **Python 3.13** – standard GIL  
- **Python 3.14** – standard GIL + `InterpreterPoolExecutor` available  
- **Python 3.14t** – free-threaded build (no GIL)  

We also attempted `InterpreterPoolExecutor` on Python 3.14 — with an important result we'll discuss below.

---

## How We Measured It

The benchmark is implemented as a small Python package:

- Synthetic document generator  
- Real embedding model wrapper (`sentence-transformers/all-MiniLM-L6-v2`)  
- Thread-based and process-based ingestion pipelines  
- Metrics module for timing and resource sampling (including child processes)  
- CLI runner that writes structured results to a CSV file  

We track:

- Wall-clock ingestion time  
- Documents per second  
- Mean and max CPU utilization (process tree: parent + all children)  
- Approximate peak memory (RSS, process tree)  

**Methodology note:** All benchmarks were run on a dedicated GCP `e2-standard-4` instance (4 vCPUs, 16 GB RAM) running Ubuntu 24.04, with no competing workloads. Each scenario was run at 1,000, 5,000, and 10,000 document scales. We tested three Python builds: Python 3.13.1, Python 3.14.3 (standard), and Python 3.14.3 free-threaded (`python3.14t`, GIL disabled). CPU utilization is measured via `psutil` across the full process tree.

---

## Results: What We Actually Found

We ran the same benchmark across Python 3.13, 3.14, and 3.14t (free-threaded) on a dedicated 4-vCPU GCP instance. The results were not what we expected.

### Throughput: Python 3.13 vs 3.14 vs 3.14t

![RAG ingestion throughput comparison](/blog/rag-pipelines-waste-cpu-modern-python/throughput_314_comparison.png)

| Build | Threads (docs/s) | Multiprocessing (docs/s) |
|-------|------------------|--------------------------|
| Python 3.13 | **21.8** | 16.2 |
| Python 3.14 | **22.0** | 13.1 |
| Python 3.14t (no-GIL) | **20.5** | 13.5 |

The key finding: **threads are consistently fastest across all Python versions** — and the free-threaded build (3.14t) is actually *slightly slower* than the standard GIL build.

Why? The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) releases the GIL during C-level BLAS operations. Threads already parallelize the heavy computation. Removing the GIL doesn't help — it adds overhead from the free-threading synchronization mechanisms.

### CPU Utilization

![CPU utilization comparison](/blog/rag-pipelines-waste-cpu-modern-python/cpu_314_comparison.png)

All three thread builds burn ~340–354% CPU on a 4-vCPU machine. Multiprocessing stays under 6% (measured from the parent process tree).

### Memory

![Memory usage comparison](/blog/rag-pipelines-waste-cpu-modern-python/memory_314_comparison.png)

| Build | Threads (MB) | Multiprocessing (MB) |
|-------|-------------|---------------------|
| Python 3.13 | 941 | 3,515 |
| Python 3.14 | 962 | 4,233 |
| Python 3.14t (no-GIL) | 1,060 | 4,551 |

The free-threaded build uses ~12% more memory for threads and ~30% more for multiprocessing compared to 3.13. Each new Python version adds some overhead — the 3.14 runtime is larger, and the no-GIL build needs additional per-object synchronization state.

### What About InterpreterPoolExecutor?

Python 3.14 ships `InterpreterPoolExecutor` in `concurrent.futures` — each worker runs in its own sub-interpreter with its own GIL. In theory, this is the best of both worlds: process-level isolation with thread-level overhead.

In practice, **it doesn’t work yet for embedding workloads**. When we tried it:

```
module numpy._core._multiarray_umath does not support loading in subinterpreters
```

NumPy, PyTorch, and most C extensions haven’t been updated to support sub-interpreters. The arguments and return values must also be “shareable” types (no dicts, no numpy arrays). This is a fundamental ecosystem limitation that will take time to resolve.

`InterpreterPoolExecutor` is real and working for **pure-Python** workloads. But for anything involving C extensions — which is virtually all ML/embedding work — it’s not usable today.

### Efficiency Map

![Efficiency comparison](/blog/rag-pipelines-waste-cpu-modern-python/efficiency_314_comparison.png)

This scatter plot shows the full picture: throughput vs CPU usage, with bubble size proportional to memory. Threads cluster in the top-right — fast but CPU-hungry. Multiprocessing sits in the bottom-left — slow, low apparent CPU, but with massive memory bubbles.

---

## Why This Matters for Cost

This isn’t a micro-optimization. It directly affects production economics.

Here’s what our benchmarks show on a dedicated 4-vCPU instance:

| Metric                       | Multiprocessing | Threads |
|------------------------------|----------------|---------|
| Throughput (docs/s)          | ~13–16         | ~21–22  |
| Memory per instance          | ~3.5–4.6 GB   | ~1 GB   |
| Pods for 100 docs/s          | 7–8            | 5       |
| Monthly infra cost           | 1.0×           | ~0.6×   |

By choosing threads (which already parallelize well for GIL-releasing C extensions), you get **46% more throughput** with **75% less memory per instance**.

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

In our benchmark we focus on ingestion throughput, but the same primitives can backstop:

- Agent sandboxes  
- Per-tenant safety budgets  
- Kill switches for misbehaving workflows  

One bad document—or one bad retrieval path—shouldn’t take down the entire pipeline.

---

## What Actually Works Today

Based on our benchmarks across three Python builds, here’s the practical guidance:

**Use threads (`ThreadPoolExecutor`) when:**

- Your embedding model or C extension releases the GIL during computation (most do)  
- Tasks are independent per document or chunk  
- You want maximum throughput with minimal memory  

**Use multiprocessing (`ProcessPoolExecutor`) when:**

- You need true process isolation (fault tolerance, security boundaries)  
- Your workload involves pure-Python code that holds the GIL  
- Memory overhead is acceptable  

**Wait on `InterpreterPoolExecutor` until:**

- NumPy, PyTorch, and other C extensions add sub-interpreter support  
- The shareable types constraint is relaxed (or you’re doing pure-Python work)  
- Python 3.15+ matures the ecosystem  

**The free-threaded build (3.14t) is interesting but premature for production:**

- Slight throughput regression for workloads where C extensions already release the GIL  
- Higher memory overhead (~12% for threads, ~30% for multiprocessing)  
- Real wins will come for pure-Python CPU-bound code that previously couldn’t parallelize  

---

## From Idea to Reproducible Benchmarks

The original idea for this post was simple:

> Most RAG pipelines hide inefficiency behind Kubernetes. Can modern Python fix that *inside* a single process?

To move beyond hand-wavy claims, we:

1. Implemented thread-based and process-based ingestion pipelines.  
2. Used a real CPU-bound embedding model instead of a synthetic workload.  
3. Measured wall-clock time, docs/sec, and CPU usage with `psutil`.  
4. Captured results into CSV and plotted them with `matplotlib`.  
5. Tested across Python 3.13, 3.14, and 3.14t (free-threaded) on a clean GCP instance.  
6. Wrapped everything in a single Docker image you can run yourself.  

The full benchmark code, raw results CSV, and plotting scripts are open source:

> **[github.com/xadnavyaai/NavyaAIBlogs/rag-pipelines-waste-cpu-modern-python](https://github.com/xadnavyaai/NavyaAIBlogs/tree/main/rag-pipelines-waste-cpu-modern-python)**

This setup is intentionally small enough to read in one sitting, but realistic enough to reflect how production RAG ingestion actually behaves.

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

- Thread-optimised RAG ingestion with GIL-releasing embedding models  
- Monitoring `InterpreterPoolExecutor` ecosystem readiness (NumPy, PyTorch sub-interpreter support)  
- Runtime safety controls for agentic AI workflows  
- The free-threaded build’s impact on pure-Python CPU-bound workloads  

If you’re building RAG systems and feeling the cost pain already—this problem is yours too.

You can:

- [Clone the benchmark repo](https://github.com/xadnavyaai/NavyaAIBlogs/tree/main/rag-pipelines-waste-cpu-modern-python) and run the benchmarks locally or via Docker.  
- Swap in your own documents and embedding models.  
- Extend the plots to compare against your current production ingestion path.  

Then ask a simple question:

> If one well-tuned process can do the work of two or three pods, what does that do to your RAG bill?  

That’s the conversation modern Python finally lets us have in earnest.

