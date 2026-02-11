---
title: "Embedding + Rerank Gateways: Small Services, Big Performance Wins"
date: "2026-02-11"
author: "NavyaAI Engineering Team"
excerpt: "Every RAG system hides an Embedding + Rerank gateway. This project makes that service concrete and measurable."
tags: ["RAG", "Embeddings", "Search", "AI Infrastructure", "Scalability", "Performance"]
category: "Engineering"
readTime: "10 min read"
---

Embedding + Rerank Gateway
==========================

Most RAG products, enterprise search tools, and internal knowledge assistants quietly depend on the same thing:

> A small service that accepts chunks of text, calls an embeddings provider, stores vectors, and serves `/search` with retrieval + rerank + citations.

Why it’s widely used
--------------------

- Every RAG product, enterprise search, “chat with docs”, and internal knowledge tool needs this.
- It sits on the hot path for both ingestion and query-time search.
- Its performance directly affects user-perceived latency and infra cost.

What this project builds
------------------------

- A minimal Embedding + Rerank gateway with:
  - `POST /v1/ingest` (batch chunks, async job)
  - `POST /v1/search` (query → topK results + citations)
  - `GET /v1/jobs/{id}`
  - `GET /healthz`
- Backed by:
  - A small CPU-only embedding model
  - In-memory vector index (cosine similarity)
  - Simple similarity-based rerank
- A benchmark harness that can answer:
  - p95 latency for `/v1/search`
  - throughput (RPS) at a fixed p95
  - approximate RSS memory at 16/64/256 concurrent users
  - cold start time for the gateway process

Why it’s interesting
--------------------

- The service is mostly:
  - I/O fan-out (parallel embed calls)
  - CPU-ish glue (chunking, hashing, normalization)
  - high concurrency (many requests, many chunks)
- It is small enough to understand end-to-end, but central enough that performance differences matter.

What the blog will show
-----------------------

- A concrete implementation of a gateway most teams already need.
- Benchmarks that translate into:
  - How many concurrent searches a single instance can handle.
  - What p95/p99 latency looks like under realistic load.
  - How much memory the service actually uses.
- Design lessons for production Embedding + Rerank gateways.

