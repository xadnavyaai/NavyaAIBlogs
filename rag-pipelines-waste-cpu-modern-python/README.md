NewPythonRAG – RAG Ingestion Performance Benchmarks
===================================================

This project backs the blog idea **“Why Most RAG Pipelines Waste CPU — and How Modern Python Fixes It”** with **real, measurable RAG ingestion performance data** using modern Python concurrency and a real embedding model. It compares three ingestion architectures:

- Baseline threads
- Multiprocessing
- Modern Python multi-interpreter style (via `InterpreterPoolExecutor` when available)

The benchmarks produce **crisp, plottable metrics** (documents/sec, CPU utilization) and charts you can embed directly in a blog post or slide deck.

Project layout
--------------

- `blog/` – Draft blog post content and narrative.
- `rag_bench/` – Python package with ingestion pipelines and benchmark runner.
- `results/` – Raw benchmark results (CSV/JSONL).
- `plots/` – Generated charts (PNG/SVG) from the results.

Quickstart (local)
------------------

1. Create and activate a virtual environment (Python 3.13+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a small benchmark locally (threads, multiprocessing, modern interpreters):

```bash
python -m rag_bench.runner --all-scenarios --num-docs 500
```

3. Generate plots from the results:

```bash
python plot_results.py --input results/results.csv --output-dir plots
```

Docker usage
------------

You can also run everything in a single container:

```bash
docker build -t newpythonrag .

docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots newpythonrag \
  python -m rag_bench.runner --all-scenarios --num-docs 2000

docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots newpythonrag \
  python plot_results.py --input results/results.csv --output-dir plots
```

The resulting PNGs in `plots/` can be embedded directly into the blog in `blog/blog.md` or into the NavyaAI frontend blog.

Related blog & frontend
-----------------------

- **Code repo (this project)**: `https://github.com/xadnavyaai/NavyaAIBlogs` (folder `rag-pipelines-waste-cpu-modern-python/`).
- **Frontend site implementation**: see the NavyaAI web repo at `https://github.com/xadnavyaai/NavyaAI-web` for how this benchmark is surfaced as a blog post.

Notes
-----

- The default embedding backend uses a real `sentence-transformers` model.
- For environments without internet/model access, you can later add a synthetic fallback embedding mode for testing-only runs.

