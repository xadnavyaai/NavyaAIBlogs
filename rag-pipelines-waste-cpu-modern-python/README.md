RAG Ingestion Benchmarks: Threads vs Multiprocessing
=====================================================

This project backs the blog post **"Why Threads Beat Multiprocessing for RAG Pipelines — GIL or No GIL"** with **real, measurable performance data** across Python 3.13, 3.14, and 3.14t (free-threaded / no-GIL).

**Key finding:** Threads are 35–70% faster than multiprocessing for embedding workloads and use 75% less memory — because libraries like NumPy and PyTorch release the GIL during heavy computation. This works on Python 3.8+, not just "modern" Python.

Project layout
--------------

- `blog/` – Blog post content (also deployed on navya.ai).
- `rag_bench/` – Python package with ingestion pipelines and benchmark runner.
- `results/` – Raw benchmark results (CSV) from GCP `e2-standard-4` runs.
- `plots/` – Generated comparison charts (PNG) from the results.

Results at a glance
-------------------

| Build | Threads (docs/s) | MP (docs/s) | Thread Memory | MP Memory |
|-------|------------------|-------------|---------------|-----------|
| Python 3.13 | **21.8** | 16.2 | 941 MB | 3,515 MB |
| Python 3.14 | **22.0** | 13.1 | 962 MB | 4,233 MB |
| Python 3.14t (no-GIL) | **20.5** | 13.5 | 1,060 MB | 4,551 MB |

Quickstart (local)
------------------

1. Create and activate a virtual environment (Python 3.13+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run benchmarks (threads + multiprocessing):

```bash
python -m rag_bench.runner --all-scenarios --num-docs 1000 --num-workers 4
```

3. Generate comparison plots from the results:

```bash
python plot_314_comparison.py
# or: python plot_results.py --input results/results.csv --output-dir plots
```

Docker usage
------------

```bash
docker build -t ragbench .

docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots ragbench \
  python -m rag_bench.runner --all-scenarios --num-docs 2000

docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/plots:/app/plots ragbench \
  python plot_314_comparison.py
```

Running on Python 3.14t (free-threaded)
----------------------------------------

```bash
# Install python3.14 and python3.14-nogil from deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.14 python3.14-venv python3.14-nogil

# Create venvs
python3.14 -m venv .venv314
python3.14t -m venv .venv314t

# Install deps (3.14t needs Rust + gcc for tokenizers/safetensors)
.venv314/bin/pip install -r requirements.txt
.venv314t/bin/pip install -r requirements.txt

# Run with tags to label results
.venv314/bin/python -m rag_bench.runner --all-scenarios --num-docs 1000 --tag "py314-"
.venv314t/bin/python -m rag_bench.runner --all-scenarios --num-docs 1000 --tag "py314t-"
```

Related
-------

- **Blog post**: [navya.ai/blog/rag-pipelines-waste-cpu-modern-python](https://navya.ai/blog/rag-pipelines-waste-cpu-modern-python)
- **Code repo**: [github.com/xadnavyaai/NavyaAIBlogs](https://github.com/xadnavyaai/NavyaAIBlogs) (folder `rag-pipelines-waste-cpu-modern-python/`)
- **Frontend site**: [github.com/xadnavyaai/NavyaAI-web](https://github.com/xadnavyaai/NavyaAI-web)

Notes
-----

- The embedding model is `sentence-transformers/all-MiniLM-L6-v2` (real CPU-bound workload).
- `InterpreterPoolExecutor` (Python 3.14) doesn't work yet for ML workloads — NumPy/PyTorch can't load in sub-interpreters.
- The `--tag` flag lets you label results by Python version for comparison plots.
