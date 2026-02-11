# Run benchmarks on a GCP instance

Use a GCP VM to run all architectures (Python, Go, Rust monolith, Split) and collect numbers. Same steps work on any Linux host with Docker and Python.

## 1. Create / use a GCP VM

- **Region:** e.g. `us-central1-c` (same as existing results).
- **Machine:** e.g. 4 vCPU, 16 GB RAM.
- **Image:** Ubuntu 22.04 LTS (or any with Docker and Python 3).
- **Firewall:** allow HTTP (or at least allow your IP for SSH; bench runs on the VM).

## 2. Copy the project to the VM

From your **local machine** (where the repo lives):

```bash
# Replace INSTANCE with your VM name or use: user@EXTERNAL_IP
# Example: gcloud compute scp --recurse ... user@104.155.173.193:~/HPRAG
cd /path/to/TechBlogs/HPRAG
tar czf - embedding-rerank-gateway-high-performance | gcloud compute ssh INSTANCE -- "mkdir -p ~/HPRAG && tar xzf - -C ~/HPRAG"
```

Or clone on the VM if the repo is public:

```bash
gcloud compute ssh INSTANCE
git clone https://github.com/your-org/NavyaAIBlogs.git
cd NavyaAIBlogs/embedding-rerank-gateway-high-performance   # or the path where the gateway lives
```

## 3. On the GCP instance: install prereqs

```bash
# Docker
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
# Log out and back in so docker runs without sudo, or run the script with sudo

# Python + venv (for the bench client)
sudo apt-get install -y python3-venv python3-pip
cd ~/HPRAG/embedding-rerank-gateway-high-performance   # or your path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. (Optional) Export ONNX model for Rust and Split

Needed only for **B. Rust monolith** and **C. Split**:

```bash
pip install 'optimum[onnx]' transformers
python scripts/export_onnx.py -o rust-gateway/model
```

## 5. Run the benchmark script

```bash
cd ~/HPRAG/embedding-rerank-gateway-high-performance
source .venv/bin/activate
chmod +x scripts/run_benchmarks_gcp.sh
./scripts/run_benchmarks_gcp.sh
```

To run only specific architectures:

```bash
./scripts/run_benchmarks_gcp.sh --python --go
./scripts/run_benchmarks_gcp.sh --rust --split
```

Output is appended to `results/GCP_BENCHMARK_RESULTS.md`. Copy the file (or paste the summary) into the blog or `results/` and fill the case study table.

## 6. One-off manual runs (alternative)

If you prefer to run each architecture by hand:

**A. Python monolith**

```bash
docker build -t embedding-gateway-python .
docker run -d -p 8000:8000 --name py embedding-gateway-python
sleep 10
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 -c 16 -n 2000
docker stats --no-stream py
docker stop py && docker rm py
```

**Go gateway**

```bash
cd go-gateway && docker build -t hprag-go . && cd ..
docker run -d -p 8000:8000 --name go hprag-go
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 -c 16 -n 2000
docker stop go && docker rm go
```

**B. Rust monolith** (after exporting model)

```bash
cd rust-gateway && docker build -t embedding-gateway-rust . && cd ..
docker run -d -p 8000:8000 -e MODEL_DIR=/app/model --name rust embedding-gateway-rust
sleep 10
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 -c 16 -n 2000
docker stop rust && docker rm rust
```

**C. Split**

```bash
docker-compose up -d
sleep 15
python -m gateway_bench.bench --base-url http://127.0.0.1:8000 -c 16 -n 2000
docker-compose down
```

Use the GCP instance so numbers are comparable (same machine as in `results/GCP_GO_GATEWAY_RESULTS.md`).
