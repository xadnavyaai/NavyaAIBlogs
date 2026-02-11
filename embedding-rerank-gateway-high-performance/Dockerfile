FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies (sentence-transformers may use git).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install a CPU-only PyTorch build explicitly, then the rest of the deps.
# This avoids pulling large CUDA/NVIDIA runtime packages on CPU-only hosts.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code.
COPY . .

# Optionally pre-download the embedding model so first run is faster.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
try:
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as exc:
    # Model download may fail in restricted build environments; ignore.
    print("Warning: failed to pre-load embedding model:", exc)
PY

# Default command: run the gateway on port 8000.
CMD ["uvicorn", "gateway_bench.service:app", "--host", "0.0.0.0", "--port", "8000"]

