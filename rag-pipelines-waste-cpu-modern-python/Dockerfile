FROM python:3.14-slim

WORKDIR /app

# Install system packages needed by sentence-transformers (if any).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code.
COPY . .

# Optionally pre-download the embedding model so first run is faster.
RUN python - <<'PY'
from rag_bench import embeddings  # will download the default model
try:
    embeddings.get_model()
except Exception as exc:
    # Model download may fail in restricted build environments; ignore.
    print("Warning: failed to pre-load embedding model:", exc)
PY

# Default command: run all scenarios with a moderate document count.
CMD ["python", "-m", "rag_bench.runner", "--all-scenarios", "--num-docs", "2000"]

