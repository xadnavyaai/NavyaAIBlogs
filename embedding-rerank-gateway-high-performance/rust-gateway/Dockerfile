# Build stage: need glibc 2.38+ for ort/ONNX Runtime (__isoc23_strtol etc). Use Ubuntu 24.04.
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_INIT_SKIP_PATH_CHECK=yes
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy manifest and source
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

# Copy pre-exported ONNX model and tokenizer (run scripts/export_onnx.py first)
COPY model ./model

RUN cargo build --release

# Runtime stage (same glibc as builder for binary compatibility)
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/embedding-gateway /app/gateway
COPY --from=builder /build/model /app/model

ENV MODEL_DIR=/app/model

EXPOSE 8000

CMD ["/app/gateway"]
