FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS base

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 \
    UV_PROJECT_ENVIRONMENT="/opt/venv" \
    PATH="/opt/venv/bin:$PATH"

# Cài đặt basic dependencies (runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    espeak-ng \
    libespeak-ng1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Config python default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# --- Stage: Builder ---
FROM base AS builder

# Cài đặt build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-dev \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt pip cho 3.12 và uv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && pip install --no-cache-dir uv

WORKDIR /build
# Copy metadata files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Cài đặt dependencies (không cài đặt project chính để tiết kiệm layer)
RUN uv sync --no-dev --frozen --no-install-project

# --- Stage: Production ---
FROM base AS prod

WORKDIR /workspace

# Copy venv từ builder
COPY --from=builder /opt/venv /opt/venv
# Copy application code
COPY . .

# Port is handled by Cloud Run ($PORT)
CMD ["python", "gradio_app.py", "--server-name", "0.0.0.0"]
