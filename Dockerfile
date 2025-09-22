# Single-stage build to maximize compatibility with remote builders (e.g., Modal)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml README.md /workspace/
COPY reasoning_gym /workspace/reasoning_gym/
COPY training /workspace/training/
COPY tools /workspace/tools/

# Install project base deps first
RUN pip install -e .

# Fix NumPy compatibility issue - pin numpy<2 to avoid 1.x vs 2.x conflicts
RUN pip install "numpy<2"

# Install PyTorch (CUDA 11.8 wheels) BEFORE flash-attn
# Rely on PyTorch index for CUDA wheels to avoid source builds
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    && python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)"

# Install flash-attn compatible with CUDA 11.8 and torch 2.1.x
RUN pip install flash-attn==2.7.3 --no-build-isolation \
    && python -c "import flash_attn, sys; print('flash_attn:', getattr(flash_attn, '__version__', 'unknown'))"

# Install verl at the specific commit (this will install compatible vLLM version)
RUN pip install git+https://github.com/volcengine/verl.git@c34206925e2a50fd452e474db857b4d488f8602d

# Install remaining training/runtime dependencies (removed vllm==0.7.3 to avoid conflict)
RUN pip install \
    hydra-core>=1.3.2 \
    omegaconf>=2.3.0 \
    ray[default]>=2.9.0 \
    wandb>=0.16.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    datasets>=2.14.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    huggingface-hub>=0.19.0 \
    fire>=0.5.0

# Create non-root user and directories (Modal will ignore USER but create dirs)
RUN useradd -m -u 1000 -s /bin/bash trainer \
    && mkdir -p /workspace/checkpoints /workspace/logs /workspace/cache \
    && chown -R trainer:trainer /workspace

# Runtime env (fix PYTHONPATH syntax for Modal)
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV WANDB_DIR=/workspace/logs

# Default command
CMD ["python", "-u", "training/train_grpo.py", "--help"]