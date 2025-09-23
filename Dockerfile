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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install packaging ninja

# Working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml README.md /workspace/
COPY reasoning_gym /workspace/reasoning_gym/
COPY training /workspace/training/
COPY tools /workspace/tools/

# Install base dependencies following the training README order
RUN pip install wheel fire

# Install project in editable mode
RUN pip install -e .

# Install verl at the specific commit 
# This will install PyTorch and other dependencies
RUN pip install git+https://github.com/volcengine/verl.git@c34206925e2a50fd452e474db857b4d488f8602d

# Fix tensordict compatibility issue with PyTorch 2.8.0
# According to verl docs, tensordict 0.6.2 is needed for newer pytorch versions
RUN pip install tensordict==0.6.2 --force-reinstall

# Upgrade vllm to a supported version (verl installs 0.2.5 which is too old)
RUN pip install vllm==0.7.3 --force-reinstall

# Get the PyTorch version that verl installed
RUN python -c "import torch; print('PyTorch version:', torch.__version__, 'CUDA:', torch.version.cuda)"

# Build flash attention from source against the current PyTorch version
RUN git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    git checkout v2.7.3 && \
    pip install . --no-build-isolation && \
    cd / && rm -rf /tmp/flash-attention

# Verify flash_attn installation
RUN python -c "import flash_attn; print('flash_attn installed successfully')"

# Install additional dependencies for training
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
    huggingface-hub>=0.19.0

# Create trainer user
RUN useradd -m -u 1000 -s /bin/bash trainer \
    && mkdir -p /workspace/checkpoints /workspace/logs /workspace/cache \
    && chown -R trainer:trainer /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV WANDB_DIR=/workspace/logs

# Disable torch.compile to avoid segmentation fault
ENV TORCH_COMPILE_DISABLE=1

# Default command
CMD ["python", "-u", "training/train_grpo.py", "--help"]