#!/usr/bin/env python3
"""
Modal deployment for Reasoning Gym training.

This script builds an image from the repository Dockerfile and runs the
training entrypoint on Modal GPUs.

Usage examples:

# Algebra (4x A100 40GB)
modal run modal/deploy.py \
  --config-name algebra_qwen_3b \
  --gpu-spec A100-40GB:4 \
  --project-name rg-grpo \
  --experiment-name intra_algebra_modal \
  --overrides "trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=1"

# Arithmetic (2 GPUs, TP=1)
modal run modal/deploy.py \
  --config-name arithmetic_qwen_3b \
  --gpu-spec A100-40GB:2 \
  --project-name rg-grpo \
  --experiment-name intra_arithmetic_modal \
  --overrides "trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=1"

Secrets:
- Create secrets once (update names if different):
  modal secret create huggingface HF_TOKEN=... HUGGING_FACE_HUB_TOKEN=...
  modal secret create wandb WANDB_API_KEY=...
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional
import os

import modal

app = modal.App("reasoning-gym-training")

# Volumes for persistence
outputs_vol = modal.Volume.from_name("rg-outputs", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("rg-checkpoints", create_if_missing=True)
cache_vol = modal.Volume.from_name("rg-cache", create_if_missing=True)

# Build image from local Dockerfile (uses .dockerignore)
REPO_ROOT = Path(__file__).resolve().parents[1]

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    context_dir=str(REPO_ROOT),
)

# Get GPU spec from environment variable or use default
GPU_SPEC = os.environ.get("MODAL_GPU_SPEC", "A100-40GB:4")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=16.0,
    memory=65536,
    volumes={
        "/outputs": outputs_vol,
        "/checkpoints": checkpoints_vol,
        "/cache": cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    timeout=86400,
)
def train(
    config_name: str = "algebra_qwen_3b",
    project_name: str = "rg-grpo",
    experiment_name: Optional[str] = None,
    overrides=None,
):
    """Run a training job for an intra_generalisation config on Modal.

    Args:
        config_name: One of {algebra_qwen_3b, arithmetic_qwen_3b, algorithmic_qwen_3b, games_qwen_3b, graphs_qwen_3b, cognition_qwen_3b}
        project_name: W&B project and checkpoint root folder
        experiment_name: Unique run name; auto-generated if None
        overrides: List of Hydra overrides, e.g. ["trainer.n_gpus_per_node=4", "actor_rollout_ref.rollout.tensor_model_parallel_size=1"]
    """
    import os
    import subprocess

    # Set up environment variables
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/hf")
    os.environ.setdefault("WANDB_DIR", "/outputs/wandb")
    os.environ.setdefault("WANDB_PROJECT", project_name)
    
    # Add NCCL debugging for troubleshooting GPU coordination issues
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if experiment_name is None:
        experiment_name = f"{config_name}-{int(time.time())}"

    workdir = "/workspace"
    os.chdir(workdir)

    ckpt_dir = f"/checkpoints/{project_name}/{experiment_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        "python", "-u", "training/train_grpo.py",
        "--config-path", "configs/intra_generalisation",
        "--config-name", config_name,
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
    ]

    if overrides:
        cmd.extend(overrides)

    print("=" * 60)
    print("Reasoning Gym training on Modal")
    print("=" * 60)
    print("Command:", " ".join(cmd))
    print("GPU specification:", GPU_SPEC)
    print("Checkpoints:", ckpt_dir)
    print("W&B project:", project_name)
    print("=" * 60)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    return {
        "project": project_name,
        "experiment": experiment_name,
        "checkpoints": ckpt_dir,
    }


@app.local_entrypoint()
def main(
    config_name: str = "algebra_qwen_3b",
    gpu_spec: str = "A100-40GB:4",
    project_name: str = "rg-grpo",
    experiment_name: Optional[str] = None,
    overrides: Optional[str] = None,
):
    """Main entrypoint with GPU specification support via environment variable."""
    
    # Set GPU spec as environment variable for the decorator to use
    if gpu_spec != GPU_SPEC:
        print(f"WARNING: To use gpu_spec={gpu_spec}, please run:")
        print(f"  export MODAL_GPU_SPEC={gpu_spec}")
        print(f"  modal run modal/deploy.py ...")
        print(f"Currently using: {GPU_SPEC}")
        print()
    
    # Parse overrides string into list
    overrides_list = None
    if overrides:
        overrides_list = overrides.split()
    
    print(f"Running with GPU spec: {GPU_SPEC}")

    result = train.remote(
        config_name=config_name,
        project_name=project_name,
        experiment_name=experiment_name,
        overrides=overrides_list,
    )
    print("\nJob result:", result) 