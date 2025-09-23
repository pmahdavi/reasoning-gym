#!/usr/bin/env python3
"""
Modal smoke test for Reasoning Gym container.

Builds the repository image from the Dockerfile on Modal, then verifies that
`flash_attn` can be imported successfully inside the container.

Usage:
  # Default: 1x A10G
  modal run modal/smoke.py

  # Specify GPU type/count
  modal run modal/smoke.py --gpu-spec A100-40GB:1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

app = modal.App("reasoning-gym-smoke")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Build image from local Dockerfile (same as training image)
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    context_dir=str(REPO_ROOT),
)


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=1800,  # 30 minutes
)
def smoke() -> dict:
    import platform

    # Import torch and flash-attn inside the container environment
    import torch
    import flash_attn

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),  # Convert to string
        "cuda_version": str(getattr(torch.version, "cuda", None)),  # Convert to string
        "gpu_count": str(torch.cuda.device_count()),  # Convert to string
        "flash_attn": str(getattr(flash_attn, "__version__", "unknown")),  # Convert to string
    }

    print("Container smoke test results:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    return info


@app.local_entrypoint()
def main(gpu_spec: Optional[str] = "A10G:1"):
    # NOTE: older versions of the modal client do not support .options() or .with_options().
    # The gpu_spec agument is ignored and the spec from the decorator is used.
    # TODO: upgrade modal client and restore dynamic gpu_spec.
    if gpu_spec != "A10G:1":
        print(f"Warning: gpu_spec='{gpu_spec}' is ignored. Using 'A10G:1' from decorator.")

    result = smoke.remote()
    print("\nSmoke test OK:", result) 