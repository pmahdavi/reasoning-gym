# Modal Deployment for Reasoning Gym

Run Reasoning Gym training on [Modal](https://modal.com) with GPUs using the repository Dockerfile.

## Prerequisites

```bash
pip install modal
modal setup  # authenticate
```

Create secrets (once):
```bash
modal secret create huggingface HF_TOKEN=your_hf_token HUGGING_FACE_HUB_TOKEN=your_hf_token
modal secret create wandb WANDB_API_KEY=your_wandb_key
```

## Smoke Test (Build + flash-attn)

Build the image from the repo Dockerfile and verify `flash_attn` imports inside the container.

```bash
# Default GPU (A10G:1)
modal run modal/smoke.py

# Specify GPU type/count
modal run modal/smoke.py --gpu-spec A100-40GB:1
```

Expected output shows Python/Torch versions, CUDA availability, GPU count, and flash_attn version.

## Quick Start: Training

```bash
# Algebra intra-generalisation (4x A100 40GB)
modal run modal/deploy.py \
  --config-name algebra_qwen_3b \
  --gpu-spec A100-40GB:4 \
  --project-name rg-grpo \
  --experiment-name intra_algebra_modal \
  --overrides trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

Smaller/cheaper run:
```bash
modal run modal/deploy.py \
  --config-name arithmetic_qwen_3b \
  --gpu-spec A100-40GB:2 \
  --project-name rg-grpo \
  --experiment-name intra_arithmetic_modal \
  --overrides trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

## Parameters

- `--config-name`: one of `algebra_qwen_3b`, `algorithmic_qwen_3b`, `arithmetic_qwen_3b`, `cognition_qwen_3b`, `games_qwen_3b`, `graphs_qwen_3b`
- `--gpu-spec`: GPU type and count, e.g. `A100-40GB:4`, `H100:8`, `A10G:2`
- `--project-name`: W&B project name and checkpoint root
- `--experiment-name`: Optional run name (auto-generated if omitted)
- `--overrides`: Hydra overrides (space-separated), e.g. `trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4`

## Outputs

- Checkpoints: stored in Modal volume at `/checkpoints/<project>/<experiment>`
- W&B: runs logged under the provided `project_name`

## Monitoring

```bash
modal app logs
modal app stats
```

## Notes

- The image is built from the repository Dockerfile on first run; subsequent runs are cached.
- Secrets are injected automatically from Modal Secret Manager.
- Volumes persist across runs (outputs, checkpoints, cache). 