#!/bin/bash
# Wrapper script for Modal training with configurable GPU specs

# Parse arguments
CONFIG_NAME="${1:-algebra_qwen_3b}"
GPU_SPEC="${2:-A100-40GB:4}"
PROJECT_NAME="${3:-rg-grpo}"
EXPERIMENT_NAME="${4:-$CONFIG_NAME-$(date +%s)}"
OVERRIDES="${5:-}"

# Extract GPU count from GPU_SPEC (e.g., "A100-40GB:4" -> "4")
GPU_COUNT=$(echo $GPU_SPEC | cut -d: -f2)

# Set default overrides based on GPU count
if [ -z "$OVERRIDES" ]; then
    OVERRIDES="trainer.n_gpus_per_node=$GPU_COUNT actor_rollout_ref.rollout.tensor_model_parallel_size=1"
fi

echo "=========================================="
echo "Reasoning Gym Modal Training"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "GPU Spec: $GPU_SPEC"
echo "GPU Count: $GPU_COUNT"
echo "Project: $PROJECT_NAME"
echo "Experiment: $EXPERIMENT_NAME"
echo "Overrides: $OVERRIDES"
echo "=========================================="

# Export GPU spec for Modal
export MODAL_GPU_SPEC=$GPU_SPEC

# Run Modal
modal run modal/deploy.py \
    --config-name "$CONFIG_NAME" \
    --gpu-spec "$GPU_SPEC" \
    --project-name "$PROJECT_NAME" \
    --experiment-name "$EXPERIMENT_NAME" \
    --overrides "$OVERRIDES" 