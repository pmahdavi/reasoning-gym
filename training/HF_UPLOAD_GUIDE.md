# HuggingFace Hub Upload Guide

This guide explains how to automatically upload your trained models (with optimizer states) to HuggingFace Hub at the end of training.

## Features

✅ **Automatic upload** at the end of training  
✅ **Model weights** in HuggingFace format  
✅ **Optimizer states** (optional) for reproducibility  
✅ **Tokenizer and config** files included  
✅ **Private or public** repositories  

## Quick Start

### 1. Configure HuggingFace Authentication

First, set your HuggingFace token:

```bash
# Option 1: Login via CLI
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN="your_token_here"
```

### 2. Add HF Upload Configuration

Add the following to your YAML config file:

```yaml
hf_upload:
  enabled: true
  repo_id: "your-username/your-model-name"
  private: false  # Set to true for private repos
  optimizer_save_mode: "full"  # or null to skip optimizer
```

### 3. Run Training

Train your model as usual. The upload will happen automatically at the end:

```bash
python train_grpo.py --config-path configs --config-name your_config
```

## Configuration Options

### Full Configuration Block

```yaml
hf_upload:
  # Enable/disable automatic upload
  enabled: true  # Set to false to disable
  
  # Repository ID (required if enabled=true)
  repo_id: "username/model-name"
  
  # Optional: Upload to an organization
  organization: null  # or "your-org-name"
  
  # Repository visibility
  private: false  # true = private, false = public
  
  # Optimizer state handling
  optimizer_save_mode: "full"  # "full" or null
  # - "full": Saves consolidated optimizer states (~3-5GB for 3B models)
  # - null: Skip optimizer states (faster upload, ~1.5GB for 3B models)
```

### Optimizer Save Modes

**"full" mode** (Recommended for reproducibility):
- ✅ Gathers all sharded optimizer states from all ranks
- ✅ Consolidates into single `optimizer.pt` file
- ✅ Allows others to resume training with exact optimizer state
- ⚠️  Larger file size (typically 2-3x model size)
- ⚠️  Takes longer to upload

**null mode** (Faster, model-only):
- ✅ Only uploads model weights and tokenizer
- ✅ Smaller file size
- ✅ Faster upload
- ❌ Can't resume training with exact optimizer state

## Example Configs

### Example 1: Full Upload with Optimizer States

```yaml
# See: training/configs/hf_upload_example.yaml
hf_upload:
  enabled: true
  repo_id: "myusername/reasoning-gym-qwen-3b-algorithmic"
  private: false
  optimizer_save_mode: "full"
```

### Example 2: Model-Only Upload (No Optimizer)

```yaml
hf_upload:
  enabled: true
  repo_id: "myusername/reasoning-gym-qwen-3b-games"
  private: true
  optimizer_save_mode: null
```

### Example 3: Organization Upload

```yaml
hf_upload:
  enabled: true
  repo_id: "my-trained-model"
  organization: "my-org"
  private: false
  optimizer_save_mode: "full"
```

## What Gets Uploaded

After training completes, your HuggingFace repository will contain:

```
your-username/your-model-name/
├── config.json              # Model configuration
├── generation_config.json   # Generation settings (if present)
├── pytorch_model.bin        # Model weights (consolidated)
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json    # Tokenizer config
├── special_tokens_map.json  # Special tokens
└── optimizer.pt             # Optimizer states (if optimizer_save_mode="full")
```

## How It Works

### Under the Hood

1. **At training end**: When `is_last_step=True`, the trainer calls `_upload_to_huggingface()`

2. **Loading sharded checkpoints**: Reads veRL's checkpoint format:
   ```
   checkpoints/project/experiment/global_step_500/actor/
   ├── model_world_size_4_rank_0.pt
   ├── model_world_size_4_rank_1.pt
   ├── model_world_size_4_rank_2.pt
   ├── model_world_size_4_rank_3.pt
   ├── optim_world_size_4_rank_0.pt  (if optimizer_save_mode="full")
   ├── optim_world_size_4_rank_1.pt
   ├── optim_world_size_4_rank_2.pt
   └── optim_world_size_4_rank_3.pt
   ```

3. **Gathering**: Consolidates all shards into single files (rank 0 only)

4. **Uploading**: Uses `HfApi` to upload the consolidated checkpoint

### Comparison with Manual Conversion

**Old workflow** (manual):
```bash
# 1. Train model
python train_grpo.py --config-path configs --config-name my_config

# 2. Manually convert FSDP to HF format
python utils/load_fsdp_to_hf.py \
    checkpoints/my-exp/global_step_500/actor/ \
    Qwen/Qwen2.5-3B-Instruct \
    my_converted_model

# 3. Manually upload to HF (not included in old script)
# ... manual steps ...
```

**New workflow** (automatic):
```bash
# Just train - upload happens automatically!
python train_grpo.py --config-path configs --config-name my_config
```

## Troubleshooting

### "Permission denied" error

Make sure you're authenticated with HuggingFace:
```bash
huggingface-cli login
```

### "Repository already exists" error

This is fine! The uploader will update the existing repository.

### Out of Memory during upload

If you run out of memory while gathering optimizer states:
1. Set `optimizer_save_mode: null` to skip optimizer states
2. Or ensure you have enough CPU RAM (typically need ~10GB for 3B models with optimizer)

### Upload fails but training continues

The upload is wrapped in a try-catch, so training won't crash if upload fails.
Check the error message and ensure:
- HuggingFace token is valid
- Repository name is valid (lowercase, hyphens, no special chars)
- You have write permissions

## Advanced Usage

### Programmatic Upload

You can also use the uploader programmatically:

```python
from utils.hf_uploader import upload_checkpoint_to_hf

# Upload a specific checkpoint
upload_checkpoint_to_hf(
    checkpoint_dir="checkpoints/my-exp/global_step_500/actor",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    repo_id="username/my-model",
    step=500,
    optimizer_save_mode="full",
)
```

### Using with Existing Checkpoints

Upload any existing veRL checkpoint:

```python
from utils.hf_uploader import HuggingFaceUploader

uploader = HuggingFaceUploader(
    repo_id="username/my-old-checkpoint",
    optimizer_save_mode="full",
)

uploader.upload_from_checkpoint_dir(
    checkpoint_dir="checkpoints/old-exp/global_step_300/actor",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    step=300,
)
```

## Comparison with prime-rl

This implementation is based on [prime-rl's HuggingFace uploader](https://github.com/PrimeIntellect-ai/prime-rl), adapted for veRL's checkpoint format:

| Feature | prime-rl | reasoning-gym |
|---------|----------|---------------|
| Model upload | ✅ | ✅ |
| Optimizer upload | ✅ | ✅ |
| Checkpoint format | FSDP state_dict API | veRL sharded format |
| Integration | Built-in | Via this feature |
| API | `get_state_dict()` | Shard consolidation |

## Contributing

Found a bug or have a suggestion? Please open an issue on the reasoning-gym repository!

