#!/usr/bin/env python
# encoding: utf-8
"""
Standalone script to upload existing veRL checkpoints to HuggingFace Hub.

This is useful for uploading checkpoints from previous training runs
that didn't have HF upload configured.

Usage:
    python scripts/upload_checkpoint_to_hf.py \\
        --checkpoint-dir checkpoints/my-exp/global_step_500/actor \\
        --model-name Qwen/Qwen2.5-3B-Instruct \\
        --repo-id username/my-model \\
        --step 500 \\
        --optimizer-save-mode full \\
        --private

Example:
    # Upload with optimizer states
    python scripts/upload_checkpoint_to_hf.py \\
        --checkpoint-dir checkpoints/rg-test/algorithmic/global_step_500/actor \\
        --model-name Qwen/Qwen2.5-3B-Instruct \\
        --repo-id myuser/reasoning-gym-qwen-3b \\
        --step 500 \\
        --optimizer-save-mode full
    
    # Upload without optimizer states (faster)
    python scripts/upload_checkpoint_to_hf.py \\
        --checkpoint-dir checkpoints/rg-test/algorithmic/global_step_500/actor \\
        --model-name Qwen/Qwen2.5-3B-Instruct \\
        --repo-id myuser/reasoning-gym-qwen-3b \\
        --step 500
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hf_uploader import upload_checkpoint_to_hf


def main():
    parser = argparse.ArgumentParser(
        description="Upload veRL checkpoint to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to veRL checkpoint directory (e.g., checkpoints/.../global_step_500/actor)",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name for loading config (e.g., Qwen/Qwen2.5-3B-Instruct)",
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on HuggingFace Hub (e.g., username/model-name)",
    )
    
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Training step number",
    )
    
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Organization name (optional)",
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    
    parser.add_argument(
        "--optimizer-save-mode",
        type=str,
        choices=["full", "none"],
        default=None,
        help="How to save optimizer states: 'full' or 'none' (default: none)",
    )
    
    args = parser.parse_args()
    
    # Convert "none" to None
    optimizer_save_mode = args.optimizer_save_mode if args.optimizer_save_mode != "none" else None
    
    # Validate checkpoint directory exists
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # Check if model shards exist
    model_shards = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_shards:
        print(f"Error: No model checkpoint shards found in {checkpoint_dir}")
        print("Expected files like: model_world_size_4_rank_0.pt")
        sys.exit(1)
    
    print(f"Found {len(model_shards)} model shards")
    
    # Upload
    print("\n" + "="*70)
    print("HuggingFace Hub Upload")
    print("="*70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model: {args.model_name}")
    print(f"Repo: {args.repo_id}")
    print(f"Step: {args.step}")
    print(f"Private: {args.private}")
    print(f"Optimizer: {optimizer_save_mode or 'not saved'}")
    print("="*70 + "\n")
    
    upload_checkpoint_to_hf(
        checkpoint_dir=checkpoint_dir,
        model_name=args.model_name,
        repo_id=args.repo_id,
        step=args.step,
        organization=args.organization,
        private=args.private,
        optimizer_save_mode=optimizer_save_mode,
    )


if __name__ == "__main__":
    main()

