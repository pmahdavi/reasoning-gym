#!/usr/bin/env python
# encoding: utf-8
"""
HuggingFace Hub uploader for reasoning-gym models.

Supports uploading model weights and optimizer states from FSDP checkpoints
to HuggingFace Hub for easy sharing and reproducibility.
"""

import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import HfApi
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class HuggingFaceUploader:
    """
    A utility class to upload model checkpoints and optimizer states to HuggingFace Hub.
    
    Supports two modes:
    1. Direct upload from FSDP model (if model/optimizer are accessible)
    2. Loading from veRL sharded checkpoint directory
    """

    def __init__(
        self,
        repo_id: str,
        organization: Optional[str] = None,
        private: bool = False,
        optimizer_save_mode: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace uploader.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., "username/model-name")
            organization: Optional organization name (if None, uses user namespace)
            private: Whether to create a private repository
            optimizer_save_mode: How to save optimizer states:
                - "full": Gather all sharded optimizer states to rank 0
                - "sharded": Keep optimizer states sharded (not implemented yet)
                - None: Don't save optimizer states
        """
        self.repo_id = repo_id
        self.organization = organization
        self.private = private
        self.optimizer_save_mode = optimizer_save_mode
        self.api = HfApi()
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    def upload_from_checkpoint_dir(
        self,
        checkpoint_dir: str | Path,
        model_name: str,
        step: int,
    ) -> None:
        """
        Upload model and optimizer states from a veRL checkpoint directory.
        
        This method loads from veRL's sharded checkpoint format:
        - model_world_size_{N}_rank_{i}.pt
        - optim_world_size_{N}_rank_{i}.pt
        
        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/.../global_step_100/actor)
            model_name: HuggingFace model name for loading config
            step: Training step number
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        if self._rank == 0:
            print(f"[HF Upload] Loading checkpoint from {checkpoint_dir}")
        
        # Create temporary directory for consolidated checkpoint
        temp_dir = Path(f"/tmp/hf_upload_{self.repo_id.replace('/', '_')}_{step}")
        if self._rank == 0:
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        try:
            # Gather model weights
            model_state_dict = self._gather_model_from_shards(checkpoint_dir, self._world_size)
            
            # Save model on rank 0
            if self._rank == 0:
                from transformers import AutoConfig, AutoModelForCausalLM
                
                print(f"[HF Upload] Loading model config from {model_name}")
                config = AutoConfig.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_config(config)
                model.load_state_dict(model_state_dict)
                
                print(f"[HF Upload] Saving model to {temp_dir}")
                model.save_pretrained(temp_dir, max_shard_size="10GB")
                
                # Save tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(temp_dir)
                
                del model, config  # Free memory
            
            # Gather and save optimizer states if requested
            if self.optimizer_save_mode:
                self._gather_and_save_optimizer_from_shards(
                    checkpoint_dir, 
                    temp_dir, 
                    self._world_size
                )
            
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # Upload to HuggingFace Hub (rank 0 only)
            if self._rank == 0:
                self._upload_to_hub(temp_dir, step)
        
        finally:
            # Cleanup
            if self._rank == 0 and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
    
    def upload_from_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
        optimizers: Optional[List[Optimizer]] = None,
    ) -> None:
        """
        Upload model and optimizer directly from FSDP model and optimizer objects.
        
        This uses PyTorch's modern get_state_dict() API for efficient gathering.
        
        Args:
            model: FSDP-wrapped model
            tokenizer: Model tokenizer
            step: Training step number
            optimizers: List of optimizers (optional)
        """
        repo_id = f"{self.organization}/{self.repo_id}" if self.organization else self.repo_id
        
        if self._rank == 0:
            print(f"[HF Upload] Uploading model to {repo_id}...")
        
        # Create temporary directory
        temp_dir = Path(f"/tmp/hf_upload_{self.repo_id.replace('/', '_')}_{step}")
        if self._rank == 0:
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Gather model weights
            self._gather_model_weights(model, temp_dir)
            
            # Save model config and tokenizer on rank 0
            if self._rank == 0:
                model.config.save_pretrained(temp_dir)
                if hasattr(model, 'generation_config') and model.generation_config:
                    model.generation_config.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)
            
            # Gather and save optimizer states if requested
            if optimizers and self.optimizer_save_mode:
                self._gather_optimizer_states(model, optimizers, temp_dir)
            
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # Upload to HuggingFace Hub (rank 0 only)
            if self._rank == 0:
                self._upload_to_hub(temp_dir, step)
        
        except Exception as e:
            if self._rank == 0:
                print(f"[HF Upload] Failed to upload: {e}")
            raise
        
        finally:
            # Cleanup
            if self._rank == 0 and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
    
    def _gather_model_from_shards(
        self, 
        checkpoint_dir: Path, 
        world_size: int
    ) -> dict[str, torch.Tensor]:
        """
        Gather model weights from veRL's sharded checkpoint files.
        
        Args:
            checkpoint_dir: Directory containing sharded checkpoints
            world_size: Number of ranks the checkpoint was saved with
            
        Returns:
            Consolidated model state dict
        """
        state_dict = defaultdict(list)
        
        if self._rank == 0:
            print(f"[HF Upload] Gathering model from {world_size} shards")
        
        for rank in range(world_size):
            filepath = checkpoint_dir / f'model_world_size_{world_size}_rank_{rank}.pt'
            if not filepath.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
            if self._rank == 0:
                print(f"[HF Upload] Loading shard {rank}/{world_size} from {filepath}")
                shard_state_dict = torch.load(filepath, map_location='cpu')
                for key, value in shard_state_dict.items():
                    # Handle DTensor if present
                    if isinstance(value, DTensor):
                        value = value.to_local()
                    state_dict[key].append(value)
        
        # Concatenate shards on rank 0
        if self._rank == 0:
            consolidated_state_dict = {}
            for key in state_dict:
                consolidated_state_dict[key] = torch.cat(state_dict[key], dim=0)
            return consolidated_state_dict
        else:
            return {}
    
    def _gather_optimizer_from_shards(
        self,
        checkpoint_dir: Path,
        world_size: int,
    ) -> dict:
        """
        Gather optimizer states from veRL's sharded checkpoint files.
        
        Args:
            checkpoint_dir: Directory containing sharded checkpoints
            world_size: Number of ranks the checkpoint was saved with
            
        Returns:
            Consolidated optimizer state dict
        """
        if self._rank == 0:
            print(f"[HF Upload] Gathering optimizer states from {world_size} shards")
        
        # Only rank 0 gathers
        if self._rank != 0:
            return {}
        
        # Load all optimizer shards
        optimizer_shards = []
        for rank in range(world_size):
            filepath = checkpoint_dir / f'optim_world_size_{world_size}_rank_{rank}.pt'
            if not filepath.exists():
                print(f"[HF Upload] Warning: Optimizer shard {rank} not found at {filepath}")
                return {}
            
            print(f"[HF Upload] Loading optimizer shard {rank}/{world_size}")
            shard = torch.load(filepath, map_location='cpu')
            optimizer_shards.append(shard)
        
        # Consolidate optimizer states
        # veRL's format: {'state': {param_id: {state_name: tensor}}, 'param_groups': [...]}
        consolidated_optim = {
            'state': {},
            'param_groups': []
        }
        
        # Merge param_groups (they should be identical across ranks, just take from first shard)
        if optimizer_shards:
            consolidated_optim['param_groups'] = optimizer_shards[0].get('param_groups', [])
        
        # Merge optimizer states
        # For FSDP sharded optimizers, each rank has different parameter IDs
        # We need to consolidate them into a single state dict
        all_param_ids = set()
        for shard in optimizer_shards:
            if 'state' in shard:
                all_param_ids.update(shard['state'].keys())
        
        for param_id in all_param_ids:
            param_states = []
            for shard in optimizer_shards:
                if 'state' in shard and param_id in shard['state']:
                    param_states.append(shard['state'][param_id])
            
            if param_states:
                # Consolidate state tensors (e.g., exp_avg, exp_avg_sq for Adam)
                consolidated_state = {}
                state_keys = param_states[0].keys()
                
                for state_key in state_keys:
                    tensors = [state[state_key] for state in param_states if state_key in state]
                    if tensors and isinstance(tensors[0], torch.Tensor):
                        # Concatenate tensor states
                        if len(tensors) > 1:
                            consolidated_state[state_key] = torch.cat(tensors, dim=0)
                        else:
                            consolidated_state[state_key] = tensors[0]
                    else:
                        # Keep non-tensor states as-is
                        consolidated_state[state_key] = param_states[0][state_key]
                
                consolidated_optim['state'][param_id] = consolidated_state
        
        return consolidated_optim
    
    def _gather_and_save_optimizer_from_shards(
        self,
        checkpoint_dir: Path,
        temp_dir: Path,
        world_size: int,
    ) -> None:
        """Gather and save optimizer states from sharded checkpoints."""
        optimizer_state_dict = self._gather_optimizer_from_shards(checkpoint_dir, world_size)
        
        if self._rank == 0 and optimizer_state_dict:
            optimizer_path = temp_dir / "optimizer.pt"
            print(f"[HF Upload] Saving consolidated optimizer state to {optimizer_path}")
            torch.save(optimizer_state_dict, optimizer_path)
    
    def _gather_model_weights(
        self,
        model: PreTrainedModel,
        temp_dir: Path,
    ) -> None:
        """
        Gather model weights from FSDP model using state_dict.
        
        Args:
            model: FSDP-wrapped model
            temp_dir: Temporary directory to save weights
        """
        if self._rank == 0:
            print("[HF Upload] Gathering sharded model weights")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
            
            cpu_state = {}
            for key, value in model.state_dict().items():
                if isinstance(value, DTensor):
                    # Convert DTensor to regular tensor by gathering all shards
                    value = value.to(torch.bfloat16)
                    value = value.full_tensor()
                
                if self._rank == 0:
                    # Clean up the key to remove FSDP prefixes
                    clean_key = key.replace("_fsdp_wrapped_module.", "")
                    cpu_state[clean_key] = value.to("cpu", non_blocking=False)
            
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            # Save on rank 0 only
            if self._rank == 0:
                model_path = temp_dir / "pytorch_model.bin"
                print(f"[HF Upload] Saving model weights to {model_path}")
                torch.save(cpu_state, model_path)
    
    def _gather_optimizer_states(
        self,
        model: PreTrainedModel,
        optimizers: List[Optimizer],
        temp_dir: Path,
    ) -> None:
        """
        Gather optimizer states using PyTorch's modern get_state_dict API.
        
        Args:
            model: FSDP-wrapped model
            optimizers: List of optimizers
            temp_dir: Temporary directory to save optimizer states
        """
        if self.optimizer_save_mode == "full":
            self._gather_optimizer_full(model, optimizers, temp_dir)
        else:
            if self._rank == 0:
                print(f"[HF Upload] Unknown optimizer_save_mode: {self.optimizer_save_mode}")
    
    def _gather_optimizer_full(
        self,
        model: PreTrainedModel,
        optimizers: List[Optimizer],
        temp_dir: Path,
    ) -> None:
        """
        Gather optimizer states using FSDP's full state dict approach.
        
        Args:
            model: FSDP-wrapped model
            optimizers: List of optimizers
            temp_dir: Temporary directory to save optimizer states
        """
        try:
            if self._rank == 0:
                print("[HF Upload] Gathering optimizer states using full state dict")
            
            # Try using modern API first
            try:
                from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
                
                options = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                )
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
                    
                    _, optimizer_state_dict = get_state_dict(
                        model,
                        optimizers,
                        options=options,
                    )
                
                if self._rank == 0:
                    optimizer_path = temp_dir / "optimizer.pt"
                    print(f"[HF Upload] Saving optimizer state to {optimizer_path}")
                    torch.save(optimizer_state_dict, optimizer_path)
                
            except ImportError:
                # Fallback to older FSDP API
                if self._rank == 0:
                    print("[HF Upload] Modern API not available, using FSDP.optim_state_dict()")
                
                from torch.distributed.fsdp import FullOptimStateDictConfig, StateDictType
                
                optim_state_dict_config = FullOptimStateDictConfig(
                    offload_to_cpu=True,
                    rank0_only=True,
                )
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
                    
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        optim_state_dict_config=optim_state_dict_config,
                    ):
                        optimizer_state_dict = FSDP.optim_state_dict(model, optimizers[0])
                
                if self._rank == 0 and optimizer_state_dict:
                    optimizer_path = temp_dir / "optimizer.pt"
                    print(f"[HF Upload] Saving optimizer state to {optimizer_path}")
                    torch.save(optimizer_state_dict, optimizer_path)
            
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                
        except Exception as e:
            if self._rank == 0:
                print(f"[HF Upload] Failed to gather optimizer states: {e}")
    
    def _upload_to_hub(self, temp_dir: Path, step: int) -> None:
        """
        Upload the contents of temp_dir to HuggingFace Hub.
        
        Args:
            temp_dir: Directory containing model, tokenizer, and optimizer files
            step: Training step number
        """
        if self._rank != 0:
            return
        
        repo_id = f"{self.organization}/{self.repo_id}" if self.organization else self.repo_id
        
        try:
            print(f"[HF Upload] Creating repository {repo_id}")
            self.api.create_repo(
                repo_id=repo_id,
                private=self.private,
                exist_ok=True,
            )
            
            print(f"[HF Upload] Uploading folder {temp_dir} to {repo_id}")
            self.api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=repo_id,
                commit_message=f"Upload model and optimizer from step {step}",
            )
            
            repo_url = f"https://huggingface.co/{repo_id}"
            print(f"[HF Upload] ✅ Successfully uploaded to {repo_url}")
            
        except Exception as e:
            print(f"[HF Upload] ❌ Failed to upload: {e}")
            raise


def upload_checkpoint_to_hf(
    checkpoint_dir: str | Path,
    model_name: str,
    repo_id: str,
    step: int,
    organization: Optional[str] = None,
    private: bool = False,
    optimizer_save_mode: Optional[str] = None,
) -> None:
    """
    Convenience function to upload a checkpoint to HuggingFace Hub.
    
    Args:
        checkpoint_dir: Path to veRL checkpoint directory
        model_name: HuggingFace model name for loading config
        repo_id: Repository ID on HuggingFace Hub
        step: Training step number
        organization: Optional organization name
        private: Whether to create a private repository
        optimizer_save_mode: How to save optimizer states ("full", "sharded", or None)
    
    Example:
        >>> upload_checkpoint_to_hf(
        ...     checkpoint_dir="checkpoints/my-exp/global_step_500/actor",
        ...     model_name="Qwen/Qwen2.5-3B-Instruct",
        ...     repo_id="username/my-trained-model",
        ...     step=500,
        ...     optimizer_save_mode="full",
        ... )
    """
    uploader = HuggingFaceUploader(
        repo_id=repo_id,
        organization=organization,
        private=private,
        optimizer_save_mode=optimizer_save_mode,
    )
    
    uploader.upload_from_checkpoint_dir(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        step=step,
    )

