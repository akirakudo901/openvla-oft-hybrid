"""Save and load the hybrid label head next to the OpenVLA-OFT action head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from prismatic.training.finetune_helpers import save_training_checkpoint


def label_head_checkpoint_file(cfg, run_dir: Path, log_step: int) -> Path:
    """Match ``save_training_checkpoint`` directory and ``label_head--*_checkpoint.pt`` naming."""
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = Path(run_dir)
        suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        suffix = f"{log_step}_checkpoint.pt"
    return checkpoint_dir / f"label_head--{suffix}"


def save_label_head(label_head: nn.Module, path: Path) -> None:
    """Write the unwrapped label-head state dict."""
    module = label_head.module if isinstance(label_head, nn.parallel.DistributedDataParallel) else label_head
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def save_hybrid_checkpoint(
    *,
    cfg,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    proprio_projector,
    action_head,
    label_head: nn.Module,
    train_dataset,
    distributed_state,
) -> None:
    """Save the upstream LoRA / action-head checkpoint, then the label head."""
    save_training_checkpoint(
        cfg=cfg,
        run_dir=run_dir,
        log_step=log_step,
        vla=vla,
        processor=processor,
        proprio_projector=proprio_projector,
        noisy_action_projector=None,
        action_head=action_head,
        train_dataset=train_dataset,
        distributed_state=distributed_state,
    )
    if distributed_state.is_main_process:
        path = label_head_checkpoint_file(cfg, run_dir, log_step)
        save_label_head(label_head, path)
        print(f"Saved label head at {path}")
    dist.barrier()
