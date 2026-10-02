"""LoRA + L1 OpenVLA-OFT fine-tune with a BIO label head and MP/L losses.

Setup, optimization, and the training step loop come from
``prismatic.training.finetune_helpers``. This script only adds the label head,
the hybrid loss, and the LeRobot batch source.

Run from the ``openvla-oft`` repo root:

    python -m hybrid.scripts.finetune --lerobot_dataset_root ... --dataset_name ...
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import wandb
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hybrid.datasets.lerobot_batch import (  # noqa: E402
    _dataset_root,
    build_hybrid_lerobot_dataset,
    count_episodes,
    split_episode_ids,
)
from hybrid.datasets.normalize import resolve_norm_mode  # noqa: E402
from hybrid.models.label_head import DEFAULT_NUM_LABEL_CLASSES, ChunkLabelHead  # noqa: E402
from hybrid.training.checkpoint import save_hybrid_checkpoint  # noqa: E402
from hybrid.training.collate import HybridActionCollator  # noqa: E402
from hybrid.training.loss import LABEL_FEATURE_KEY, compute_hybrid_loss  # noqa: E402
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics  # noqa: E402
from prismatic.training.finetune_helpers import (  # noqa: E402
    build_finetune_heads,
    build_finetune_optimizer,
    continuous_action_hidden_states,
    forward_vla_batch,
    init_module,
    load_finetune_vla,
    log_metrics_to_wandb,
    prepare_finetune_run,
    run_finetune_loop,
    wrap_ddp,
)
@dataclass
class HybridFinetuneConfig:
    """L1 LoRA fine-tune plus MP/L action and label losses."""

    # fmt: off
    vla_path: str = "openvla/openvla-7b"
    lerobot_dataset_root: Path = Path("datasets/lerobot")
    dataset_name: str = "libero_spatial"
    run_root_dir: Path = Path("runs")

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    # "quantile" maps [q01, q99] to [-1, 1]. "minmax" maps [min, max] and writes
    # those bounds into q01/q99 so existing LIBERO unnormalization inverts them.
    action_norm: str = "minmax"

    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    use_val_set: bool = False
    val_ratio: float = 0.1
    val_freq: int = 10_000
    val_time_limit: int = 180
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: int | None = None
    image_aug: bool = False
    num_workers: int = 0

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True

    num_label_classes: int = DEFAULT_NUM_LABEL_CLASSES
    mp_l1_weight: float = 1.0
    mp_ce_weight: float = 1.0
    l_ce_weight: float = 1.0

    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: str | None = None
    run_id_override: str | None = None
    wandb_log_freq: int = 10
    # fmt: on


def _build_train_dataset(cfg: HybridFinetuneConfig, processor, episodes=None):
    """Load the LeRobot export and map each window to an OFT batch."""
    return build_hybrid_lerobot_dataset(cfg, processor, episodes=episodes)


def _build_val_dataset(cfg: HybridFinetuneConfig, processor, episodes, train_dataset):
    """Held-out windows normalized with the training-split statistics."""
    return build_hybrid_lerobot_dataset(
        cfg,
        processor,
        episodes=episodes,
        dataset_statistics_override=train_dataset.dataset_statistics,
    )


def run_hybrid_validation(
    vla,
    action_head,
    label_head,
    proprio_projector,
    val_dataloader,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
) -> None:
    """Average hybrid metrics on the held-out loader, then return the VLA to train mode."""
    val_start_time = time.time()
    vla.eval()
    all_val_metrics = []
    with torch.no_grad():
        for batch in val_dataloader:
            _loss, metrics = run_hybrid_forward(
                vla=vla,
                action_head=action_head,
                label_head=label_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                mp_l1_weight=cfg.mp_l1_weight,
                mp_ce_weight=cfg.mp_ce_weight,
                l_ce_weight=cfg.l_ce_weight,
                num_label_classes=cfg.num_label_classes,
            )
            all_val_metrics.append(metrics)
            if time.time() - val_start_time > cfg.val_time_limit:
                break
    avg_val_metrics = {}
    if all_val_metrics:
        for metric_name in all_val_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
            if values:
                avg_val_metrics[metric_name] = sum(values) / len(values)
    avg_val_metrics["val_batches_count"] = len(all_val_metrics)
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)
    vla.train()


def run_hybrid_forward(
    vla,
    action_head,
    label_head,
    proprio_projector,
    batch,
    device_id: int,
    *,
    use_proprio: bool,
    use_film: bool,
    num_patches: int,
    mp_l1_weight: float,
    mp_ce_weight: float,
    l_ce_weight: float,
    num_label_classes: int,
):
    """VLM forward, L1 action head, label head, then the MP/L chunk loss."""
    output = forward_vla_batch(
        vla,
        batch,
        device_id,
        use_proprio=use_proprio,
        use_film=use_film,
        proprio_projector=proprio_projector,
    )
    actions_hidden_states = continuous_action_hidden_states(output, batch, device_id, num_patches)
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    label_dtype = next(label_head.parameters()).dtype
    label_logits = label_head.module(actions_hidden_states.to(dtype=label_dtype))

    actions = batch["actions"].to(device=device_id, dtype=predicted_actions.dtype)
    frame_labels = batch[LABEL_FEATURE_KEY].to(device_id)
    is_pad = batch["is_pad"].to(device_id) if "is_pad" in batch else None
    loss, metrics = compute_hybrid_loss(
        predicted_actions,
        actions,
        label_logits,
        frame_labels,
        is_pad=is_pad,
        mp_l1_weight=mp_l1_weight,
        mp_ce_weight=mp_ce_weight,
        l_ce_weight=l_ce_weight,
        num_label_classes=num_label_classes,
    )
    metrics["loss_value"] = loss.item()
    return loss, metrics


@draccus.wrap()
def hybrid_finetune(cfg: HybridFinetuneConfig) -> None:
    """Fine-tune OpenVLA-OFT with LoRA, an L1 action head, and a BIO label head."""
    resolve_norm_mode(cfg.action_norm)
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert cfg.use_l1_regression and not cfg.use_diffusion, "Hybrid fine-tune is the L1 action head only."

    run_dir, distributed_state, device_id = prepare_finetune_run(
        cfg, banner="Hybrid fine-tuning OpenVLA", wandb_name_prefix="hybrid+"
    )
    processor, vla = load_finetune_vla(cfg, device_id)
    vla = wrap_ddp(vla, device_id, find_unused=True)
    proprio_projector, action_head, _noisy_action_projector, num_patches = build_finetune_heads(
        cfg, vla, device_id
    )
    label_head = init_module(
        ChunkLabelHead,
        "label_head",
        cfg,
        device_id,
        {"input_dim": vla.module.llm_dim, "num_label_classes": cfg.num_label_classes},
        to_bf16=False,
    )
    optimizer, scheduler, original_lr = build_finetune_optimizer(
        cfg, [vla, action_head, label_head, proprio_projector]
    )

    train_episodes = None
    val_episodes = None
    if cfg.use_val_set:
        dataset_root = _dataset_root(cfg.lerobot_dataset_root, cfg.dataset_name)
        train_episodes, val_episodes = split_episode_ids(
            count_episodes(dataset_root), cfg.val_ratio
        )
        print(
            f"Validation split: {len(train_episodes)} train episodes, "
            f"{len(val_episodes)} val episodes"
        )
    train_dataset = _build_train_dataset(cfg, processor, episodes=train_episodes)
    # Same early write as vla-scripts/finetune.py so the run dir has stats before the first checkpoint.
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=HybridActionCollator(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        ),
        num_workers=cfg.num_workers,
    )
    val_dataloader = None
    if cfg.use_val_set:
        val_dataset = _build_val_dataset(cfg, processor, val_episodes, train_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=dataloader.collate_fn,
            num_workers=cfg.num_workers,
        )

    def compute_loss(batch, _batch_idx):
        return run_hybrid_forward(
            vla=vla,
            action_head=action_head,
            label_head=label_head,
            proprio_projector=proprio_projector,
            batch=batch,
            device_id=device_id,
            use_proprio=cfg.use_proprio,
            use_film=cfg.use_film,
            num_patches=num_patches,
            mp_l1_weight=cfg.mp_l1_weight,
            mp_ce_weight=cfg.mp_ce_weight,
            l_ce_weight=cfg.l_ce_weight,
            num_label_classes=cfg.num_label_classes,
        )

    def save_checkpoint(log_step):
        save_hybrid_checkpoint(
            cfg=cfg,
            run_dir=run_dir,
            log_step=log_step,
            vla=vla,
            processor=processor,
            proprio_projector=proprio_projector,
            action_head=action_head,
            label_head=label_head,
            train_dataset=train_dataset,
            distributed_state=distributed_state,
        )

    def validate(log_step):
        run_hybrid_validation(
            vla=vla,
            action_head=action_head,
            label_head=label_head,
            proprio_projector=proprio_projector,
            val_dataloader=val_dataloader,
            device_id=device_id,
            cfg=cfg,
            num_patches=num_patches,
            log_step=log_step,
            distributed_state=distributed_state,
        )

    vla.train()
    run_finetune_loop(
        cfg,
        dataloader,
        distributed_state=distributed_state,
        optimizer=optimizer,
        scheduler=scheduler,
        original_lr=original_lr,
        compute_loss=compute_loss,
        metric_names=(
            "loss_value",
            "mp_l1_loss",
            "l_l1_loss",
            "weighted_l1_loss",
            "mp_ce_loss",
            "l_ce_loss",
            "weighted_label_ce_loss",
            "label_accuracy",
        ),
        save_checkpoint=save_checkpoint,
        validate=validate if cfg.use_val_set else None,
        cycle_loader=True,
    )


if __name__ == "__main__":
    hybrid_finetune()
