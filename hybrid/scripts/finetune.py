"""LoRA + L1 OpenVLA-OFT fine-tune with a BIO label head and MP/L losses.

Reuses helpers from ``vla-scripts/finetune.py``. Does not change that file.
The LeRobot window adapter (``hybrid.datasets.lerobot_batch``) supplies batches;
this script trains on them.

Run from the ``openvla-oft`` repo root:

    python -m hybrid.scripts.finetune --lerobot_dataset_root ... --dataset_name ...
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.robot.openvla_utils import (  # noqa: E402
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from hybrid.models.label_head import DEFAULT_NUM_LABEL_CLASSES, ChunkLabelHead  # noqa: E402
from hybrid.training.checkpoint import save_hybrid_checkpoint  # noqa: E402
from hybrid.training.collate import HybridActionCollator  # noqa: E402
from hybrid.training.loss import LABEL_FEATURE_KEY, compute_hybrid_loss  # noqa: E402
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.models.projectors import ProprioProjector  # noqa: E402
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_OFT = None


def _oft():
    """Load ``vla-scripts/finetune.py`` (the directory name is not a package)."""
    global _OFT
    if _OFT is None:
        path = _REPO_ROOT / "vla-scripts" / "finetune.py"
        spec = importlib.util.spec_from_file_location("openvla_oft_finetune", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load upstream finetune helpers from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _OFT = module
    return _OFT


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

    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
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


def _build_train_dataset(cfg: HybridFinetuneConfig, processor):
    """Load the LeRobot → OFT dataset from the adapter module (todo 4)."""
    try:
        from hybrid.datasets.lerobot_batch import build_hybrid_lerobot_dataset
    except ImportError as exc:
        raise ImportError(
            "hybrid.datasets.lerobot_batch.build_hybrid_lerobot_dataset is not available yet. "
            "The fine-tune loop expects samples with OFT keys plus "
            f"{LABEL_FEATURE_KEY!r} and optional is_pad."
        ) from exc
    return build_hybrid_lerobot_dataset(cfg, processor)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def run_hybrid_forward(
    vla,
    action_head,
    label_head,
    proprio_projector,
    batch,
    device_id: int,
    *,
    use_proprio: bool,
    num_patches: int,
    mp_l1_weight: float,
    mp_ce_weight: float,
    l_ce_weight: float,
    num_label_classes: int,
):
    """VLM forward, L1 action head, label head, then the MP/L chunk loss."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=False,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
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
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert cfg.use_l1_regression and not cfg.use_diffusion, (
        "Hybrid fine-tune is the L1 action head only."
    )
    assert not cfg.use_film, "Hybrid fine-tune does not wrap the vision backbone with FiLM."

    oft = _oft()
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Hybrid fine-tuning OpenVLA `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = oft.get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"hybrid+{run_id}")

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}"
    )

    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla = get_peft_model(
        vla,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        ),
    )
    vla.print_trainable_parameters()
    vla = oft.wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = oft.init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    action_head = oft.init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )
    label_head = oft.init_module(
        ChunkLabelHead,
        "label_head",
        cfg,
        device_id,
        {"input_dim": vla.module.llm_dim, "num_label_classes": cfg.num_label_classes},
        to_bf16=False,
    )

    num_patches = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        num_patches += 1

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    trainable_params += [param for param in label_head.parameters() if param.requires_grad]
    if proprio_projector is not None:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    train_dataset = _build_train_dataset(cfg, processor)
    collator = HybridActionCollator(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )

    metric_names = (
        "loss_value",
        "mp_l1_loss",
        "l_l1_loss",
        "weighted_l1_loss",
        "mp_ce_loss",
        "l_ce_loss",
        "weighted_label_ce_loss",
        "label_accuracy",
    )
    recent_metrics = {name: deque(maxlen=cfg.grad_accumulation_steps) for name in metric_names}

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(_cycle(dataloader)):
            loss, metrics = run_hybrid_forward(
                vla=vla,
                action_head=action_head,
                label_head=label_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                num_patches=num_patches,
                mp_l1_weight=cfg.mp_l1_weight,
                mp_ce_weight=cfg.mp_ce_weight,
                l_ce_weight=cfg.l_ce_weight,
                num_label_classes=cfg.num_label_classes,
            )
            (loss / cfg.grad_accumulation_steps).backward()
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = oft.compute_smoothened_metrics(recent_metrics)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                oft.log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_hybrid_checkpoint(
                    oft.save_training_checkpoint,
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

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    hybrid_finetune()
