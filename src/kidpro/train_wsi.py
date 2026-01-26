from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import WeightedRandomSampler

from .config.load import CONFIG, CONFIG_EXPORT, resolve_best_model_from_mlflow
from .data.dataset_mil import MILDataset
from .data.split_mil import build_mil_split_csv
from .data.transform import get_transforms
from .modeling.factory_wsi import build_model_mil
from .modeling.sources import load_state_dict_generic
from .training.loop_mil import fit_mil

log = logging.getLogger(__name__)


def _compute_model_hash(model: nn.Module) -> str:
  """Compute hash of model parameters for cache invalidation."""
  hasher = hashlib.md5()
  for p in model.parameters():
    hasher.update(p.data.cpu().numpy().tobytes())
  return hasher.hexdigest()[:16]


@hydra.main(version_base=None, config_path="conf", config_name="config_wsi")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)

  if not cfg.model.freeze_backbone:
    raise ValueError("train_wsi requires model.freeze_backbone=true.")

  CONFIG_EXPORT(cfg, rr)

  # Build slide-level CSV (SlideName / GT / split)
  df = build_mil_split_csv(cfg)

  df_tr = df[df["split"] == "train"].reset_index(drop=True)
  df_va = df[df["split"] == "val"].reset_index(drop=True)

  # -------------------------
  # Sanity check mode: subset data for quick iteration
  # -------------------------
  if cfg.train.sanity_check:
    df_tr = df_tr.head(cfg.train.sanity_check_samples)
    df_va = df_va.head(cfg.train.sanity_check_samples)
    log.info("[SANITY CHECK] Running on %d train / %d val samples for 1 epoch", len(df_tr), len(df_va))

  train_tf, val_tf = get_transforms(cfg)

  ds_tr = MILDataset(cfg, df_tr, transform=train_tf)
  ds_va = MILDataset(cfg, df_va, transform=val_tf)

  def _mil_collate(batch: list[tuple[object, torch.Tensor, str]]) -> tuple[object, torch.Tensor, str]:
    if len(batch) != 1:
      raise ValueError("MIL collate expects batch_size=1.")
    return batch[0]

  # -------------------------
  # Balanced sampling (optional)
  # -------------------------
  sampler = None
  if cfg.train.use_balanced_sampling:
    class_counts = df_tr["GT"].value_counts()
    sample_weights = [1.0 / class_counts.loc[int(gt)] for gt in df_tr["GT"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(df_tr), replacement=True)
    log.info("[BALANCED SAMPLING] Class counts: %s", class_counts.to_dict())

  # NOTE:
  # Using BOTH a balanced sampler and class-weighted loss "double counts" imbalance handling:
  # - The sampler changes the *data distribution* the optimizer sees (expected gradient)
  # - The loss weights scale per-example gradients
  # If the sampler already makes classes ~uniform, additional class weights will generally
  # overweight the minority class and can bias training toward predicting that class.
  effective_use_class_weights = bool(cfg.train.use_class_weights) and (sampler is None)
  if cfg.train.use_balanced_sampling and cfg.train.use_class_weights:
    log.warning(
      "[IMBALANCE HANDLING] Both use_balanced_sampling=true and use_class_weights=true. "
      "Disabling class weights because a balanced sampler is already in use."
    )

  dl_tr = torch.utils.data.DataLoader(
    ds_tr,
    batch_size=1,
    shuffle=(sampler is None),  # Only shuffle if no sampler
    sampler=sampler,
    num_workers=cfg.dataset.data.num_workers,
    pin_memory=cfg.dataset.data.pin_memory,
    persistent_workers=(cfg.dataset.data.num_workers > 0),
    prefetch_factor=4 if cfg.dataset.data.num_workers > 0 else None,
    collate_fn=_mil_collate,
  )
  dl_va = torch.utils.data.DataLoader(
    ds_va,
    batch_size=1,
    shuffle=False,
    num_workers=cfg.dataset.data.num_workers,
    pin_memory=cfg.dataset.data.pin_memory,
    persistent_workers=(cfg.dataset.data.num_workers > 0),
    prefetch_factor=4 if cfg.dataset.data.num_workers > 0 else None,
    collate_fn=_mil_collate,
  )

  model = build_model_mil(cfg).to(rr.device)
  if cfg.model.lora.enabled:
    try:
      ckpt_path = resolve_best_model_from_mlflow(cfg, "tile_model")
      model.tile_encoder = load_state_dict_generic(cast(nn.Module, model.tile_encoder), ckpt_path)
      log.info(f"[LORA INIT] Loaded tile checkpoint: {ckpt_path}")
    except Exception as e:
      raise RuntimeError(
        "Failed to resolve or load tile-trained best_model.pt for LoRA initialization."
      ) from e

  # -------------------------
  # Compute tile_encoder hash for cache invalidation
  # -------------------------
  tile_encoder_hash = _compute_model_hash(model.tile_encoder)
  ds_tr.set_tile_encoder_hash(tile_encoder_hash)
  ds_va.set_tile_encoder_hash(tile_encoder_hash)
  log.info("[TILE ENCODER HASH] %s", tile_encoder_hash)

  # -------------------------
  # Class-weighted loss (optional)
  # -------------------------
  if effective_use_class_weights:
    num_classes = int(getattr(cfg.dataset.task, "num_classes", 2))
    class_counts = df_tr["GT"].value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    total = len(df_tr)
    counts = class_counts.to_numpy()
    # Avoid division by zero if a class is missing in training split
    counts_safe = counts.copy()
    counts_safe[counts_safe == 0] = 1
    weights = torch.tensor(total / (num_classes * counts_safe), dtype=torch.float32)
    log.info("[CLASS WEIGHTS] Class counts: %s, weights: %s", class_counts.to_dict(), weights.tolist())
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(rr.device))
  else:
    criterion = torch.nn.CrossEntropyLoss()

  # -------------------------
  # Parameter groups with different learning rates
  # -------------------------
  slide_encoder_params = list(model.slide_encoder.parameters())
  classifier_params = list(model.classifier.parameters())

  param_groups = [
    {"params": slide_encoder_params, "lr": cfg.train.lr, "name": "slide_encoder"},
    {"params": classifier_params, "lr": cfg.train.lr * cfg.train.head_lr_multiplier, "name": "classifier"},
  ]
  optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

  log.info(
    "[PARAM GROUPS] slide_encoder: %d params @ lr=%.2e | classifier: %d params @ lr=%.2e",
    sum(p.numel() for p in slide_encoder_params),
    cfg.train.lr,
    sum(p.numel() for p in classifier_params),
    cfg.train.lr * cfg.train.head_lr_multiplier,
  )

  # -------------------------
  # Learning rate scheduler with warmup
  # -------------------------
  scheduler = None
  if cfg.train.scheduler_type != "none":
    # For sanity check, override epochs to 1
    epochs = 1 if cfg.train.sanity_check else cfg.train.epochs
    total_steps = epochs * len(dl_tr)
    warmup_steps = cfg.train.warmup_epochs * len(dl_tr)

    if warmup_steps > 0 and warmup_steps < total_steps:
      warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
      if cfg.train.scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
      else:
        # Step scheduler not implemented, fallback to cosine
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
      scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])
      log.info("[SCHEDULER] Warmup %d steps, then %s decay for %d steps", warmup_steps, cfg.train.scheduler_type, total_steps - warmup_steps)
    elif cfg.train.scheduler_type == "cosine":
      scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
      log.info("[SCHEDULER] Cosine annealing for %d steps (no warmup)", total_steps)

  best_path = fit_mil(cfg, rr, model, dl_tr, dl_va, criterion, optimizer, scheduler=scheduler)
  log.info(f"[RUN COMPLETE] run_dir={run_dir} best={best_path}")


if __name__ == "__main__":
  main()
