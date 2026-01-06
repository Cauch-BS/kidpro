from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch.nn import Module
from torch.optim import Optimizer

from ..config.schema import AppCfg


def build_model(cfg: AppCfg) -> Module:
  if cfg.model.name != "unet":
    raise ValueError(f"Unsupported model.name={cfg.model.name}")

  if cfg.dataset.task.type == "binary":
    classes = 1
  else:
    # includes background
    classes = len(cfg.dataset.task.layer_ids) + 1

  model = smp.Unet(
    encoder_name=cfg.model.encoder_name,
    encoder_weights=cfg.model.encoder_weights,
    in_channels=cfg.model.in_channels,
    classes=classes,
    activation=None,
  )
  return model


def build_loss(cfg: AppCfg) -> smp.losses.DiceLoss:
  if cfg.dataset.task.type == "binary":
    return smp.losses.DiceLoss(mode="binary", from_logits=True)
  return smp.losses.DiceLoss(mode="multiclass", from_logits=True)


def build_optimizer(cfg: AppCfg, model: Module) -> Optimizer:
  # Specifically returns AdamW, which is a type of Optimizer
  return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
