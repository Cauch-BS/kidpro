# src/kidpro/modeling/factory_cls.py
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from ..config.schema import AppCfg


def _resolve_num_classes(cfg: AppCfg) -> int:
  """
  Single source-of-truth resolution for classification head size.

  Priority:
    1) cfg.model.num_classes if explicitly set (> 1)
    2) cfg.dataset.task.num_classes if present (> 1)
  """
  # Model config first (most explicit for modeling)
  mc = getattr(cfg.model, "num_classes", None)
  if isinstance(mc, int) and mc > 1:
    return mc

  # Fall back to task config (classification-specific)
  tc = getattr(cfg.dataset.task, "num_classes", None)
  if isinstance(tc, int) and tc > 1:
    return tc

  raise ValueError(
    "Classification requires num_classes > 1. "
    "Set model.num_classes (preferred) or dataset.task.num_classes."
  )

def build_model_cls(cfg: AppCfg) -> Module:
  if cfg.dataset.task.type != "classification":
    raise ValueError(
      f"build_model_cls called with dataset.task.type={cfg.dataset.task.type!r} "
      "(expected 'classification')"
    )

  # With your schema.py, classification implies model.name == "timm"
  if cfg.model.name != "timm":
    raise ValueError("Classification requires model.name='timm'.")

  # schema.py guarantees arch exists when model.name == 'timm'
  arch = cfg.model.arch
  assert arch is not None  # for type-checkers; schema enforces this

  # schema cross-field validation ensures consistency if model.num_classes is set
  num_classes = cfg.model.num_classes or cfg.dataset.task.num_classes

  return timm.create_model(
    arch,
    pretrained=cfg.model.pretrained,
    num_classes=num_classes,
    in_chans=cfg.model.in_channels,
  )


def build_loss_cls(cfg: AppCfg) -> nn.Module:
  # CrossEntropyLoss expects class indices (0..C-1)
  if cfg.dataset.task.type != "classification":
    raise ValueError(
      f"build_loss_cls called with dataset.task.type={cfg.dataset.task.type!r} "
      "(expected 'classification')"
    )
  return nn.CrossEntropyLoss()


def build_optimizer_cls(cfg: AppCfg, model: Module) -> Optimizer:
  return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
