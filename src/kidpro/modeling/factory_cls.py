from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from ..config.schema import AppCfg
from .sources import build_foundation, freeze_module


class _LinearClassifier(nn.Module):
  def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int) -> None:
    super().__init__()
    self.backbone = backbone
    self.classifier = nn.Linear(feat_dim, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = cast(torch.Tensor, self.backbone(x))
    return self.classifier(features) # type: ignore


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

  foundation = build_foundation(cfg)
  if cfg.model.freeze_backbone:
    freeze_module(foundation.backbone)
  return _LinearClassifier(
    backbone=foundation.backbone,
    feat_dim=foundation.feat_dim,
    num_classes=num_classes,
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
