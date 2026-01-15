from __future__ import annotations

from torch.nn import Module

from ..config.schema import AppCfg
from .sources import build_foundation, freeze_module


def build_model_mil(cfg: AppCfg) -> Module:
  """
  Build MIL model with attention mechanism.

  Returns GatedAttentionMIL by default (can be configured).
  """
  if cfg.dataset.task.type != "mil":
    raise ValueError(
      f"build_model_mil called with dataset.task.type={cfg.dataset.task.type!r} "
      "(expected 'mil')"
    )
  foundation = build_foundation(cfg)
  if cfg.model.freeze_backbone:
    freeze_module(foundation.backbone)
  backbone = foundation.backbone
  feat_dim = foundation.feat_dim

  # MIL head configuration
  num_classes = getattr(cfg.dataset.task, "num_classes", 2)

  # Import attention models
  from .attention import MultiHeadFlashAttentionMIL

  # Build attention MIL model
  model = MultiHeadFlashAttentionMIL(
    backbone=backbone,
    feat_dim=feat_dim,
    num_classes=num_classes,
    num_heads=cfg.model.num_heads,
    dropout=cfg.model.attn_dropout,
  )

  return model
