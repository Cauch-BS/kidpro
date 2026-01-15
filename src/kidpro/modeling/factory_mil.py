from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.nn import Module

from ..config.schema import AppCfg


def _load_backbone_weights(backbone: nn.Module, ckpt_path: Path) -> None:
  """
  Loads a timm-compatible checkpoint into a backbone.

  Works with:
  - .safetensors (recommended)
  - .pt/.pth (state_dict)

  Filters out common head keys so you can load imagenet weights even if your backbone
  has no head (num_classes=0).
  """
  ckpt_path = Path(ckpt_path)
  if not ckpt_path.exists():
    raise FileNotFoundError(f"init_ckpt not found: {ckpt_path}")

  state: dict[str, torch.Tensor]
  if ckpt_path.suffix == ".safetensors":
    try:
      from safetensors.torch import load_file
    except Exception as e:
      raise RuntimeError(
        "safetensors is required to load .safetensors checkpoints. "
        "Install with: pip install safetensors"
      ) from e
    state = load_file(str(ckpt_path))
  else:
    obj = torch.load(str(ckpt_path), map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

  # Strip typical classifier / head parameters
  head_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
  filtered = {k: v for k, v in state.items() if not k.startswith(head_prefixes)}

  missing, unexpected = backbone.load_state_dict(filtered, strict=False)
  print("[MIL CKPT]", "missing:", missing[:8], "unexpected:", unexpected[:8])


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
  if cfg.model.name != "timm":
    raise ValueError("MIL currently requires model.name='timm' (timm backbone).")

  arch = cfg.model.arch
  if not arch:
    raise ValueError("model.arch is required for MIL (e.g., resnet50).")

  # Create backbone without classifier head
  backbone = timm.create_model(
    arch,
    pretrained=False,     # IMPORTANT: no download
    num_classes=0,        # backbone only
    global_pool="avg",
    in_chans=cfg.model.in_channels,
  )

  # Load weights if provided
  ckpt = getattr(cfg.model, "init_ckpt", None)
  if ckpt is not None:
    _load_backbone_weights(backbone, Path(ckpt))

  # Get feature dimension
  feat_dim = int(backbone.num_features)  # type: ignore

  # MIL head configuration
  num_classes = getattr(cfg.dataset.task, "num_classes", 2)

  # Import attention models
  from .attention import MultiHeadFlashAttentionMIL

  # Build attention MIL model
  model = MultiHeadFlashAttentionMIL(
    backbone=backbone,
    feat_dim=feat_dim,
    num_classes=num_classes,
    num_heads=4,
    dropout=0.25,
  )

  return model
