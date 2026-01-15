from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.nn import Module

from ..config.schema import AppCfg


class MILTopKMean(nn.Module):
  """
  Bag = slide, instances = patches.

  Forward expects x: (N, C, H, W) and returns slide_logits: (1, num_classes).
  This matches your notebook logic (Top-K mean of positive logits).
  """

  def __init__(self, backbone: Module, num_classes: int = 2, top_k: int = 10):
    super().__init__()
    self.top_k = int(top_k)
    self.backbone = backbone  # outputs (N, D)
    feat_dim = getattr(self.backbone, "num_features", None)
    if feat_dim is None:
      raise ValueError("Backbone must expose .num_features (timm models do).")
    self.classifier = nn.Linear(int(feat_dim), int(num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    feats = self.backbone(x)              # (N, D)
    inst_logits = self.classifier(feats)  # (N, C)

    # notebook parity: use positive-class logit only, aggregate top-k then build (1,2)
    if inst_logits.size(1) != 2:
      # If you later extend to >2 classes, you should redesign aggregation.
      raise ValueError(f"MILTopKMean currently assumes num_classes=2, got {inst_logits.size(1)}")

    pos = inst_logits[:, 1]  # (N,)
    k = min(self.top_k, pos.size(0))
    topk_pos, _ = torch.topk(pos, k=k)
    slide_pos = topk_pos.mean()  # scalar

    # Build a 2-logit vector. Keep parity with notebook.
    slide_logits = torch.stack(
      [torch.tensor(0.0, device=x.device, dtype=slide_pos.dtype), slide_pos]
    ).unsqueeze(0)  # (1,2)

    return slide_logits

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

  # Strip typical classifier / head parameters (varies across timm architectures)
  head_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
  filtered = {k: v for k, v in state.items() if not k.startswith(head_prefixes)}

  missing, unexpected = backbone.load_state_dict(filtered, strict=False)

  # This is normal: your backbone has no head, and checkpoints often include head weights.
  print("[MIL CKPT]", "missing:", missing[:8], "unexpected:", unexpected[:8])


def build_model_mil(cfg: AppCfg) -> Module:
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

  # Create a backbone without classifier head; avoid any HF download
  backbone = timm.create_model(
    arch,
    pretrained=False,     # IMPORTANT: no download
    num_classes=0,        # backbone only
    global_pool="avg",
    in_chans=cfg.model.in_channels,
  )

  ckpt = getattr(cfg.model, "init_ckpt", None)
  if ckpt is not None:
    _load_backbone_weights(backbone, Path(ckpt))

  # MIL head size comes from task (not model.num_classes)
  num_classes = getattr(cfg.dataset.task, "num_classes", 2)
  top_k = getattr(cfg.dataset.task, "top_k", 10)

  return MILTopKMean(backbone=backbone, num_classes=num_classes, top_k=top_k)
