from __future__ import annotations

import logging
import math
from typing import cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import Module
from torch.optim import Optimizer

from ..config.schema import AppCfg, SegTaskCfg
from .sources import build_foundation, freeze_module

log = logging.getLogger(__name__)


class SimpleUpsampleDecoder(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, num_upsamples: int = 2) -> None:
    super().__init__()
    mid_channels = max(64, in_channels // 2)
    blocks: list[nn.Module] = []
    for idx in range(num_upsamples):
      in_ch = in_channels if idx == 0 else mid_channels
      blocks.append(
        nn.Sequential(
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
          nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1),
          nn.GELU(),
        )
      )
    self.blocks = nn.ModuleList(blocks)
    self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      x = block(x)
    return cast(torch.Tensor, self.out_conv(x))


def _has_lora_targets(module: nn.Module, target_modules: list[str]) -> bool:
  for name, _ in module.named_modules():
    for target in target_modules:
      if name == target or name.endswith(f".{target}"):
        return True
  return False


def _apply_lora(cfg: AppCfg, encoder: nn.Module, freeze_base: bool) -> nn.Module:
  lora_cfg = cfg.model.lora
  if not lora_cfg.enabled:
    return encoder

  if freeze_base:
    freeze_module(encoder)

  if not _has_lora_targets(encoder, lora_cfg.target_modules):
    log.warning(
      "LoRA enabled but no target modules matched. "
      "Skipping LoRA wrap; encoder remains frozen."
    )
    return encoder

  peft_cfg = LoraConfig(
    r=lora_cfg.r,
    lora_alpha=lora_cfg.alpha,
    lora_dropout=lora_cfg.dropout,
    bias=lora_cfg.bias,
    target_modules=lora_cfg.target_modules,
    task_type=TaskType.FEATURE_EXTRACTION,
  )
  return cast(nn.Module, get_peft_model(encoder, peft_cfg))


class FoundationSegmentationModel(nn.Module):
  def __init__(self, cfg: AppCfg, num_classes: int) -> None:
    super().__init__()
    foundation = build_foundation(cfg)
    self.backbone = foundation.backbone
    self.feat_dim = foundation.feat_dim
    encoder = getattr(self.backbone, "tile_encoder", self.backbone)
    if cfg.model.lora.enabled:
      encoder = _apply_lora(cfg, encoder, freeze_base=cfg.model.freeze_backbone)
      if getattr(self.backbone, "tile_encoder", None) is not None:
        self.backbone.tile_encoder = encoder
      else:
        self.backbone = encoder
    elif cfg.model.freeze_backbone:
      freeze_module(self.backbone)

    self.decoder = SimpleUpsampleDecoder(self.feat_dim, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    feats = self._extract_spatial_features(x)
    logits = self.decoder(feats)
    if logits.shape[-2:] != x.shape[-2:]:
      logits = F.interpolate(
        logits,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
      )
    return cast(torch.Tensor, logits)

  def _extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
    encoder = getattr(self.backbone, "tile_encoder", self.backbone)
    forward_features = getattr(encoder, "forward_features", None)
    if callable(forward_features):
      feats = cast(torch.Tensor, forward_features(x))
    else:
      feats = cast(torch.Tensor, encoder(x))

    if isinstance(feats, (list, tuple)):
      feats = feats[-1]

    if isinstance(feats, torch.Tensor) and feats.dim() == 4:
      return feats
    if isinstance(feats, torch.Tensor) and feats.dim() == 3:
      return self._tokens_to_map(feats, encoder)

    raise ValueError(
      "Foundation backbone did not return spatial features. "
      "Ensure the model supports forward_features with token outputs."
    )

  def _tokens_to_map(self, tokens: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    grid_h, grid_w = self._resolve_grid_size(tokens, encoder)

    if tokens.shape[1] == grid_h * grid_w + 1:
      tokens = tokens[:, 1:, :]
    elif tokens.shape[1] != grid_h * grid_w:
      raise ValueError(
        f"Token sequence length ({tokens.shape[1]}) does not match grid "
        f"{grid_h}x{grid_w} for segmentation."
      )

    bsz, _, feat_dim = tokens.shape
    return tokens.transpose(1, 2).reshape(bsz, feat_dim, grid_h, grid_w)

  def _resolve_grid_size(self, tokens: torch.Tensor, encoder: nn.Module) -> tuple[int, int]:
    patch_embed = getattr(encoder, "patch_embed", None)
    grid = getattr(patch_embed, "grid_size", None) if patch_embed is not None else None
    if grid is not None:
      return int(grid[0]), int(grid[1])

    num_tokens = tokens.shape[1]
    if num_tokens > 1 and int(math.sqrt(num_tokens - 1)) ** 2 == num_tokens - 1:
      num_tokens = num_tokens - 1

    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
      raise ValueError(
        "Cannot infer spatial grid size from token sequence. "
        "Set a backbone with patch_embed.grid_size."
      )
    return side, side


def build_model(cfg: AppCfg) -> Module:
  task = cfg.dataset.task
  if not isinstance(task, SegTaskCfg):
    raise ValueError(
      f"build_model called with dataset.task.type={task.type!r} (expected segmentation)."
    )

  if task.type == "binary":
    classes = 1
  else:
    # includes background
    classes = len(task.layer_ids) + 1

  if cfg.model.name == "unet":
    model = smp.Unet(
      encoder_name=cfg.model.encoder_name,
      encoder_weights=cfg.model.encoder_weights,
      in_channels=cfg.model.in_channels,
      classes=classes,
      activation=None,
    )
    return cast(Module, model)

  if cfg.model.name in {"timm", "prov_gigapath", "uni2_h", "virchow2"}:
    return cast(Module, FoundationSegmentationModel(cfg, num_classes=classes))

  raise ValueError(f"Unsupported model.name={cfg.model.name}")


def build_loss(cfg: AppCfg) -> smp.losses.DiceLoss:
  if cfg.dataset.task.type == "binary":
    return smp.losses.DiceLoss(mode="binary", from_logits=True)
  return smp.losses.DiceLoss(mode="multiclass", from_logits=True)


def build_optimizer(cfg: AppCfg, model: Module) -> Optimizer:
  # Specifically returns AdamW, which is a type of Optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  return torch.optim.AdamW(params, lr=cfg.train.lr)
