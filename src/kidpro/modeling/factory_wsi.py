from __future__ import annotations

import logging

from torch.nn import Module

from ..config.schema import AppCfg
from .lora import apply_lora
from .sources import build_foundation, freeze_module

log = logging.getLogger(__name__)


def build_model_mil(cfg: AppCfg) -> Module:
  """
  Build MIL model with configurable slide aggregator.

  Supports aggregator_type:
    - "longnet": Full LongNet encoder with positional embeddings
    - "mean_pool": Simple mean pooling baseline
    - "max_pool": Simple max pooling baseline
  """
  if cfg.dataset.task.type != "mil":
    raise ValueError(
      f"build_model_mil called with dataset.task.type={cfg.dataset.task.type!r} "
      "(expected 'mil')"
    )
  foundation = build_foundation(cfg)
  backbone = foundation.backbone
  tile_encoder = getattr(backbone, "tile_encoder", backbone)
  lora_cfg = cfg.model.lora
  apply_to = set(lora_cfg.apply_to)

  if lora_cfg.enabled and "backbone" in apply_to:
    # MIL should not fine-tune the tiling model, even with LoRA enabled.
    # fine-tuning the tiling model should only occur during tile segmentation training.
    tile_encoder = apply_lora(cfg, tile_encoder, freeze_base=True)
    freeze_module(tile_encoder)
    if getattr(backbone, "tile_encoder", None) is not None:
      backbone.tile_encoder = tile_encoder
    else:
      backbone = tile_encoder
  elif cfg.model.freeze_backbone:
    freeze_module(backbone)

  # MIL head configuration
  num_classes = getattr(cfg.dataset.task, "num_classes", 2)
  aggregator_type = cfg.model.aggregator_type

  from .agg import build_mil_model

  log.info("[MIL Factory] Building model with aggregator_type=%s", aggregator_type)

  model = build_mil_model(cfg, tile_encoder=tile_encoder, num_classes=num_classes)

  return model
