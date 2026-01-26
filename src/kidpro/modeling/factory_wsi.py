from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from torch.nn import Module

from ..config.schema import AppCfg
from .lora import apply_lora
from .sources import build_foundation, freeze_module, load_state_dict_generic

log = logging.getLogger(__name__)


def _resolve_longnet_weights_path(cfg: AppCfg) -> Path:
  weights = cfg.model.longnet_weights
  if weights is None:
    raise ValueError("model.longnet_weights is required to preload LongNet.")
  if weights.source == "local":
    return Path(weights.local_path)  # type: ignore[arg-type]
  if weights.source == "hf_cache":
    return Path(weights.hf_cache_path)  # type: ignore[arg-type]
  raise ValueError(f"Unknown longnet_weights.source={weights.source!r}")


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
  feat_dim = foundation.feat_dim
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
  in_chans = int(getattr(cfg.model, "foundation_dim", feat_dim))
  dim = cfg.model.longnet_dim
  aggregator_type = cfg.model.aggregator_type

  from .longnet import LongNetMIL, LongNetViT, SimpleAggregator

  log.info("[MIL Factory] Building model with aggregator_type=%s", aggregator_type)

  slide_encoder: Union[LongNetViT, SimpleAggregator]
  if aggregator_type == "longnet":
    slide_encoder = LongNetViT(
      in_chans=in_chans,
      embed_dim=dim,
      depth=cfg.model.longnet_depth,
      slide_ngrids=cfg.model.longnet_slide_ngrids,
      tile_size=cfg.dataset.data.patch_size,
      max_wsi_size=cfg.model.longnet_max_wsi_size,
      global_pool=False,
      dropout=cfg.model.longnet_dropout,
      input_norm=cfg.model.longnet_input_norm,
      input_dropout=cfg.model.longnet_input_dropout,
    )
    if cfg.model.longnet_pretrained:
      ckpt_path = _resolve_longnet_weights_path(cfg)
      log.info("Loading LongNet weights from %s", ckpt_path)
      slide_encoder = load_state_dict_generic(slide_encoder, ckpt_path)  # type: ignore[assignment]
    if lora_cfg.enabled and "longnet" in apply_to:
      slide_encoder = apply_lora(cfg, slide_encoder, freeze_base=True)
  elif aggregator_type in ("mean_pool", "max_pool"):
    pool_type = "mean" if aggregator_type == "mean_pool" else "max"
    slide_encoder = SimpleAggregator( # type: ignore[assignment]
      in_dim=in_chans,
      embed_dim=dim,
      pool_type=pool_type,
      dropout=cfg.model.longnet_input_dropout,
    )
    log.info("[MIL Factory] Using SimpleAggregator with pool_type=%s", pool_type)
  else:
    raise ValueError(f"Unknown aggregator_type: {aggregator_type}")

  model = LongNetMIL(tile_encoder=tile_encoder, slide_encoder=slide_encoder, num_classes=num_classes)

  return model
