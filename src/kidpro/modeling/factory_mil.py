from __future__ import annotations

from torch.nn import Module

from ..config.schema import AppCfg
from .sources import build_foundation, freeze_module


def build_model_mil(cfg: AppCfg) -> Module:
  """
  Build MIL model with LongNet head.
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

  from .longnet import LongNetMIL, LongNetViT

  tile_encoder = getattr(backbone, "tile_encoder", backbone)
  dim = cfg.model.longnet_dim
  slide_encoder = LongNetViT(
    in_chans=int(getattr(cfg.model, "foundation_dim", feat_dim)),
    embed_dim=dim,
    depth=cfg.model.longnet_depth,
    slide_ngrids=cfg.model.longnet_slide_ngrids,
    tile_size=cfg.dataset.data.patch_size,
    max_wsi_size=cfg.model.longnet_max_wsi_size,
    global_pool=False,
    dropout=cfg.model.longnet_dropout,
  )
  model = LongNetMIL(tile_encoder=tile_encoder, slide_encoder=slide_encoder, num_classes=num_classes)

  return model
