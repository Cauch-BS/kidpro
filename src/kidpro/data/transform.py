from __future__ import annotations

import logging

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from ..config.schema import AppCfg, SegTaskCfg

log = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(cfg: AppCfg) -> tuple[A.Compose, A.Compose]:
  ps = cfg.dataset.data.patch_size
  task = cfg.dataset.task
  input_size = cfg.model.input_size or ps
  resize_size = input_size
  if input_size > 0 and input_size != ps:
    resize_size = int(round(input_size / 0.875))

  if isinstance(task, SegTaskCfg) and input_size > 0:
    train_tf = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.Resize(resize_size, resize_size, interpolation=cv2.INTER_CUBIC),
      A.CenterCrop(input_size, input_size),
      A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
      ToTensorV2(),
    ])
    val_tf = A.Compose([
      A.Resize(resize_size, resize_size, interpolation=cv2.INTER_CUBIC),
      A.CenterCrop(input_size, input_size),
      A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
      ToTensorV2(),
    ])
  else:
    crop_size = input_size if input_size > 0 and input_size != ps else None

    # Most foundation backbones used for MIL (e.g., Prov-GigaPath via timm) use
    # ImageNet mean/std normalization in their published preprocessing.
    # (Prov-GigaPath model card example.)
    use_imagenet_norm = cfg.model.name in {"timm", "prov_gigapath", "uni2_h", "virchow2"}
    if use_imagenet_norm:
      log.info("[TRANSFORMS] Using ImageNet normalization for model.name=%s", cfg.model.name)

    # If tile-encoder embeddings are cached, MIL training must see deterministic tiles;
    # otherwise the cached embeddings would correspond to one random augmentation sample.
    is_mil = getattr(task, "type", None) == "mil"
    mil_cache = getattr(getattr(cfg.dataset, "data", None), "mil_cache", None)
    emb_cache_enabled = bool(
      is_mil
      and mil_cache is not None
      and getattr(mil_cache, "enabled", False)
      and (
        getattr(mil_cache, "cache_tile_embeddings", False)
        or getattr(mil_cache, "cache_pooled_embeddings", False)  # backwards compat
      )
    )
    flip_p = 0.0 if emb_cache_enabled else 0.5
    if is_mil and emb_cache_enabled:
      log.info("[TRANSFORMS] MIL embedding cache enabled; disabling random flips for determinism.")
    elif is_mil:
      log.info("[TRANSFORMS] MIL embedding cache disabled; random flips enabled (p=0.5).")

    train_tf = A.Compose([
      A.HorizontalFlip(p=flip_p),
      A.VerticalFlip(p=flip_p),
      A.Resize(ps, ps, interpolation=cv2.INTER_CUBIC),
      A.CenterCrop(crop_size, crop_size) if crop_size else A.NoOp(),
      A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD) if use_imagenet_norm else A.Normalize(),
      ToTensorV2(),
    ])
    val_tf = A.Compose([
      A.Resize(ps, ps, interpolation=cv2.INTER_CUBIC),
      A.CenterCrop(crop_size, crop_size) if crop_size else A.NoOp(),
      A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD) if use_imagenet_norm else A.Normalize(),
      ToTensorV2(),
    ])
  return train_tf, val_tf
