from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config.schema import AppCfg


def get_transforms(cfg: AppCfg) -> tuple[A.Compose, A.Compose]:
  ps = cfg.dataset.data.patch_size
  train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(),
    A.Resize(ps, ps),
    ToTensorV2(),
  ])
  val_tf = A.Compose([
    A.Resize(ps, ps),
    A.Normalize(),
    ToTensorV2(),
  ])
  return train_tf, val_tf
