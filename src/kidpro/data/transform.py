from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from ..config.schema import AppCfg, SegTaskCfg


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
      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2(),
    ])
    val_tf = A.Compose([
      A.Resize(resize_size, resize_size, interpolation=cv2.INTER_CUBIC),
      A.CenterCrop(input_size, input_size),
      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      ToTensorV2(),
    ])
  else:
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
