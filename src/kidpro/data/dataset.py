from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd  # <--- Imported pandas
import torch
from torch.utils.data import Dataset

from ..config.schema import AppCfg


class SegDataset(Dataset):
  def __init__(self, cfg: AppCfg, df: pd.DataFrame, transform: Optional[Any] = None):
    self.cfg = cfg
    self.df = df.reset_index(drop=True)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    row = self.df.iloc[idx]

    task = self.cfg.dataset.task
    if task.type == "mil":
      raise ValueError("SegDataset requires a segmentation task configuration.")

    img = cv2.imread(row["path"])
    if img is None:
      raise FileNotFoundError(f"Image not found: {row['path']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    slide_dir = Path(row["path"]).parents[1]
    mask_root = slide_dir / "masks"

    masks = []
    for lid in task.layer_ids:
      mp = mask_root / f"layer{lid}" / row["name"]
      if mp.exists():
        m = cv2.imread(str(mp), 0) > 127
      else:
        m = np.zeros(img.shape[:2], bool)
      masks.append(m)

    if task.type == "binary":
      mask = masks[0].astype(np.float32)
    else:
      mask = np.zeros(img.shape[:2], np.int64)
      for i, m in enumerate(masks):
        mask[m] = i + 1  # background=0

    if self.transform:
      out = self.transform(image=img, mask=mask)
      img, mask = out["image"], out["mask"]

    img = img.float()
    if task.type == "binary":
      mask = mask.float()
    else:
      mask = mask.long()

    return img, mask
