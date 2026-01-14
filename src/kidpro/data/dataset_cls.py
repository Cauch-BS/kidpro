from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatchClsDataset(Dataset):
  def __init__(self, df: pd.DataFrame, transform: Optional[Any] = None):
    self.df = df.reset_index(drop=True)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    row = self.df.iloc[idx]
    img = cv2.imread(row["path"])
    if img is None:
      raise RuntimeError(f"Failed to read image: {row['path']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if self.transform:
      img = self.transform(image=img)["image"]

    y = torch.tensor(int(row["y"]), dtype=torch.long)
    return img, y
