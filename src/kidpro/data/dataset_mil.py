from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config.schema import AppCfg


class MILDataset(Dataset):
  def __init__(self, cfg: AppCfg, df_slide: pd.DataFrame, transform: Optional[Callable] = None) -> None:
    self.cfg = cfg
    if cfg.dataset.task.type != "mil":
      raise ValueError("MILDataset requires task.type == 'mil'")
    self.df = df_slide.reset_index(drop=True)
    self.patch_root = Path(cfg.dataset.paths.root_dir)
    self.transform = transform
    self.max_patches = cfg.dataset.task.max_patches
    self.sample_mode = cfg.dataset.task.sample_mode

    # FIXED: Proper type annotation for cache
    self._patch_paths_cache: dict[str, list[Path]] = {}

    for c in ["SlideName", "GT", "split"]:
      if c not in self.df.columns:
        raise ValueError(f"Missing required column: {c}")

  def __len__(self) -> int:
    return len(self.df)

  def _collect_patch_paths(self, slide_name: str) -> list[Path]:
    """Collect and cache patch paths for a given slide."""
    # FIXED: Thread-safe cache access (though DataLoader workers have separate memory)
    cached = self._patch_paths_cache.get(slide_name)
    if cached is not None:
      return cached

    img_dir = self.patch_root / slide_name / "images"
    if not img_dir.exists():
      paths: list[Path] = []
    else:
      paths = sorted(img_dir.glob("*.png"))

    # Cache even if empty, to avoid repeated filesystem calls
    self._patch_paths_cache[slide_name] = paths
    return paths

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    row = self.df.iloc[idx]
    slide_name = str(row["SlideName"])
    gt_val = row["GT"]

    if pd.isna(gt_val):
      raise RuntimeError(f"GT is NaN for slide {slide_name}. This should not be in MIL split set.")

    y = torch.tensor(int(gt_val), dtype=torch.long)

    patch_paths = self._collect_patch_paths(slide_name)
    if len(patch_paths) == 0:
      raise RuntimeError(
        f"No patches found for slide: {slide_name} at {self.patch_root/slide_name/'images'}"
      )

    # Sampling
    if self.max_patches is not None and len(patch_paths) > self.max_patches:
      if self.sample_mode == "first":
        patch_paths = patch_paths[: self.max_patches]
      else:
        # FIXED: Convert to list for numpy compatibility
        patch_paths = list(np.random.choice(patch_paths, self.max_patches, replace=False)) #type: ignore

    imgs = []
    for p in patch_paths:
      img = cv2.imread(str(p))
      if img is None:
        continue
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      if self.transform:
        img = self.transform(image=img)["image"]
      else:
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

      imgs.append(img)

    if len(imgs) == 0:
      raise RuntimeError(f"All patches failed to load for slide: {slide_name}")

    x = torch.stack(imgs, dim=0)  # (N,3,H,W)
    return x, y, slide_name
