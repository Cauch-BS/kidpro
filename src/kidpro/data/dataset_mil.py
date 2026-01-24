from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config.schema import AppCfg
from ..utils.wsidata import (
    extract_tile_xy,
    open_wsidata,
    reservoir_sample,
    tile_image_to_array,
)


class MILDataset(Dataset):
  def __init__(self, cfg: AppCfg, df_slide: pd.DataFrame, transform: Optional[Callable] = None) -> None:
    self.cfg = cfg
    if cfg.dataset.task.type != "mil":
      raise ValueError("MILDataset requires task.type == 'mil'")
    self.df = df_slide.reset_index(drop=True)
    self.cache_root = Path(cfg.dataset.paths.cache_dir) if cfg.dataset.paths.cache_dir else None
    self.tiles_key = cfg.dataset.paths.tiles_key
    self.transform = transform
    self.max_patches = cfg.dataset.task.max_patches
    self.sample_mode = cfg.dataset.task.sample_mode
    self.wsi_dir = Path(cfg.dataset.paths.wsi_dir) if cfg.dataset.paths.wsi_dir else None
    self.wsi_ext = cfg.dataset.paths.wsi_ext or ".svs"
    if not self.wsi_ext.startswith("."):
      self.wsi_ext = f".{self.wsi_ext}"

    self._cache_paths: dict[str, Path] = {}

    for c in ["SlideName", "GT", "split"]:
      if c not in self.df.columns:
        raise ValueError(f"Missing required column: {c}")

  def __len__(self) -> int:
    return len(self.df)

  def _resolve_cache_path(self, slide_name: str) -> Path:
    cached = self._cache_paths.get(slide_name)
    if cached is not None:
      return cached

    if self.cache_root is None:
      raise RuntimeError("dataset.paths.cache_dir is required for MIL wsidata loading.")
    cache_path = self.cache_root / f"{slide_name}.zarr"
    self._cache_paths[slide_name] = cache_path
    return cache_path

  def _resolve_slide_path(self, slide_name: str) -> Path:
    if self.wsi_dir is None:
      raise RuntimeError("dataset.paths.wsi_dir is required for MIL wsidata loading.")
    return self.wsi_dir / f"{slide_name}{self.wsi_ext}"

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    row = self.df.iloc[idx]
    slide_name = str(row["SlideName"])
    gt_val = row["GT"]

    if pd.isna(gt_val):
      raise RuntimeError(f"GT is NaN for slide {slide_name}. This should not be in MIL split set.")

    y = torch.tensor(int(gt_val), dtype=torch.long)

    cache_path = self._resolve_cache_path(slide_name)
    if not cache_path.exists():
      raise RuntimeError(f"Missing wsidata cache for slide {slide_name}: {cache_path}")

    slide_path = self._resolve_slide_path(slide_name)
    wsi = open_wsidata(str(slide_path), cache_path)
    try:
      tiles = wsi.iter.tile_images(self.tiles_key)
      selected_tiles = reservoir_sample(tiles, self.max_patches, self.sample_mode)
      if not selected_tiles:
        raise RuntimeError(f"No tiles available for slide: {slide_name}")

      imgs = []
      coords = []
      for tile in selected_tiles:
        arr = tile_image_to_array(tile)
        if self.transform:
          img = self.transform(image=arr)["image"]
        else:
          img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0 # type: ignore

        imgs.append(img)
        tile_x, tile_y = extract_tile_xy(tile)
        coords.append([tile_x, tile_y])

      if not imgs:
        raise RuntimeError(f"All tiles failed to load for slide: {slide_name}")
    finally:
      if hasattr(wsi, "close"):
        wsi.close()

    # imgs already contains tensors (from ToTensorV2 or manual conversion)
    # but guard in case a numpy array slips through.
    x = torch.stack(
      [img if isinstance(img, torch.Tensor) else torch.from_numpy(img) for img in imgs],
      dim=0,
    )  # (N,3,H,W)
    xy = torch.tensor(coords, dtype=torch.float32)  # (N,2) # type: ignore
    return x, y, xy, slide_name
