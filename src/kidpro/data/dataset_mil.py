from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from ..config.schema import AppCfg
from ..utils.wsidata import extract_tile_xy, open_wsidata, tile_image_to_array

try:
  import zarr
  from numcodecs import Blosc
except Exception:  # pragma: no cover - optional dependency
  zarr = None  # type: ignore
  Blosc = None  # type: ignore


class TileStream:
  def __init__(self, dataset: "MILDataset", slide_name: str) -> None:
    self._dataset = dataset
    self.slide_name = slide_name

  def iter_tiles(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    return self._dataset._iter_tile_tensors(self.slide_name)

  def iter_batches(self, batch_size: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    batch_imgs: list[torch.Tensor] = []
    batch_coords: list[torch.Tensor] = []
    for img, coord in self._dataset._iter_tile_tensors(self.slide_name):
      batch_imgs.append(img)
      batch_coords.append(coord)
      if len(batch_imgs) >= batch_size:
        yield self._dataset._stack_tiles(batch_imgs), torch.stack(batch_coords, dim=0)
        batch_imgs = []
        batch_coords = []
    if batch_imgs:
      yield self._dataset._stack_tiles(batch_imgs), torch.stack(batch_coords, dim=0)


class MILDataset(Dataset):
  def __init__(self, cfg: AppCfg, df_slide: pd.DataFrame, transform: Optional[Callable] = None) -> None:
    self.cfg = cfg
    if cfg.dataset.task.type != "mil":
      raise ValueError("MILDataset requires task.type == 'mil'")
    self.df = df_slide.reset_index(drop=True)
    self.cache_root = Path(cfg.dataset.paths.cache_dir) if cfg.dataset.paths.cache_dir else None
    self.tiles_key = cfg.dataset.paths.tiles_key
    self.transform = transform
    self.cache_cfg = cfg.dataset.data.mil_cache
    self.wsi_dir = Path(cfg.dataset.paths.wsi_dir) if cfg.dataset.paths.wsi_dir else None
    self.wsi_ext = cfg.dataset.paths.wsi_ext or ".svs"
    if not self.wsi_ext.startswith("."):
      self.wsi_ext = f".{self.wsi_ext}"

    self._cache_paths: dict[str, Path] = {}
    self._tile_cache_dir = self._resolve_tile_cache_dir()
    self._memory_cache: Optional[OrderedDict[str, tuple[list[np.ndarray], list[list[int]]]]] = None
    if self.cache_cfg.memory_max_slides > 0:
      self._memory_cache = OrderedDict()

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

  def _resolve_tile_cache_dir(self) -> Optional[Path]:
    if not self.cache_cfg.enabled:
      return None
    if self.cache_root is None:
      raise RuntimeError("dataset.paths.cache_dir is required for MIL tile caching.")
    return self.cache_root / self.cache_cfg.cache_subdir

  def _zarr_cache_path(self, slide_name: str) -> Optional[Path]:
    if self._tile_cache_dir is None:
      return None
    safe_slide = slide_name.replace("/", "_")
    return self._tile_cache_dir / f"{safe_slide}.zarr"

  def _zarr_compressor(self):
    if self.cache_cfg.compression == "blosc":
      if Blosc is None:
        raise RuntimeError("numcodecs is required for blosc compression.")
      return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    return None

  def _get_memory_cache(self, slide_name: str) -> Optional[tuple[list[np.ndarray], list[list[int]]]]:
    if self._memory_cache is None:
      return None
    cached = self._memory_cache.get(slide_name)
    if cached is not None:
      self._memory_cache.move_to_end(slide_name)
    return cached

  def _put_memory_cache(self, slide_name: str, tiles: list[np.ndarray], coords: list[list[int]]) -> None:
    if self._memory_cache is None:
      return
    self._memory_cache[slide_name] = (tiles, coords)
    self._memory_cache.move_to_end(slide_name)
    while len(self._memory_cache) > self.cache_cfg.memory_max_slides:
      self._memory_cache.popitem(last=False)

  def _open_zarr_group(self, cache_path: Path, mode: str):
    if zarr is None:
      raise RuntimeError("zarr is required for MIL tile caching.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(cache_path), mode=mode)

  def _iter_tile_arrays(self, slide_name: str) -> Iterable[tuple[np.ndarray, list[int]]]:
    cached = self._get_memory_cache(slide_name)
    if cached is not None:
      tiles, coords = cached
      for arr, coord in zip(tiles, coords):
        yield arr, coord
      return

    cache_path = self._zarr_cache_path(slide_name)
    worker_info = get_worker_info()
    allow_write = worker_info is None
    chunk_size = self.cache_cfg.chunk_size

    if cache_path is not None and cache_path.exists():
      group = self._open_zarr_group(cache_path, mode="r")
      if group.attrs.get("complete", False):
        tiles = group["tiles"]
        coords = group["coords"]
        mem_tiles: Optional[list[np.ndarray]] = [] if self._memory_cache is not None else None
        mem_coords: Optional[list[list[int]]] = [] if self._memory_cache is not None else None
        for start in range(0, tiles.shape[0], chunk_size):
          end = min(start + chunk_size, tiles.shape[0])
          tiles_chunk = tiles[start:end]
          coords_chunk = coords[start:end]
          for arr, coord in zip(tiles_chunk, coords_chunk):
            coord_list = [int(coord[0]), int(coord[1])]
            if mem_tiles is not None and mem_coords is not None:
              mem_tiles.append(arr)
              mem_coords.append(coord_list)
            yield arr, coord_list
        if mem_tiles is not None and mem_coords is not None:
          self._put_memory_cache(slide_name, mem_tiles, mem_coords)
        return
      if allow_write:
        group = self._open_zarr_group(cache_path, mode="w")
      else:
        group = None
    else:
      group = self._open_zarr_group(cache_path, mode="a") if cache_path is not None and allow_write else None

    cache_path_wsi = self._resolve_cache_path(slide_name)
    if not cache_path_wsi.exists():
      raise RuntimeError(f"Missing wsidata cache for slide {slide_name}: {cache_path_wsi}")

    slide_path = self._resolve_slide_path(slide_name)
    wsi = open_wsidata(str(slide_path), cache_path_wsi)
    found_tile = False
    mem_tiles: Optional[list[np.ndarray]] = [] if self._memory_cache is not None else None
    mem_coords: Optional[list[list[int]]] = [] if self._memory_cache is not None else None
    tiles_buffer: list[np.ndarray] = []
    coords_buffer: list[list[int]] = []
    tiles_ds = None
    coords_ds = None
    try:
      tiles_iter = wsi.iter.tile_images(self.tiles_key)
      for tile in tiles_iter:
        found_tile = True
        arr = tile_image_to_array(tile)
        tile_x, tile_y = extract_tile_xy(tile)
        coord_list = [tile_x, tile_y]

        if group is not None and tiles_ds is None:
          compressor = self._zarr_compressor()
          tiles_ds = group.create_array(
            "tiles",
            shape=(0, *arr.shape),
            chunks=(chunk_size, *arr.shape),
            dtype="uint8",
            compressor=compressor,
          )
          coords_ds = group.create_array(
            "coords",
            shape=(0, 2),
            chunks=(chunk_size, 2),
            dtype="int32",
            compressor=compressor,
          )
          group.attrs["complete"] = False

        if tiles_ds is not None and coords_ds is not None:
          tiles_buffer.append(arr)
          coords_buffer.append(coord_list)
          if len(tiles_buffer) >= chunk_size:
            tiles_ds.append(np.stack(tiles_buffer, axis=0))
            coords_ds.append(np.asarray(coords_buffer, dtype=np.int32))
            tiles_buffer = []
            coords_buffer = []

        if mem_tiles is not None and mem_coords is not None:
          mem_tiles.append(arr)
          mem_coords.append(coord_list)

        yield arr, coord_list
    finally:
      if hasattr(wsi, "close"):
        wsi.close()

    if not found_tile:
      raise RuntimeError(f"No tiles available for slide: {slide_name}")

    if tiles_ds is not None and coords_ds is not None:
      if tiles_buffer:
        tiles_ds.append(np.stack(tiles_buffer, axis=0))
        coords_ds.append(np.asarray(coords_buffer, dtype=np.int32))
      group.attrs["complete"] = True

    if mem_tiles is not None and mem_coords is not None:
      self._put_memory_cache(slide_name, mem_tiles, mem_coords)

  def _apply_transform(self, arr: np.ndarray) -> torch.Tensor:
    if self.transform is not None:
      img = self.transform(image=arr)["image"]
      if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
      return img
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

  def _stack_tiles(self, imgs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
      [img if isinstance(img, torch.Tensor) else torch.from_numpy(img) for img in imgs],
      dim=0,
    )

  def _iter_tile_tensors(self, slide_name: str) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for arr, coord in self._iter_tile_arrays(slide_name):
      img = self._apply_transform(arr)
      coord_t = torch.tensor(coord, dtype=torch.float32)
      yield img, coord_t

  def __getitem__(self, idx: int) -> tuple[TileStream, torch.Tensor, str]:
    row = self.df.iloc[idx]
    slide_name = str(row["SlideName"])
    gt_val = row["GT"]

    if pd.isna(gt_val):
      raise RuntimeError(f"GT is NaN for slide {slide_name}. This should not be in MIL split set.")

    y = torch.tensor(int(gt_val), dtype=torch.long)
    return TileStream(self, slide_name), y, slide_name
