from __future__ import annotations

import contextlib
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from ..config.schema import AppCfg, MILCacheCfg
from ..utils.wsidata import extract_tile_xy, open_wsidata, tile_image_to_array

try:
  import zarr
except Exception:  # pragma: no cover - optional dependency
  zarr = None  # type: ignore

try:
  from openslide.lowlevel import OpenSlideError
except Exception:  # pragma: no cover - optional dependency
  OpenSlideError = None  # type: ignore

log = logging.getLogger(__name__)

_TILE_EMB_CACHE_VERSION = 2


@contextlib.contextmanager
def _suppress_stderr() -> Iterator[None]:
  """Suppress stderr output from C libraries like libtiff/OpenSlide."""
  devnull = os.open(os.devnull, os.O_WRONLY)
  old_stderr = os.dup(2)
  try:
    os.dup2(devnull, 2)
    yield
  finally:
    os.dup2(old_stderr, 2)
    os.close(devnull)
    os.close(old_stderr)


class TileStream:
  def __init__(self, dataset: "MILDataset", slide_name: str) -> None:
    self._dataset = dataset
    self.slide_name = slide_name

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

  def get_cached_tile_embeddings(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Get cached tile embeddings and coords for this slide."""
    return self._dataset._get_cached_tile_embeddings(self.slide_name)

  def set_cached_tile_embeddings(self, embeddings: np.ndarray, coords: np.ndarray) -> None:
    """Cache tile embeddings and coords for this slide."""
    self._dataset._set_cached_tile_embeddings(self.slide_name, embeddings, coords)


class MILDataset(Dataset):
  def __init__(self, cfg: AppCfg, df_slide: pd.DataFrame, transform: Optional[Callable] = None) -> None:
    self.cfg = cfg
    if cfg.dataset.task.type != "mil":
      raise ValueError("MILDataset requires task.type == 'mil'")
    self.df = df_slide.reset_index(drop=True)
    self.cache_root = Path(cfg.dataset.paths.cache_dir) if cfg.dataset.paths.cache_dir else None
    self.wsi_cache_root = (
      Path(cfg.dataset.paths.wsi_cache_dir)
      if cfg.dataset.paths.wsi_cache_dir
      else self.cache_root
    )
    self.tiles_key = cfg.dataset.paths.tiles_key
    self.transform = transform
    self.cache_cfg: MILCacheCfg = cfg.dataset.data.mil_cache
    self.wsi_dir = Path(cfg.dataset.paths.wsi_dir) if cfg.dataset.paths.wsi_dir else None
    self.wsi_ext = cfg.dataset.paths.wsi_ext or ".svs"
    if not self.wsi_ext.startswith("."):
      self.wsi_ext = f".{self.wsi_ext}"

    self._cache_paths: dict[str, Path] = {}
    self._memory_cache: Optional[OrderedDict[str, tuple[list[np.ndarray], list[list[int]]]]] = None
    if self.cache_cfg.memory_max_slides > 0:
      self._memory_cache = OrderedDict()

    # Track slides that have already warned about unreadable tiles (suppress duplicates)
    self._warned_slides: set[str] = set()

    # Hash of tile_encoder weights for cache invalidation (set by train_wsi.py)
    self._tile_encoder_hash: Optional[str] = None

    for c in ["SlideName", "GT", "split"]:
      if c not in self.df.columns:
        raise ValueError(f"Missing required column: {c}")

  def set_tile_encoder_hash(self, hash_str: str) -> None:
    """Set the tile encoder hash for cache invalidation."""
    if self._tile_encoder_hash == hash_str:
      return
    self._tile_encoder_hash = hash_str

  def _tile_emb_cache_enabled(self) -> bool:
    return bool(self.cache_cfg.enabled and self.cache_cfg.cache_tile_embeddings)

  def __len__(self) -> int:
    return len(self.df)

  def _resolve_cache_path(self, slide_name: str) -> Path:
    cached = self._cache_paths.get(slide_name)
    if cached is not None:
      return cached

    if self.wsi_cache_root is None:
      raise RuntimeError(
        "dataset.paths.wsi_cache_dir (or dataset.paths.cache_dir) is required for MIL wsidata loading."
      )
    cache_path = self.wsi_cache_root / f"{slide_name}.zarr"
    self._cache_paths[slide_name] = cache_path
    return cache_path

  def _resolve_slide_path(self, slide_name: str) -> Path:
    if self.wsi_dir is None:
      raise RuntimeError("dataset.paths.wsi_dir is required for MIL wsidata loading.")
    return self.wsi_dir / f"{slide_name}{self.wsi_ext}"

  def _emb_cache_path(self, slide_name: str) -> Optional[Path]:
    """Path for tile EMBEDDING cache (separate from tile image cache)."""
    if self.cache_root is None:
      return None
    safe_slide = slide_name.replace("/", "_")
    # Use completely separate directory (mil_embeds_cache) to avoid tile cache operations
    # affecting embedding cache
    emb_dir = self.cache_root / "mil_embeds_cache"
    return emb_dir / f"{safe_slide}.zarr"

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

  def _open_zarr_group(self, cache_path: Path, mode: Literal["r", "r+", "a", "w", "w-"]) -> Any:
    if zarr is None:
      raise RuntimeError("zarr is required for MIL tile caching.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(cache_path), mode=mode)

  def _read_tile_embeddings(self, group: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read cached tile embeddings and coords from zarr group."""
    # Check if cache is complete
    if not group.attrs.get("tile_emb_complete", False):
      return None

    # One-time cache bust: ensure old embedding caches (possibly created under
    # stochastic augmentation / older policies) are rebuilt exactly once.
    if group.attrs.get("tile_emb_cache_version") != _TILE_EMB_CACHE_VERSION:
      log.info(
        "[CACHE MISS] tile_emb_cache_version mismatch: cached=%s, expected=%s",
        group.attrs.get("tile_emb_cache_version"),
        _TILE_EMB_CACHE_VERSION,
      )
      return None

    # Check model hash matches (invalidate if tile_encoder changed)
    if self._tile_encoder_hash is not None:
      cached_hash = group.attrs.get("tile_emb_model_hash")
      if cached_hash != self._tile_encoder_hash:
        log.info("[CACHE MISS] tile_encoder hash mismatch: cached=%s, current=%s", cached_hash, self._tile_encoder_hash)
        return None

    # Check for new format (has both embeddings and coords)
    if "tile_embeddings" not in group or "tile_coords" not in group:
      log.info("[CACHE MISS] Old cache format detected, will rebuild")
      return None

    embeddings = np.asarray(group["tile_embeddings"])
    coords = np.asarray(group["tile_coords"])
    return embeddings, coords

  def _is_tile_emb_cache_valid(self, group: Any) -> bool:
    """Fast check (attrs/keys only) to decide whether to reuse tile embeddings cache."""
    if not group.attrs.get("tile_emb_complete", False):
      return False
    if group.attrs.get("tile_emb_cache_version") != _TILE_EMB_CACHE_VERSION:
      return False
    if self._tile_encoder_hash is not None:
      cached_hash = group.attrs.get("tile_emb_model_hash")
      if cached_hash != self._tile_encoder_hash:
        return False
    if "tile_embeddings" not in group or "tile_coords" not in group:
      return False
    return True

  def _write_tile_embeddings(self, group: Any, embeddings: np.ndarray, coords: np.ndarray) -> None:
    """Write tile embeddings and coords to zarr group."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)

    # Write embeddings
    if "tile_embeddings" in group:
      del group["tile_embeddings"]
    group.create_array(
      "tile_embeddings",
      data=embeddings,
      chunks=(min(64, embeddings.shape[0]), embeddings.shape[1]) if embeddings.ndim == 2 else embeddings.shape,
    )

    # Write coords
    if "tile_coords" in group:
      del group["tile_coords"]
    group.create_array(
      "tile_coords",
      data=coords,
      chunks=(min(64, coords.shape[0]), coords.shape[1]) if coords.ndim == 2 else coords.shape,
    )

    # Mark as complete and store hash
    group.attrs["tile_emb_complete"] = True
    group.attrs["tile_emb_cache_version"] = _TILE_EMB_CACHE_VERSION
    if self._tile_encoder_hash is not None:
      group.attrs["tile_emb_model_hash"] = self._tile_encoder_hash

    # Remove legacy marker if present (older cache format).
    if "pooled_emb_complete" in group.attrs:
      del group.attrs["pooled_emb_complete"]

  def _get_cached_tile_embeddings(self, slide_name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Get cached tile embeddings and coords for a slide."""
    if not self._tile_emb_cache_enabled():
      return None
    # Use separate embedding cache path (not shared with tile image cache)
    cache_path = self._emb_cache_path(slide_name)
    if cache_path is None or not cache_path.exists():
      return None
    try:
      group = self._open_zarr_group(cache_path, mode="r")
      result = self._read_tile_embeddings(group)
      if result is None:
        log.debug("[EMB CACHE INVALID] slide=%s (validation failed)", slide_name)
      return result
    except Exception as exc:
      log.warning("[EMB CACHE READ ERROR] slide=%s: %s", slide_name, exc)
      return None

  def _set_cached_tile_embeddings(self, slide_name: str, embeddings: np.ndarray, coords: np.ndarray) -> None:
    """Cache tile embeddings and coords for a slide."""
    if not self._tile_emb_cache_enabled():
      return
    worker_info = get_worker_info()
    if worker_info is not None:
      return  # Don't write from worker processes
    # Use separate embedding cache path (not shared with tile image cache)
    cache_path = self._emb_cache_path(slide_name)
    if cache_path is None:
      return
    try:
      group = self._open_zarr_group(cache_path, mode="a")
      # If cache is already valid, don't rewrite (important when a slide can be
      # sampled multiple times, e.g. with replacement sampling or repeated eval).
      if self._is_tile_emb_cache_valid(group):
        return
      self._write_tile_embeddings(group, embeddings, coords)
      log.debug("[EMB CACHE WRITE] slide=%s embeddings=%s", slide_name, embeddings.shape)
    except Exception as exc:
      log.warning("[EMB CACHE WRITE ERROR] slide=%s: %s", slide_name, exc)

  def _safe_iter_tiles(self, wsi: Any, slide_name: str) -> Iterable[Any]:
    it = iter(wsi.iter.tile_images(self.tiles_key))
    consecutive = 0
    max_errors = 10

    while True:
      try:
        # Suppress libtiff stderr output (e.g., "TIFFReadRawTile: Read error...")
        with _suppress_stderr():
          tile = next(it)
      except StopIteration:
        return
      except Exception as exc:
        if OpenSlideError is None or not isinstance(exc, OpenSlideError):
          raise
        consecutive += 1
        # Only warn once per slide to avoid log spam
        if slide_name not in self._warned_slides:
          log.debug(
            "Skipping unreadable tile for slide %s (%s/%s): %s",
            slide_name, consecutive, max_errors, exc,
          )
          self._warned_slides.add(slide_name)
        if consecutive >= max_errors:
          raise RuntimeError(f"Exceeded max errors ({max_errors}) for slide {slide_name}.") from exc
        continue

      consecutive = 0
      yield tile

  def _iter_from_wsi(
    self,
    slide_name: str,
    wsi: Any,
  ) -> Iterable[tuple[np.ndarray, list[int]]]:
    found = False
    for tile in self._safe_iter_tiles(wsi, slide_name):
      found = True
      arr = tile_image_to_array(tile)
      x, y = extract_tile_xy(tile)
      coord = [x, y]
      yield arr, coord

    if not found:
      raise RuntimeError(f"No tiles available for slide: {slide_name}")

  def _iter_tile_arrays(self, slide_name: str) -> Iterable[tuple[np.ndarray, list[int]]]:
    # 1) memory cache
    cached = self._get_memory_cache(slide_name)
    if cached is not None:
      tiles, coords = cached
      yield from zip(tiles, coords)
      return

    # 2) load from WSI (using wsidata cache)
    cache_path_wsi = self._resolve_cache_path(slide_name)
    if not cache_path_wsi.exists():
      raise RuntimeError(
        "Missing wsidata cache for slide "
        f"{slide_name}: {cache_path_wsi}. "
        "Run preprocessing to create wsidata caches or update "
        "dataset.paths.wsi_cache_dir (or dataset.paths.cache_dir)."
      )

    slide_path = self._resolve_slide_path(slide_name)
    wsi = open_wsidata(str(slide_path), cache_path_wsi)

    mem_tiles_live: list[np.ndarray] | None = [] if self._memory_cache is not None else None
    mem_coords_live: list[list[int]] | None = [] if self._memory_cache is not None else None

    try:
      for arr, coord in self._iter_from_wsi(slide_name, wsi):
        if mem_tiles_live is not None and mem_coords_live is not None:
          mem_tiles_live.append(arr)
          mem_coords_live.append(coord)
        yield arr, coord
    finally:
      if hasattr(wsi, "close"):
        wsi.close()

    if mem_tiles_live is not None and mem_coords_live is not None:
      self._put_memory_cache(slide_name, mem_tiles_live, mem_coords_live)

  def _apply_transform(self, arr: np.ndarray) -> torch.Tensor:
    if self.transform is not None:
      img = self.transform(image=arr)["image"]
      if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
      return cast(torch.Tensor, img)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

  def _stack_tiles(self, imgs: list[torch.Tensor | np.ndarray]) -> torch.Tensor:
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

    # Ensure target has batch dimension for CrossEntropyLoss
    y = torch.tensor([int(gt_val)], dtype=torch.long)
    return TileStream(self, slide_name), y, slide_name
