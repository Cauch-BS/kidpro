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
  from numcodecs import Blosc
except Exception:  # pragma: no cover - optional dependency
  zarr = None  # type: ignore
  Blosc = None  # type: ignore

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

  def get_cached_tile_embeddings(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Get cached tile embeddings and coords for this slide."""
    return self._dataset._get_cached_tile_embeddings(self.slide_name)

  def set_cached_tile_embeddings(self, embeddings: np.ndarray, coords: np.ndarray) -> None:
    """Cache tile embeddings and coords for this slide."""
    self._dataset._set_cached_tile_embeddings(self.slide_name, embeddings, coords)


class _ZarrTileWriter:
  def __init__(self, group: Any, chunk_size: int, compressor: Any | None) -> None:
    self.group = group
    self.chunk_size = chunk_size
    self.tiles_buf: list[np.ndarray] = []
    self.coords_buf: list[list[int]] = []
    self.tiles_ds: Any | None = None
    self.coords_ds: Any | None = None
    self.compressor = compressor

  def _ensure_created(self, arr_shape: tuple[int, ...]) -> None:
    if self.tiles_ds is not None:
      return
    compressors = [self.compressor] if self.compressor is not None else None
    self.tiles_ds = self.group.create_array(
      "tiles",
      shape=(0, *arr_shape),
      chunks=(self.chunk_size, *arr_shape),
      dtype="uint8",
      compressors=compressors,
    )
    self.coords_ds = self.group.create_array(
      "coords",
      shape=(0, 2),
      chunks=(self.chunk_size, 2),
      dtype="int32",
      compressors=compressors,
    )
    self.group.attrs["complete"] = False

  def add(self, arr: np.ndarray, coord: list[int]) -> None:
    self._ensure_created(arr.shape)
    self.tiles_buf.append(arr)
    self.coords_buf.append(coord)
    if len(self.tiles_buf) >= self.chunk_size:
      self.flush()

  def flush(self) -> None:
    if not self.tiles_buf:
      return
    if self.tiles_ds is None or self.coords_ds is None:
      raise RuntimeError("Zarr datasets are not initialized.")
    self.tiles_ds.append(np.stack(self.tiles_buf, axis=0))
    self.coords_ds.append(np.asarray(self.coords_buf, dtype=np.int32))
    self.tiles_buf.clear()
    self.coords_buf.clear()

  def close(self, complete: bool) -> None:
    self.flush()
    if complete:
      self.group.attrs["complete"] = True

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
    self._tile_cache_dir = self._resolve_tile_cache_dir()
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

  def _pooled_emb_enabled(self) -> bool:
    return bool(self.cache_cfg.enabled and self.cache_cfg.cache_pooled_embeddings)

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

  def _zarr_compressor(self) -> Optional[Any]:
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

    # Remove old format markers if present
    if "pooled_emb_complete" in group.attrs:
      del group.attrs["pooled_emb_complete"]

  def _get_cached_tile_embeddings(self, slide_name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Get cached tile embeddings and coords for a slide."""
    if not self._pooled_emb_enabled():
      return None
    cache_path = self._zarr_cache_path(slide_name)
    if cache_path is None or not cache_path.exists():
      return None
    group = self._open_zarr_group(cache_path, mode="r")
    return self._read_tile_embeddings(group)

  def _set_cached_tile_embeddings(self, slide_name: str, embeddings: np.ndarray, coords: np.ndarray) -> None:
    """Cache tile embeddings and coords for a slide."""
    if not self._pooled_emb_enabled():
      return
    worker_info = get_worker_info()
    if worker_info is not None:
      return  # Don't write from worker processes
    cache_path = self._zarr_cache_path(slide_name)
    if cache_path is None:
      return
    group = self._open_zarr_group(cache_path, mode="a")
    # If cache is already valid, don't rewrite (important when a slide can be
    # sampled multiple times, e.g. with replacement sampling or repeated eval).
    if self._is_tile_emb_cache_valid(group):
      log.debug("[EMB CACHE SKIP WRITE] slide=%s (already valid)", slide_name)
      return
    self._write_tile_embeddings(group, embeddings, coords)

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
          log.warning(
            "Skipping unreadable tile for slide %s (%s/%s): %s",
            slide_name, consecutive, max_errors, exc,
          )
          self._warned_slides.add(slide_name)
        if consecutive >= max_errors:
          raise RuntimeError(f"Exceeded max errors ({max_errors}) for slide {slide_name}.") from exc
        continue

      consecutive = 0
      yield tile

  def _iter_from_zarr(
    self, group: Any, chunk_size: int
  ) -> Iterable[tuple[np.ndarray, list[int]]]:
    zarr_tiles = cast(Any, group["tiles"])
    zarr_coords = cast(Any, group["coords"])

    for start in range(0, zarr_tiles.shape[0], chunk_size):
      end = min(start + chunk_size, zarr_tiles.shape[0])
      tiles_chunk = zarr_tiles[start:end]
      coords_chunk = zarr_coords[start:end]
      for arr, coord in zip(tiles_chunk, coords_chunk):
        yield arr, [int(coord[0]), int(coord[1])]


  def _iter_from_wsi_building_cache(
    self,
    slide_name: str,
    wsi: Any,
    writer: _ZarrTileWriter | None,
  ) -> Iterable[tuple[np.ndarray, list[int]]]:
    found = False
    for tile in self._safe_iter_tiles(wsi, slide_name):
      found = True
      arr = tile_image_to_array(tile)
      x, y = extract_tile_xy(tile)
      coord = [x, y]
      if writer is not None:
        writer.add(arr, coord)
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

    chunk_size = self.cache_cfg.chunk_size
    worker_info = get_worker_info()
    allow_write = worker_info is None  # your policy

    cache_path = self._zarr_cache_path(slide_name)
    group = None

    # 2) zarr cache read path (complete only)
    if cache_path is not None and cache_path.exists():
      g = self._open_zarr_group(cache_path, mode="r")
      if g.attrs.get("complete", False):
        mem_tiles: list[np.ndarray] | None = [] if self._memory_cache is not None else None
        mem_coords: list[list[int]] | None = [] if self._memory_cache is not None else None
        for arr, coord in self._iter_from_zarr(g, chunk_size):
          if mem_tiles is not None and mem_coords is not None:
            mem_tiles.append(arr)
            mem_coords.append(coord)
          yield arr, coord
        if mem_tiles is not None and mem_coords is not None:
          self._put_memory_cache(slide_name, mem_tiles, mem_coords)
        return
      if allow_write:
        group = self._open_zarr_group(cache_path, mode="w")  # rebuild
    else:
      if cache_path is not None and allow_write:
        group = self._open_zarr_group(cache_path, mode="a")

    # 3) build path: stream from WSI; optionally write cache
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

    writer = None
    if group is not None:
      writer = _ZarrTileWriter(group, chunk_size, self._zarr_compressor())

    mem_tiles_live: list[np.ndarray] | None = [] if self._memory_cache is not None else None
    mem_coords_live: list[list[int]] | None = [] if self._memory_cache is not None else None

    try:
      for arr, coord in self._iter_from_wsi_building_cache(slide_name, wsi, writer):
        if mem_tiles_live is not None and mem_coords_live is not None:
          mem_tiles_live.append(arr)
          mem_coords_live.append(coord)
        yield arr, coord
      if writer is not None:
        writer.close(complete=True)
    finally:
      if writer is not None:
        # if an exception occurs mid-build, ensure we don't mark complete
        # (writer.close called above only on success)
        pass
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

    # Ensure target has batch dimension for CrossEntropyLoss
    y = torch.tensor([int(gt_val)], dtype=torch.long)
    return TileStream(self, slide_name), y, slide_name
