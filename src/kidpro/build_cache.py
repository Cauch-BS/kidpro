"""
Build tile embedding cache for MIL training.

This script pre-computes and caches tile embeddings for all slides,
eliminating the need to compute them during training. This is especially
useful for large datasets or when running multiple training experiments
with the same tile encoder.

Usage:
    python -m kidpro.build_cache                              # Use defaults
    python -m kidpro.build_cache cache.batch_size=64          # Larger batch size
    python -m kidpro.build_cache model=uni2_h                 # Different encoder
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from .config.load import CONFIG
from .data.dataset_mil import MILDataset, TileStream
from .data.transform import get_transforms
from .modeling.patches import build_foundation, freeze_module

log = logging.getLogger(__name__)


def _compute_model_hash(model: nn.Module) -> str:
    """Compute hash of model parameters for cache invalidation."""
    hasher = hashlib.md5()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


@torch.no_grad()
def _extract_embeddings(
    tile_encoder: nn.Module,
    tile_stream: TileStream,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract tile embeddings for a single slide."""
    all_embeddings = []
    all_coords = []

    for batch_imgs, batch_coords in tile_stream.iter_batches(batch_size):
        batch_imgs = batch_imgs.to(device)
        embeddings = tile_encoder(batch_imgs)
        all_embeddings.append(embeddings.cpu())
        all_coords.append(batch_coords)

    if not all_embeddings:
        raise RuntimeError(f"No tiles found for slide: {tile_stream.slide_name}")

    return torch.cat(all_embeddings, dim=0), torch.cat(all_coords, dim=0)


def _get_slide_count(cache_path: Path, tile_encoder_hash: str) -> Optional[int]:
    """Check if slide is already cached with matching hash. Returns embedding count or None."""
    if not cache_path.exists():
        return None

    try:
        import zarr

        group = zarr.open_group(str(cache_path), mode="r")

        # Check cache validity
        if not group.attrs.get("tile_emb_complete", False):
            return None

        # Version check
        from .data.dataset_mil import _TILE_EMB_CACHE_VERSION

        if group.attrs.get("tile_emb_cache_version") != _TILE_EMB_CACHE_VERSION:
            return None

        # Hash check
        cached_hash = group.attrs.get("tile_emb_model_hash")
        if cached_hash != tile_encoder_hash:
            return None

        # Check for embeddings array
        if "tile_embeddings" not in group:
            return None

        emb_arr = group["tile_embeddings"]
        return emb_arr.shape[0]  # type: ignore
    except Exception:
        return None


def _is_cache_in_progress(cache_path: Path) -> bool:
    """Check if cache exists but is incomplete (likely being written by another process)."""
    if not cache_path.exists():
        return False

    try:
        import zarr

        group = zarr.open_group(str(cache_path), mode="r")
        # If cache exists but tile_emb_complete is False, another process may be writing
        return not group.attrs.get("tile_emb_complete", False)
    except Exception:
        # If we can't read it, assume it's in progress
        return True


def build_cache(
    cfg: Any,
    device: str,
    batch_size: int,
    skip_existing: bool = True,
    force_rebuild: bool = False,
) -> dict[str, int]:
    """
    Build tile embedding cache for all slides in the label CSV.

    Returns dict with processing statistics.
    """
    # Load all slides from label_csv
    if cfg.dataset.paths.label_csv is None:
        raise ValueError("dataset.paths.label_csv is required for cache building.")

    label_csv = Path(cfg.dataset.paths.label_csv)
    df = pd.read_csv(label_csv)

    # Require SlideName column; GT can be missing for unlabeled slides
    if "SlideName" not in df.columns:
        raise ValueError("Label CSV must have 'SlideName' column.")

    # Keep unique slides only
    df = df.drop_duplicates(subset=["SlideName"]).reset_index(drop=True)

    # Fill missing GT with dummy value (we don't use it for caching)
    if "GT" not in df.columns:
        df["GT"] = 0
    else:
        df["GT"] = df["GT"].fillna(0)

    # Add dummy split column (required by MILDataset but not used)
    df["split"] = "cache"

    log.info("[BUILD CACHE] Found %d unique slides in %s", len(df), label_csv)

    # Get validation transforms (deterministic, no augmentation)
    _, val_tf = get_transforms(cfg)

    # Build tile encoder
    log.info("[BUILD CACHE] Building tile encoder...")
    foundation = build_foundation(cfg)
    backbone = foundation.backbone
    tile_encoder = getattr(backbone, "tile_encoder", backbone)
    freeze_module(tile_encoder)
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()

    # Compute tile encoder hash for cache invalidation
    tile_encoder_hash = _compute_model_hash(tile_encoder)
    log.info("[TILE ENCODER HASH] %s", tile_encoder_hash)

    # Create dataset
    ds = MILDataset(cfg, df, transform=val_tf)
    ds.set_tile_encoder_hash(tile_encoder_hash)

    processed = 0
    skipped = 0
    in_progress = 0
    failed = 0
    cached_tiles = 0
    start_time = time.time()

    pbar = tqdm(range(len(ds)), desc="[caching]", unit="slide")

    for idx in pbar:
        tile_stream, _, slide_name = ds[idx]

        # Check cache path
        cache_path = ds._emb_cache_path(slide_name)
        if cache_path is None:
            failed += 1
            log.error("[BUILD CACHE] No cache path for slide=%s", slide_name)
            continue

        # Check if already cached (unless force_rebuild)
        if skip_existing and not force_rebuild:
            existing_count = _get_slide_count(cache_path, tile_encoder_hash)
            if existing_count is not None:
                skipped += 1
                cached_tiles += existing_count
                pbar.set_postfix(done=processed, skip=skipped, busy=in_progress, fail=failed)
                continue

            # Skip if cache exists but is incomplete (another process is writing)
            if _is_cache_in_progress(cache_path):
                in_progress += 1
                log.debug("[BUILD CACHE] Skipping in-progress slide=%s", slide_name)
                pbar.set_postfix(done=processed, skip=skipped, busy=in_progress, fail=failed)
                continue

        try:
            # Extract embeddings
            embeddings, coords = _extract_embeddings(
                tile_encoder, tile_stream, batch_size, device
            )

            # Cache embeddings
            tile_stream.set_cached_tile_embeddings(
                embeddings.numpy(), coords.numpy()
            )

            processed += 1
            cached_tiles += embeddings.shape[0]
            pbar.set_postfix(done=processed, skip=skipped, busy=in_progress, fail=failed)

        except Exception as e:
            failed += 1
            log.error("[BUILD CACHE] Failed for slide=%s: %s", slide_name, e)
            pbar.set_postfix(done=processed, skip=skipped, busy=in_progress, fail=failed)
            continue

    elapsed = time.time() - start_time

    return {
        "processed": processed,
        "skipped": skipped,
        "in_progress": in_progress,
        "failed": failed,
        "total_tiles": cached_tiles,
        "elapsed_seconds": int(elapsed),
    }


@hydra.main(version_base=None, config_path="conf", config_name="cache")
def main(hcfg: DictConfig) -> None:
    run_dir = Path.cwd()
    cfg, rr = CONFIG(hcfg, run_dir=run_dir)

    # Extract cache-specific config
    cache_cfg = hcfg.get("cache", {})
    batch_size = int(cache_cfg.get("batch_size", 32))
    skip_existing = bool(cache_cfg.get("skip_existing", True))
    force_rebuild = bool(cache_cfg.get("force_rebuild", False))

    log.info("[BUILD CACHE] Starting cache build")
    log.info("[BUILD CACHE] batch_size=%d, skip_existing=%s, force_rebuild=%s",
             batch_size, skip_existing, force_rebuild)
    log.info("[BUILD CACHE] label_csv=%s", cfg.dataset.paths.label_csv)
    log.info("[BUILD CACHE] cache_dir=%s", cfg.dataset.paths.cache_dir)
    log.info("[BUILD CACHE] device=%s", rr.device)

    # Verify cache is enabled
    if not cfg.dataset.data.mil_cache.enabled:
        log.warning("[BUILD CACHE] mil_cache.enabled=false, but building cache anyway.")
    if not cfg.dataset.data.mil_cache.cache_pooled_embeddings:
        log.warning("[BUILD CACHE] mil_cache.cache_pooled_embeddings=false, but building cache anyway.")

    stats = build_cache(
        cfg,
        device=rr.device,
        batch_size=batch_size,
        skip_existing=skip_existing,
        force_rebuild=force_rebuild,
    )

    log.info("[BUILD CACHE] Complete. Summary:")
    log.info("  processed:   %d", stats["processed"])
    log.info("  skipped:     %d (already cached)", stats["skipped"])
    log.info("  in_progress: %d (being cached by another process)", stats["in_progress"])
    log.info("  failed:      %d", stats["failed"])
    log.info("  tiles:       %d", stats["total_tiles"])
    log.info("  time:        %ds", stats["elapsed_seconds"])

    if stats["in_progress"] > 0:
        log.info("[BUILD CACHE] Re-run to cache the %d in-progress slides after train_wsi.py finishes.", stats["in_progress"])


if __name__ == "__main__":
    main()
