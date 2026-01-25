from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from wsidata import WSIData, open_wsi
except ImportError as exc:
    raise ImportError(
        "wsidata is required for WSI tile IO. Install wsidata to use MIL pipelines."
        "Use `conda install -c conda-forge wsidata` to install."
    ) from exc


_OME_ZARR_QUIETED = False


def _quiet_ome_zarr_logs() -> None:
    global _OME_ZARR_QUIETED
    if _OME_ZARR_QUIETED:
        return
    logging.getLogger("ome_zarr").setLevel(logging.WARNING)
    logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)
    _OME_ZARR_QUIETED = True


def open_wsidata(slide_path: str, store_path: Optional[Path] = None) -> WSIData:
    _quiet_ome_zarr_logs()
    slide = Path(slide_path)
    if not slide.exists():
        raise FileNotFoundError(f"WSI file not found: {slide_path}")
    if not slide.is_file():
        raise FileNotFoundError(f"WSI file is not a file: {slide_path}")

    if store_path is None:
        wsi: WSIData = open_wsi(slide)
        return wsi

    store = Path(store_path)
    if not store.exists():
        raise FileNotFoundError(f"WSI store not found: {store_path}")
    if not store.is_dir():
        raise FileNotFoundError(f"WSI store is not a directory: {store_path}")

    wsi = open_wsi(slide, store=str(store))
    return wsi


def tile_image_to_array(tile: object) -> np.ndarray:
    image = getattr(tile, "image", tile)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def extract_tile_xy(tile: object) -> tuple[int, int]:
    if hasattr(tile, "x") and hasattr(tile, "y"):
        return int(getattr(tile, "x")), int(getattr(tile, "y"))
    meta = getattr(tile, "meta", None)
    if isinstance(meta, dict):
        if "x" in meta and "y" in meta:
            return int(meta["x"]), int(meta["y"])
    raise RuntimeError("Unable to extract tile coordinates from wsidata tile metadata.")
