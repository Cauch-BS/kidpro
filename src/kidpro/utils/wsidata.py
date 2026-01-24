from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from wsidata import WSIData, open_wsi
except ImportError as exc:
    raise ImportError(
        "wsidata is required for WSI tile IO. Install wsidata to use MIL pipelines."
        "Use `conda install -c conda-forge wsidata` to install."
    ) from exc


def open_wsidata(filepath: str) -> WSIData:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"WSI file not found: {filepath}")
    elif not path.is_file():
        raise FileNotFoundError(f"WSI file is not a file: {filepath}")
    else:
        wsi: WSIData = open_wsi(path)
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


def reservoir_sample(
    tiles: Iterable[object],
    max_items: Optional[int],
    sample_mode: str,
) -> list[object]:
    if max_items is None:
        return list(tiles)

    selected: list[object] = []
    if sample_mode == "first":
        for idx, tile in enumerate(tiles):
            if idx >= max_items:
                break
            selected.append(tile)
        return selected

    # Reservoir sampling for random selection without loading all tiles
    for idx, tile in enumerate(tiles):
        if idx < max_items:
            selected.append(tile)
            continue
        j = np.random.randint(0, idx + 1)
        if j < max_items:
            selected[j] = tile
    return selected
