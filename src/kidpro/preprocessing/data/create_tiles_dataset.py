#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#
# Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/preprocessing/create_tiles_dataset.py
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from monai.data.wsi_reader import WSIReader
from PIL import Image

from kidpro.preprocessing.data import tiling
from kidpro.preprocessing.data.foreground_segmentation import (
    LoadROId,
    segment_foreground,
)


def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Exclude tiles that are mostly background based on estimated occupancy.

    :param foreground_mask: Boolean array of shape (*, H, W).
    :param occupancy_threshold: Tiles with lower occupancy (between 0 and 1) will be discarded.
    :return: A tuple containing which tiles were selected and the estimated occupancies. These will
    be boolean and float arrays of shape (*,), or scalars if `foreground_mask` is a single tile.
    """
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()  # type: ignore


def save_image(array_chw: np.ndarray, path: Path) -> Image.Image:
    """Save an image array in (C, H, W) format to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float,
                  occupancy_threshold: float, hsv_s_threshold: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Split the foreground of an input slide image into tiles.

    :param slide_image: The RGB image array in (C, H, W) format.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param hsv_s_threshold: Optional HSV value cutoff (0 to 1) to filter black frame pixels.
    :return: A tuple containing the image tiles (N, C, H, W) and tile coordinates (N, 2).
    """
    image_tiles, tile_locations = tiling.tile_array_2d(slide_image, tile_size=tile_size,
                                                       constant_values=255)
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")
    foreground_mask, _ = segment_foreground(
        image_tiles,
        foreground_threshold,
        hsv_s_threshold=hsv_s_threshold,
    )
    selected, _ = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")

    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]

    if len(tile_locations) == 0:
        logging.warn("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    return image_tiles, tile_locations

def _has_existing_tiles(slide_images_dir: Path) -> bool:
    return slide_images_dir.exists() and any(slide_images_dir.glob("*.png"))


def process_slide(sample: Dict[str, Any], level: int, margin: int, tile_size: int,
                  foreground_threshold: Optional[float], occupancy_threshold: float,
                  output_dir: Path, overwrite: bool = False,
                  hsv_s_threshold: Optional[float] = None) -> Path:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.
    :param level: Magnification level at which to process the slide.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param hsv_s_threshold: Optional HSV value cutoff (0 to 1) to filter black frame pixels.
    :param output_dir: Root directory for the output dataset; outputs for a single slide will be
    saved inside `output_dir/slide_id/images`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_id = str(sample["slide_id"])
    slide_image_path = Path(sample["image"])
    output_slide_dir = output_dir / slide_id / "images"

    if _has_existing_tiles(output_slide_dir) and not overwrite:
        logging.info("Skipping slide %s (tiles already exist).", slide_id)
        return output_slide_dir

    output_slide_dir.mkdir(parents=True, exist_ok=True)

    loader = LoadROId(
        WSIReader(backend="OpenSlide"),
        level=level,
        margin=margin,
        foreground_threshold=foreground_threshold,
        hsv_s_threshold=hsv_s_threshold,
    )
    sample = loader(sample)  # load 'image' from disk

    logging.info("Tiling slide %s from %s ...", slide_id, slide_image_path)
    image_tiles, rel_tile_locations = generate_tiles(
        sample["image"],
        tile_size,
        sample["foreground_threshold"],
        occupancy_threshold,
        hsv_s_threshold=hsv_s_threshold,
    )

    # origin in level-0 coordinate
    tile_locations = (sample["scale"] * rel_tile_locations + sample["origin"]).astype(int)

    n_tiles = image_tiles.shape[0]
    logging.info("%s tiles selected for slide %s.", n_tiles, slide_id)
    for idx in range(n_tiles):
        x, y = tile_locations[idx]
        out_name = f"{slide_id}_X0Y0_{x:06d}_{y:06d}.png"
        save_image(image_tiles[idx], output_slide_dir / out_name)

    return output_slide_dir


def build_slide_samples(
    csv_path: Path,
    slide_col: str,
    image_col: Optional[str],
    wsi_dir: Optional[Path],
    wsi_ext: str,
) -> list[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if slide_col not in df.columns:
        raise ValueError(f"Missing required column: {slide_col}")
    df = df.drop_duplicates(subset=[slide_col]).copy()

    if image_col is None:
        if "image_path" in df.columns:
            image_col = "image_path"
        elif "image" in df.columns:
            image_col = "image"

    image_col_present = image_col and image_col in df.columns
    if not image_col_present and wsi_dir is None:
        raise ValueError(
            f"Missing {image_col!r} column and wsi_dir is not set; "
            "cannot resolve slide paths."
        )

    wsi_ext = wsi_ext if wsi_ext.startswith(".") else f".{wsi_ext}"
    samples: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        slide_id = str(row[slide_col])
        if image_col_present:
            image_path = Path(row[image_col])  # type: ignore[index]
        else:
            image_path = wsi_dir / f"{slide_id}{wsi_ext}"  # type: ignore[operator]
        samples.append({"slide_id": slide_id, "image": str(image_path)})

    return samples


def process_dataset(
    samples: Iterable[Dict[str, Any]],
    output_dir: Path,
    level: int,
    tile_size: int,
    margin: int,
    foreground_threshold: Optional[float],
    occupancy_threshold: float,
    hsv_s_threshold: Optional[float] = None,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        process_slide(
            sample=sample,
            level=level,
            margin=margin,
            tile_size=tile_size,
            foreground_threshold=foreground_threshold,
            occupancy_threshold=occupancy_threshold,
            output_dir=output_dir,
            hsv_s_threshold=hsv_s_threshold,
            overwrite=overwrite,
        )
