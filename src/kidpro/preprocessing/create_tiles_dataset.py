import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import tqdm
from PIL import Image

from ..utils.wsidata import extract_tile_xy, open_wsidata, tile_image_to_array

try:
    from wsidata import WSIData
except ImportError as exc:
    raise ImportError(
        "wsidata is required for wsidata preprocessing; install wsidata to continue."
        "Use `conda install -c conda-forge wsidata` to install."
    ) from exc

try:
    import lazyslide as zs
except ImportError as exc:
    raise ImportError(
        "lazyslide is required for wsi preprocessing; install lazyslide to continue."
        "Use `conda install -c conda-forge lazyslide` to install."
    ) from exc


def _has_existing_tiles(slide_images_dir: Path) -> bool:
    return slide_images_dir.exists() and any(slide_images_dir.glob("*.png"))


def _resolve_mpp(slide_path: Path, level: int) -> Optional[float]:
    try:
        slide: WSIData = open_wsidata(str(slide_path))
    except Exception as exc:
        logging.warning("Failed to open slide for MPP lookup: %s", exc)
        return None

    try:
        props = slide.properties
        if props.mpp is None:
            return None
        mpp = float(props.mpp)
        if level < props.n_level:
            mpp *= float(props.level_downsample[level])
        return mpp
    except Exception as exc:
        logging.warning("Failed to resolve MPP from slide metadata: %s", exc)
        return None
    finally:
        slide.close()


def _tile_tissues(
    wsi: WSIData,
    tiles_key: str,
    tile_size: int,
    level: int,
    mpp: float,
) -> None:
    zs.pp.tile_tissues(
        wsi,
        key_added = tiles_key,
        tile_px = tile_size,
        mpp = mpp,
        ops_level = level,
        return_tiles = False,
    )


def _export_patch_layout(
    wsi: WSIData,
    tiles_key: str,
    patch_root: Path,
    slide_id: str,
    overwrite: bool,
) -> Path:
    output_slide_dir = patch_root / slide_id / "images"
    if _has_existing_tiles(output_slide_dir) and not overwrite:
        logging.info("Skipping patch export for %s (tiles already exist).", slide_id)
        return output_slide_dir

    output_slide_dir.mkdir(parents=True, exist_ok=True)
    for tile in wsi.iter.tile_images(tiles_key):
        arr = tile_image_to_array(tile)
        x, y = extract_tile_xy(tile)
        out_name = f"{slide_id}_X0Y0_{x:06d}_{y:06d}.png"
        Image.fromarray(arr).save(output_slide_dir / out_name)
    return output_slide_dir


def _export_tiles_preview(
    wsi: Any,
    tiles_key: str,
    tiles_dir: Path,
    slide_id: str,
    overwrite: bool,
) -> Path:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = tiles_dir / slide_id
    if _has_existing_tiles(preview_dir) and not overwrite:
        logging.info("Skipping preview export for %s (tiles already exist).", slide_id)
        return preview_dir

    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, tile in enumerate(wsi.iter.tile_images(tiles_key)):
        arr = tile_image_to_array(tile)
        tile_id = getattr(tile, "id", idx)
        Image.fromarray(arr).save(preview_dir / f"tile_{tile_id}.png")
    return preview_dir


def process_slide(
    sample: dict[str, Any],
    level: int,
    tiles_key: str,
    tile_size: int,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool = False,
    save_tiles: bool = False,
    export_patches: bool = True,
    tiles_dir: Optional[Path] = None,
) -> Path:
    slide_id = str(sample["slide_id"])
    slide_image_path = Path(sample["image"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{slide_id}.zarr"

    wsi = None
    try:
        if cache_path.exists() and not overwrite:
            logging.info("Using existing wsidata cache for %s", slide_id)
            wsi = open_wsidata(str(cache_path))
        else:
            logging.info("Creating wsidata cache for %s from %s", slide_id, slide_image_path)
            wsi = open_wsidata(str(slide_image_path))
            zs.pp.find_tissues(wsi)
            act_mpp = _resolve_mpp(slide_image_path, level) or 0.5
            used_mpp = max(act_mpp, 0.5)
            if used_mpp != act_mpp:
                logging.warning("Using MPP %s for %s (actual MPP: %s)", used_mpp, slide_id, act_mpp)
            _tile_tissues(wsi, tiles_key, tile_size, level, used_mpp)
            wsi.write(str(cache_path))

        if export_patches:
            _export_patch_layout(wsi, tiles_key, output_dir, slide_id, overwrite)
        if save_tiles:
            preview_root = tiles_dir if tiles_dir is not None else output_dir / "tiles_preview"
            _export_tiles_preview(wsi, tiles_key, preview_root, slide_id, overwrite)
    finally:
        if wsi is not None:
            wsi.close()

    return cache_path


def build_slide_samples(
    csv_path: Path,
    slide_col: str,
    image_col: Optional[str],
    wsi_dir: Optional[Path],
    wsi_ext: str,
) -> list[dict[str, Any]]:
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
    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        slide_id = str(row[slide_col])
        if image_col_present:
            image_path = Path(row[image_col])  # type: ignore[index]
        else:
            image_path = wsi_dir / f"{slide_id}{wsi_ext}"  # type: ignore[operator]
        samples.append({"slide_id": slide_id, "image": str(image_path)})

    return samples


def process_dataset(
    samples: Iterable[dict[str, Any]],
    output_dir: Path,
    cache_dir: Path,
    level: int,
    tile_size: int,
    overwrite: bool = False,
    save_tiles: bool = False,
    export_patches: bool = True,
    tiles_key: str = "tiles",
    tiles_dir: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in tqdm.tqdm(samples, desc="Processing slides"):
        process_slide(
            sample=sample,
            level=level,
            tile_size=tile_size,
            output_dir=output_dir,
            cache_dir=cache_dir,
            tiles_key=tiles_key,
            overwrite=overwrite,
            save_tiles=save_tiles,
            export_patches=export_patches,
            tiles_dir=tiles_dir,
        )
