import logging
import multiprocessing
from functools import partial
from pathlib import Path

import cv2
import hydra
import numpy as np
import openslide
from omegaconf import DictConfig
from tqdm import tqdm

# Ensure these are available
from ..config.load import PATCH_CONFIG
from .magnification import pick_level_for_target_mag
from .rasterize import rasterize_xml_mask

logger = logging.getLogger(__name__)


def _process_one_slide(
  xml_path: Path,
  svs_dir: Path,
  out_dir: Path,
  layer_ids: list[int],
  target_mag: float,
  mask_mag: float,
  patch_size: int,
  stride: int,
  overlap_th: float,
) -> None:
  """
  Worker function to process a single slide.
  Raises exceptions (FileNotFoundError, ValueError) on failure instead of returning strings.
  """
  # Note: No broad try/except block here. We let unexpected errors propagate
  # so the scheduler/caller can handle the traceback.

  svs_path = svs_dir / f"{xml_path.stem}.svs"

  # --- Check 1: File Existence ---
  if not svs_path.exists():
    msg = f"Missing SVS file: {xml_path.stem}"
    logger.error(msg)
    raise FileNotFoundError(msg)

  name_ = svs_path.stem

  # --- Check 2: Name Validation ---
  if len(name_) != 14:
    msg = f"Invalid name length (expected 14): {name_}"
    logger.error(msg)
    raise ValueError(msg)

  slide = openslide.OpenSlide(str(svs_path))

  # Ensure slide is closed even if errors occur during processing
  try:
    # 1. Magnification
    lvl_img, scale_img, base_mag = pick_level_for_target_mag(slide, target_mag)

    # 2. Rasterize Mask
    downsample, layer_masks, union_mask = rasterize_xml_mask(
      xml_path, slide, layer_ids, base_mag, mask_mag
    )
    Hm, Wm = union_mask.shape

    # 3. Prepare Output Dirs
    slide_dir = out_dir / name_
    img_dir = slide_dir / "images"
    mask_root = slide_dir / "masks"

    img_dir.mkdir(parents=True, exist_ok=True)
    for lid in layer_ids:
      (mask_root / f"layer{lid}").mkdir(parents=True, exist_ok=True)

    # 4. Patch Loop
    stride_mask = int(round(stride * (mask_mag / target_mag)))
    patch_mask_size = int(round(patch_size * (mask_mag / target_mag)))

    for y in range(0, Hm, stride_mask):
      for x in range(0, Wm, stride_mask):
        if x + patch_mask_size >= Wm or y + patch_mask_size >= Hm:
          continue

        patch_union = union_mask[y : y + patch_mask_size, x : x + patch_mask_size]
        overlap = np.count_nonzero(patch_union) / patch_union.size

        if overlap < overlap_th:
          continue

        # --- Save Image ---
        x0 = int(x * downsample)
        y0 = int(y * downsample)
        out_name = f"{name_}_X0Y0_{x0:06d}_{y0:06d}.png"

        req = int(round(patch_size * scale_img))
        region = slide.read_region((x0, y0), lvl_img, (req, req))

        img = cv2.resize(
          np.array(region)[:, :, :3],
          (patch_size, patch_size),
          interpolation=cv2.INTER_LINEAR,
        )

        cv2.imwrite(
          str(img_dir / out_name),
          cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
          [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )

        # --- Save Masks ---
        for lid in layer_ids:
          mp = layer_masks[lid][y : y + patch_mask_size, x : x + patch_mask_size]

          if target_mag != mask_mag:
            mp = cv2.resize(
              mp,
              (patch_size, patch_size),
              interpolation=cv2.INTER_NEAREST,
            )

          # === CHECK FOR EMPTY MASK ===
          if np.count_nonzero(mp) == 0:
            # Log a warning for data quality monitoring
            logger.warning(f"Empty mask generated: {out_name} (Layer {lid})")
            continue

          cv2.imwrite(
            str(mask_root / f"layer{lid}" / out_name),
            (mp * 255).astype(np.uint8),
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
          )
  finally:
    # Ensures slide file handle is released even if an error crashes the loop
    slide.close()


def patch_multi(
  segmentation_type: str,
  layer_ids: list[int],
  svs_dir: Path,
  xml_dir: Path,
  out_home: Path,
  output_map: dict[str, str],
  target_mag: float = 10.0,
  mask_mag: float = 2.5,
  patch_size: int = 256,
  stride: int = 256,
  overlap_th: float = 0.05,
  num_workers: int = 16,
  allowed_mags: list[float] | None = None,
) -> None:
  if not xml_dir.exists():
    raise FileNotFoundError(f"XML directory does not exist: {xml_dir}")

  xml_list = sorted(xml_dir.glob("*.xml"))
  if not xml_list:
    raise RuntimeError(f"No XML files found in: {xml_dir}")

  if segmentation_type not in output_map:
    raise KeyError(f"Invalid type. Choose from: {list(output_map.keys())}")

  out_dir = out_home / output_map[segmentation_type]
  out_dir.mkdir(parents=True, exist_ok=True)

  allowed = set(allowed_mags) if allowed_mags else {2.5, 10.0, 40.0}
  if target_mag not in allowed or mask_mag not in allowed:
    raise ValueError(f"Invalid magnification. Allowed: {allowed}")

  worker_func = partial(
    _process_one_slide,
    svs_dir=svs_dir,
    out_dir=out_dir,
    layer_ids=layer_ids,
    target_mag=target_mag,
    mask_mag=mask_mag,
    patch_size=patch_size,
    stride=stride,
    overlap_th=overlap_th,
  )

  logger.info(
    "Starting multiprocessing with %s workers for %s slides.",
    num_workers,
    len(xml_list),
  )

  with multiprocessing.Pool(num_workers) as pool:
    results = list(
      tqdm(
        pool.imap_unordered(worker_func, xml_list),
        total=len(xml_list),
        desc="Processing Slides",
      )
    )

  for res in results:
    if res:
      logger.info("Worker message: %s", res)

  logger.info("Processing complete for %s.", segmentation_type)


def _configure_logging(log_path: Path, level: str) -> None:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  logging.basicConfig(
    filename=str(log_path),
    filemode="a",
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
  )


@hydra.main(version_base=None, config_path="conf", config_name="patch/default")
def main(hcfg: DictConfig) -> None:
  cfg = PATCH_CONFIG(hcfg)

  log_path = Path(cfg.logging.log_file)
  if not log_path.is_absolute():
    log_path = Path.cwd() / log_path
  _configure_logging(log_path, cfg.logging.level)

  patch_multi(
    segmentation_type=cfg.segmentation_type,
    layer_ids=cfg.layer_ids,
    svs_dir=cfg.paths.svs_dir,
    xml_dir=cfg.paths.xml_dir,
    out_home=cfg.paths.out_home,
    output_map=cfg.output_map,
    target_mag=cfg.params.target_mag,
    mask_mag=cfg.params.mask_mag,
    patch_size=cfg.params.patch_size,
    stride=cfg.params.stride,
    overlap_th=cfg.params.overlap_th,
    num_workers=cfg.params.num_workers,
    allowed_mags=cfg.params.allowed_mags,
  )


if __name__ == "__main__":
  main()
