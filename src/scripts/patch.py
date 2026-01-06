import logging
import multiprocessing
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import openslide
from tqdm import tqdm

# Ensure these are available
from .magnification import pick_level_for_target_mag
from .rasterize import rasterize_xml_mask

SVS_DIR   = Path("/home/khdp-user/workspace/dataset/Slide")
XML_DIR   = Path("/home/khdp-user/workspace/dataset/Annotation/Glomerulus")
OUT_HOME  = Path("/home/khdp-user/workspace/output")
OUT_PARS  = {
    "glomerulus" : "Glom_patch",
    "ifta" : "IFTA_patch",
    "inflammation" : "Infla_patch",
    "outcome" : "MIL_patch"
}

ALLOWED_MAGS = {2.5, 10.0, 40.0}

logging.basicConfig(
    filename='process.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def _process_one_slide(
    xml_path: Path,
    out_dir: Path,
    layer_ids: list[int],
    target_mag: float,
    mask_mag: float,
    patch_size: int,
    stride: int,
    overlap_th: float
) -> None:
    """
    Worker function to process a single slide.
    Raises exceptions (FileNotFoundError, ValueError) on failure instead of returning strings.
    """
    # Note: No broad try/except block here. We let unexpected errors propagate
    # so the scheduler/caller can handle the traceback.

    svs_path = SVS_DIR / f"{xml_path.stem}.svs"

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
        img_dir   = slide_dir / "images"
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

                patch_union = union_mask[y : y+patch_mask_size , x : x+patch_mask_size]
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
                    [cv2.IMWRITE_PNG_COMPRESSION, 1]
                )

                # --- Save Masks ---
                for lid in layer_ids:
                    mp = layer_masks[lid][y:y+patch_mask_size, x:x+patch_mask_size]

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
                        [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    )
    finally:
        # Ensures slide file handle is released even if an error crashes the loop
        slide.close()


def patch_multi(
    segmentation_type: str,
    layer_ids: list[int],
    target_mag: float = 10.0,
    mask_mag: float = 2.5,
    patch_size: int = 512,
    stride: int = 512,
    overlap_th: float = 0.05,
    num_workers: int = 16,
) -> None:

    xml_list = sorted(XML_DIR.glob("*.xml"))
    if not xml_list:
        raise RuntimeError("No XML files found.")

    if segmentation_type not in OUT_PARS:
        raise KeyError(f"Invalid type. Choose from: {list(OUT_PARS.keys())}")

    out_dir = OUT_HOME / OUT_PARS[segmentation_type]
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_mag not in ALLOWED_MAGS or mask_mag not in ALLOWED_MAGS:
        raise ValueError(f"Invalid magnification. Allowed: {ALLOWED_MAGS}")

    worker_func = partial(
        _process_one_slide,
        out_dir=out_dir,
        layer_ids=layer_ids,
        target_mag=target_mag,
        mask_mag=mask_mag,
        patch_size=patch_size,
        stride=stride,
        overlap_th=overlap_th
    )

    print(f"Starting multiprocessing with {num_workers} workers for {len(xml_list)} slides...")

    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_func, xml_list),
            total=len(xml_list),
            desc="Processing Slides"
        ))

    for res in results:
        if res:
            print(res)

    print(f"[DONE] Processing complete for {segmentation_type}.")

if __name__ == "__main__":
    patch_multi("glomerulus", [2])
    # =========================
    # CONFIG
    #   Glom segmentation (dataset/Annotation/Glmerulus)
    #     Id == 1 -> Core ROI
    #     Id == 2 -> Glomerulus
    # =========================

    patch_multi("ifta", [1, 2, 3])
    # =========================
    # CONFIG
    #   segmentation annotation info
    #   IFTA & Excluded ROI (dataset/Annotation/IFTA_Exception)
    #     Id == 1 -> IFTA
    #     Id == 2 -> Medulla (Excluded ROI)
    #     Id == 3 -> Extrarenal tissue & capsule (Excluded ROI)
    #     Id == 4 -> Core ROI
    #     Id == 5 -> IFTA control
    # =========================

    patch_multi("inflammation", [1])
    # =========================
    # CONFIG
    #   Inflammation (dataset/Annotation/Inflammation)
    #     Id == 1 -> Inflammation
    #     Id == 2 -> Inflammation control
    # =========================

    patch_multi("outcome", [1, 2], overlap_th = 0.3)
    # =========================
    # CONFIG
    #   Glom segmentation (dataset/Annotation/Glmerulus)
    #     Id == 1 -> Core ROI
    #     Id == 2 -> Glomerulus
    # =========================
