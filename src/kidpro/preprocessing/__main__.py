import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..config.load import PREPROCESS_CONFIG
from .data.create_tiles_dataset import build_slide_samples, process_dataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="preprocess")
def main(hcfg: DictConfig) -> None:
    cfg = PREPROCESS_CONFIG(hcfg)

    paths = cfg.paths
    slide_csv = Path(paths.label_csv)
    slide_samples = build_slide_samples(
        csv_path=slide_csv,
        slide_col="SlideName",
        image_col=None,
        wsi_dir=getattr(paths, "wsi_dir", None),
        wsi_ext=getattr(paths, "wsi_ext", ".svs"),
    )

    preprocess = cfg.preprocess
    output_dir = Path(paths.root_dir)
    process_dataset(
        samples=slide_samples,
        output_dir=output_dir,
        level=preprocess.level,
        tile_size=cfg.data.patch_size,
        margin=preprocess.margin,
        foreground_threshold=preprocess.foreground_threshold,
        occupancy_threshold=preprocess.occupancy_threshold,
        hsv_s_threshold=getattr(preprocess, "hsv_s_threshold", None),
        overwrite=preprocess.overwrite,
    )
    log.info("Preprocessing complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
