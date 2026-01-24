import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..config.load import PREPROCESS_CONFIG
from .create_tiles_dataset import build_slide_samples, process_dataset

log = logging.getLogger(__name__)


def _configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        filemode="a",
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess")
def main(hcfg: DictConfig) -> None:
    cfg = PREPROCESS_CONFIG(hcfg)

    log_path = Path(cfg.logging.log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    _configure_logging(log_path, cfg.logging.level)

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
    cache_dir = Path(paths.cache_dir)
    process_dataset(
        samples=slide_samples,
        cache_dir=cache_dir,
        level=preprocess.level,
        tile_size=cfg.data.patch_size,
        overwrite=preprocess.overwrite,
        tiles_key=preprocess.tiles_key,
    )
    log.info("Preprocessing complete. Cache: %s", cache_dir)


if __name__ == "__main__":
    main()
