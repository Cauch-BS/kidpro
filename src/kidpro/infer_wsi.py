from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig

from .config.load import CONFIG, RuntimeResolved, resolve_best_model_from_mlflow
from .config.schema import AppCfg, InferenceCfg
from .data.transform import get_transforms
from .modeling.factory_wsi import build_model_mil
from .modeling.sources import load_state_dict_generic
from .preprocessing.create_tiles_dataset import process_slide
from .utils.wsidata import extract_tile_xy, open_wsidata, tile_image_to_array

log = logging.getLogger(__name__)


def _resolve_fallback_weights(cfg: AppCfg, infer_cfg: InferenceCfg) -> Path:
  fallback = infer_cfg.fallback_weights
  if fallback is None:
    fallback = Path.cwd() / "models" / "best_model.pt"
  fallback = Path(fallback)
  if not fallback.exists():
    raise FileNotFoundError(
      f"Fallback weights not found at {fallback}. "
      "Provide inference.fallback_weights or add models/best_model.pt."
    )
  return fallback


def _resolve_wsi_weights(cfg: AppCfg, infer_cfg: InferenceCfg) -> Tuple[Path, str]:
  if cfg.mlflow.enabled:
    try:
      model_name = cfg.mlflow.registry_model_name
      ckpt_path = resolve_best_model_from_mlflow(cfg, model_name)
      return ckpt_path, "mlflow"
    except Exception as e:
      log.warning("MLflow resolution failed; falling back to local weights. err=%s", e)
  fallback = _resolve_fallback_weights(cfg, infer_cfg)
  return fallback, "fallback"


def _derive_slide_id(infer_cfg: InferenceCfg) -> str:
  if infer_cfg.slide_id:
    return infer_cfg.slide_id
  return Path(infer_cfg.wsi_path).stem


def _resolve_cache_dir(cfg: AppCfg, infer_cfg: InferenceCfg) -> Path:
  if infer_cfg.cache_dir:
    return Path(infer_cfg.cache_dir)
  if cfg.dataset.paths.cache_dir:
    return Path(cfg.dataset.paths.cache_dir)
  return Path(cfg.run_dir or Path.cwd()) / "wsidata_cache"


def _resolve_patch_dir(cfg: AppCfg, infer_cfg: InferenceCfg) -> Path:
  if infer_cfg.patch_dir:
    return Path(infer_cfg.patch_dir)
  return Path(cfg.run_dir or Path.cwd()) / "tiles"


def _ensure_cache(
  cfg: AppCfg,
  infer_cfg: InferenceCfg,
  slide_id: str,
  tile_size: int,
) -> Path:
  cache_dir = _resolve_cache_dir(cfg, infer_cfg)
  cache_dir.mkdir(parents=True, exist_ok=True)
  cache_path = cache_dir / f"{slide_id}.zarr"
  if cache_path.exists() and not infer_cfg.preprocess.overwrite:
    return cache_path

  patch_dir = _resolve_patch_dir(cfg, infer_cfg)
  tiles_dir = patch_dir / "preview" if infer_cfg.preprocess.save_tiles else None
  process_slide(
    sample={"slide_id": slide_id, "image": str(infer_cfg.wsi_path)},
    level=infer_cfg.preprocess.level,
    tile_size=tile_size,
    output_dir=patch_dir,
    cache_dir=cache_dir,
    tiles_key=infer_cfg.preprocess.tiles_key,
    overwrite=infer_cfg.preprocess.overwrite,
    save_tiles=infer_cfg.preprocess.save_tiles,
    export_patches=infer_cfg.preprocess.export_patches,
    tiles_dir=tiles_dir,
  )
  return cache_path


def _load_tiles_from_cache(
  cache_path: Path,
  tiles_key: str,
  transform: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  wsi = open_wsidata(str(cache_path))
  try:
    imgs: list[torch.Tensor] = []
    coords: list[list[int]] = []
    for tile in wsi.iter.tile_images(tiles_key):
      arr = tile_image_to_array(tile)
      if transform:
        img_t = transform(image=arr)["image"]
      else:
        img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # type: ignore

      if not isinstance(img_t, torch.Tensor):
        img_t = torch.from_numpy(img_t)
      imgs.append(img_t)

      tile_x, tile_y = extract_tile_xy(tile)
      coords.append([tile_x, tile_y])

    if not imgs:
      raise RuntimeError(f"No tiles found in wsidata cache: {cache_path}")
  finally:
    if hasattr(wsi, "close"):
      wsi.close()

  x = torch.stack(imgs, dim=0)
  xy = torch.tensor(coords, dtype=torch.float32)
  return x, xy


@torch.no_grad()
def run_wsi_inference(cfg: AppCfg, rr: RuntimeResolved) -> Dict[str, Any]:
  if cfg.inference is None:
    raise ValueError("Missing inference config. Provide inference.* in the Hydra config.")
  infer_cfg = cfg.inference

  slide_id = _derive_slide_id(infer_cfg)
  output_dir = Path(infer_cfg.output_dir) if infer_cfg.output_dir else Path(cfg.run_dir or Path.cwd())
  output_dir.mkdir(parents=True, exist_ok=True)

  tile_size = infer_cfg.tile_size or cfg.dataset.data.patch_size
  cache_path = _ensure_cache(cfg, infer_cfg, slide_id=slide_id, tile_size=tile_size)

  _, val_tf = get_transforms(cfg)
  x, coords = _load_tiles_from_cache(
    cache_path,
    tiles_key=infer_cfg.preprocess.tiles_key,
    transform=val_tf,
  )

  model = build_model_mil(cfg)
  ckpt_path, source = _resolve_wsi_weights(cfg, infer_cfg)
  load_state_dict_generic(model, ckpt_path)
  model = model.to(rr.device)
  model.eval()

  x = x.to(rr.device, non_blocking=True)
  coords = coords.to(rr.device, non_blocking=True)

  use_amp = rr.device == "cuda"
  if use_amp:
    with torch.autocast(device_type="cuda"):
      logits = model(x, coords)
  else:
    logits = model(x, coords)
  probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
  pred = int(torch.argmax(logits, dim=1).item())

  patch_dir = _resolve_patch_dir(cfg, infer_cfg)
  result = {
    "slide_id": slide_id,
    "wsi_path": str(infer_cfg.wsi_path),
    "num_patches": int(x.size(0)),
    "probabilities": probs,
    "predicted_class": pred,
    "weights_path": str(ckpt_path),
    "weights_source": source,
    "tiles_cache": str(cache_path),
    "patch_dir": str(patch_dir / slide_id / "images")
    if infer_cfg.preprocess.export_patches
    else None,
  }

  out_path = output_dir / infer_cfg.output_json
  with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
  log.info("Inference complete. Output: %s", out_path)

  if infer_cfg.cleanup_tiles and infer_cfg.preprocess.export_patches:
    shutil.rmtree(patch_dir / slide_id, ignore_errors=True)

  return result


@hydra.main(version_base=None, config_path="conf", config_name="infer_wsi")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)
  run_wsi_inference(cfg, rr)


if __name__ == "__main__":
  main()
