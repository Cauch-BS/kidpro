from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import hydra
import torch
from omegaconf import DictConfig

from .config.load import CONFIG, RuntimeResolved, resolve_best_model_from_mlflow
from .config.schema import AppCfg, InferenceCfg
from .data.transform import get_transforms
from .modeling.factory_wsi import build_model_mil
from .modeling.sources import load_state_dict_generic
from .preprocessing.data.create_tiles_dataset import process_slide

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


def _generate_tiles(
  infer_cfg: InferenceCfg,
  slide_id: str,
  tile_size: int,
) -> Path:
  tiles_root = Path(infer_cfg.tiles_dir) if infer_cfg.tiles_dir else Path.cwd() / "tiles"
  tiles_root.mkdir(parents=True, exist_ok=True)

  process_slide(
    sample={"slide_id": slide_id, "image": str(infer_cfg.wsi_path)},
    level=infer_cfg.preprocess.level,
    margin=infer_cfg.preprocess.margin,
    tile_size=tile_size,
    foreground_threshold=infer_cfg.preprocess.foreground_threshold,
    occupancy_threshold=infer_cfg.preprocess.occupancy_threshold,
    output_dir=tiles_root,
    overwrite=infer_cfg.preprocess.overwrite,
  )

  return tiles_root / slide_id / "images"


def _load_slide_patches(
  slide_images_dir: Path,
  transform: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  patch_paths = sorted(slide_images_dir.glob("*.png"))
  if not patch_paths:
    raise RuntimeError(f"No patches found under {slide_images_dir}")

  coord_re = re.compile(r"_X0Y0_(\d{6})_(\d{6})\.png$")
  imgs: list[torch.Tensor] = []
  coords: list[list[int]] = []

  for p in patch_paths:
    img = cv2.imread(str(p))
    if img is None:
      continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if transform:
      img_t = transform(image=img)["image"]
    else:
      img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # type: ignore

    if not isinstance(img_t, torch.Tensor):
      img_t = torch.from_numpy(img_t)
    imgs.append(img_t)

    match = coord_re.search(p.name)
    if not match:
      raise RuntimeError(f"Missing coordinate suffix in patch filename: {p.name}")
    coords.append([int(match.group(1)), int(match.group(2))])

  if not imgs:
    raise RuntimeError(f"All patches failed to load for slide dir: {slide_images_dir}")

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
  slide_images_dir = _generate_tiles(infer_cfg, slide_id=slide_id, tile_size=tile_size)

  _, val_tf = get_transforms(cfg)
  x, coords = _load_slide_patches(slide_images_dir, transform=val_tf)

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

  result = {
    "slide_id": slide_id,
    "wsi_path": str(infer_cfg.wsi_path),
    "num_patches": int(x.size(0)),
    "probabilities": probs,
    "predicted_class": pred,
    "weights_path": str(ckpt_path),
    "weights_source": source,
    "tiles_dir": str(slide_images_dir),
  }

  out_path = output_dir / infer_cfg.output_json
  with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
  log.info("Inference complete. Output: %s", out_path)

  if infer_cfg.cleanup_tiles:
    shutil.rmtree(slide_images_dir.parent, ignore_errors=True)

  return result


@hydra.main(version_base=None, config_path="conf", config_name="infer_wsi")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)
  run_wsi_inference(cfg, rr)


if __name__ == "__main__":
  main()
