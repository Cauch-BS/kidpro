from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import hydra
import numpy as np
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


def _original_cwd() -> Path:
  """
  Hydra changes the process CWD into the run dir.
  For repo-relative defaults, prefer Hydra's original working directory.
  """
  try:
    from hydra.utils import get_original_cwd

    return Path(get_original_cwd())
  except Exception:
    return Path.cwd()


def _resolve_fallback_weights(cfg: AppCfg, infer_cfg: InferenceCfg) -> Path:
  fallback = infer_cfg.fallback_weights
  if fallback is None:
    fallback = _original_cwd() / "models" / "best_model.pt"
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
      log.warning(
        "MLflow resolution failed; falling back to local weights. err=%s",
        e,
        exc_info=True,
      )
  fallback = _resolve_fallback_weights(cfg, infer_cfg)
  return fallback, "fallback"


def _derive_slide_id(infer_cfg: InferenceCfg) -> str:
  if infer_cfg.slide_id:
    return infer_cfg.slide_id
  return Path(infer_cfg.wsi_path).stem


def _resolve_cache_dir(cfg: AppCfg, infer_cfg: InferenceCfg) -> Path:
  if infer_cfg.cache_dir:
    return Path(infer_cfg.cache_dir)
  # Prefer WSI-specific cache location if provided.
  if cfg.dataset.paths.wsi_cache_dir:
    return Path(cfg.dataset.paths.wsi_cache_dir)
  if cfg.dataset.paths.cache_dir:
    return Path(cfg.dataset.paths.cache_dir)
  return Path(cfg.run_dir or Path.cwd()) / "wsidata_cache"


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

  process_slide(
    sample={"slide_id": slide_id, "image": str(infer_cfg.wsi_path)},
    level=infer_cfg.preprocess.level,
    tile_size=tile_size,
    cache_dir=cache_dir,
    tiles_key=infer_cfg.preprocess.tiles_key,
    overwrite=infer_cfg.preprocess.overwrite,
  )
  return cache_path


def _tile_to_chw_float_tensor(arr: Any, transform: Optional[Callable]) -> torch.Tensor:
  """
  Convert a tile image (typically HWC uint8 numpy) into a CHW float tensor.

  If a transform is provided, we accept either:
    - torch.Tensor (CHW or HWC)
    - numpy array (HWC or CHW)
  and normalize shape/dtype as best-effort.
  """
  if transform is not None:
    out = transform(image=arr)["image"]
  else:
    out = arr

  if isinstance(out, torch.Tensor):
    t = out
  else:
    # Albumentations may return numpy; wsidata returns numpy.
    t = torch.from_numpy(np.asarray(out))

  if t.ndim != 3:
    raise ValueError(f"Expected a 3D image tensor/array, got shape={tuple(t.shape)}")

  # If it's HWC, convert to CHW.
  if t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
    t = t.permute(2, 0, 1)

  # Ensure float in [0,1] when source is integer-like.
  if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
    t = t.float() / 255.0
  else:
    t = t.float()

  return t


def _iter_tiles_from_cache(
  slide_path: Path,
  cache_path: Path,
  tiles_key: str,
  transform: Optional[Callable],
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
  """
  Yield (tile_tensor, coords_xy) pairs from wsidata cache.
  - tile_tensor: (C,H,W) float32 tensor on CPU
  - coords_xy: (2,) float32 tensor on CPU in pixel space
  """
  wsi = open_wsidata(str(slide_path), cache_path)
  try:
    found = False
    for tile in wsi.iter.tile_images(tiles_key):
      found = True
      arr = tile_image_to_array(tile)
      img_t = _tile_to_chw_float_tensor(arr, transform=transform)
      tile_x, tile_y = extract_tile_xy(tile)
      coords_t = torch.tensor([tile_x, tile_y], dtype=torch.float32)
      yield img_t, coords_t
    if not found:
      raise RuntimeError(f"No tiles found in wsidata cache: {cache_path}")
  finally:
    if hasattr(wsi, "close"):
      wsi.close()

def _encode_tiles(model: Any, x: torch.Tensor) -> torch.Tensor:
  if hasattr(model, "tile_encoder"):
    encoder = getattr(model, "tile_encoder")
    for kw in ("x", "pixel_values", "inputs_embeds"):
      try:
        return cast(torch.Tensor, encoder(**{kw: x}))
      except TypeError:
        continue
    return cast(torch.Tensor, encoder(x))
  return cast(torch.Tensor, model.encode_tiles(x))

@torch.no_grad()
def _encode_slide_from_cache(
  *,
  model: torch.nn.Module,
  device: str,
  slide_path: Path,
  cache_path: Path,
  tiles_key: str,
  transform: Optional[Callable],
  batch_size: int,
  use_amp: bool,
) -> Tuple[torch.Tensor, int]:
  """
  Stream tiles from cache, compute tile embeddings in batches, then aggregate once.

  Returns:
    logits: (1, num_classes)
    num_tiles: number of tiles processed
  """
  # Encode tiles in batches to avoid holding all images in memory.
  feats_chunks: list[torch.Tensor] = []
  coords_chunks: list[torch.Tensor] = []

  batch_imgs: list[torch.Tensor] = []
  batch_coords: list[torch.Tensor] = []
  num_tiles = 0

  def flush_batch() -> None:
    nonlocal batch_imgs, batch_coords
    if not batch_imgs:
      return
    x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
    coords = torch.stack(batch_coords, dim=0).to(device, non_blocking=True)

    # model is a MILTemplate at runtime; use encode_tiles when available.
    if not hasattr(model, "encode_tiles"):
      raise TypeError("MIL model is expected to expose encode_tiles() for batched inference.")

    if use_amp:
      with torch.autocast(device_type="cuda"):
        feats = _encode_tiles(model, x)  # type: ignore[attr-defined, operator]
    else:
      feats = _encode_tiles(model, x)  # type: ignore[attr-defined, operator]

    # Keep embeddings on CPU to reduce GPU memory pressure.
    feats_chunks.append(feats.detach().to("cpu"))
    coords_chunks.append(coords.detach().to("cpu"))
    batch_imgs = []
    batch_coords = []

  for img_t, coords_t in _iter_tiles_from_cache(
    slide_path=slide_path,
    cache_path=cache_path,
    tiles_key=tiles_key,
    transform=transform,
  ):
    batch_imgs.append(img_t)
    batch_coords.append(coords_t)
    num_tiles += 1
    if len(batch_imgs) >= batch_size:
      flush_batch()

  flush_batch()

  feats_all = torch.cat(feats_chunks, dim=0)
  coords_all = torch.cat(coords_chunks, dim=0)

  feats_all = feats_all.to(device, non_blocking=True)
  coords_all = coords_all.to(device, non_blocking=True)

  if not hasattr(model, "encode_slide"):
    raise TypeError("MIL model is expected to expose encode_slide() for embedding aggregation.")

  if use_amp:
    with torch.autocast(device_type="cuda"):
      logits = model.encode_slide(feats_all, coords_all)  # type: ignore[attr-defined, operator]
  else:
    logits = model.encode_slide(feats_all, coords_all)  # type: ignore[attr-defined, operator]

  return logits, num_tiles


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

  model = build_model_mil(cfg)
  ckpt_path, source = _resolve_wsi_weights(cfg, infer_cfg)
  model = load_state_dict_generic(model, ckpt_path)
  model = model.to(rr.device)
  model.eval()

  use_amp = rr.device == "cuda"
  batch_size = int(getattr(infer_cfg, "batch_size", 64))

  logits, num_tiles = _encode_slide_from_cache(
    model=model,
    device=rr.device,
    slide_path=Path(infer_cfg.wsi_path),
    cache_path=cache_path,
    tiles_key=infer_cfg.preprocess.tiles_key,
    transform=val_tf,
    batch_size=batch_size,
    use_amp=use_amp,
  )

  # Shape-robust post-processing
  if logits.ndim == 1:
    logits_2d = logits.unsqueeze(0)
  else:
    logits_2d = logits
  probs_t = torch.softmax(logits_2d, dim=-1)
  probs = probs_t.squeeze(0).tolist()
  pred = int(torch.argmax(probs_t, dim=-1).item())

  result = {
    "slide_id": slide_id,
    "wsi_path": str(infer_cfg.wsi_path),
    "num_patches": int(num_tiles),
    "probabilities": probs,
    "predicted_class": pred,
    "weights_path": str(ckpt_path),
    "weights_source": source,
    "tiles_cache": str(cache_path),
  }

  out_path = output_dir / infer_cfg.output_json
  with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
  log.info("Inference complete. Output: %s", out_path)

  return result


@hydra.main(version_base=None, config_path="conf", config_name="infer_wsi")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)
  run_wsi_inference(cfg, rr)


if __name__ == "__main__":
  main()
