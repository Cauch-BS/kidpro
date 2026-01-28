from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence, Tuple

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from .config.load import CONFIG, RuntimeResolved, resolve_best_model_from_mlflow
from .config.schema import AppCfg
from .data.transform import get_transforms
from .modeling.factory_wsi import build_model_mil
from .modeling.sources import load_state_dict_generic
from .preprocessing.create_tiles_dataset import process_slide
from .utils.wsidata import extract_tile_xy, open_wsidata, tile_image_to_array

log = logging.getLogger(__name__)


def _original_cwd() -> Path:
  try:
    from hydra.utils import get_original_cwd

    return Path(get_original_cwd())
  except Exception:
    return Path.cwd()


def _resolve_cache_dir(cfg: AppCfg, infer: Dict[str, Any]) -> Path:
  if infer.get("cache_dir"):
    return Path(infer["cache_dir"])
  if cfg.dataset.paths.wsi_cache_dir:
    return Path(cfg.dataset.paths.wsi_cache_dir)
  if cfg.dataset.paths.cache_dir:
    return Path(cfg.dataset.paths.cache_dir)
  return Path(cfg.run_dir or Path.cwd()) / "wsidata_cache"


def _resolve_fallback_weights(infer: Dict[str, Any]) -> Path:
  fp = infer.get("fallback_weights")
  if fp is None:
    fp = _original_cwd() / "models" / "best_model.pt"
  fp = Path(fp)
  if not fp.exists():
    raise FileNotFoundError(
      f"Fallback weights not found at {fp}. "
      "Provide infer_ensem.fallback_weights or add models/best_model.pt."
    )
  return fp


def _resolve_weights_paths(cfg: AppCfg, infer: Dict[str, Any]) -> list[Path]:
  paths = infer.get("weights_paths") or []
  if isinstance(paths, (str, Path)):
    paths = [paths]
  out = [Path(p) for p in paths]
  if out:
    return out

  # Default: best from MLflow if enabled, else fallback.
  if cfg.mlflow.enabled:
    try:
      model_name = cfg.mlflow.registry_model_name
      ckpt_path = resolve_best_model_from_mlflow(cfg, model_name)
      return [Path(ckpt_path)]
    except Exception as e:
      log.warning(
        "MLflow resolution failed; falling back to local weights. err=%s",
        e,
        exc_info=True,
      )
  return [_resolve_fallback_weights(infer)]


def _ensure_cache_for_slide(
  cfg: AppCfg,
  infer: Dict[str, Any],
  *,
  slide_id: str,
  wsi_path: Path,
  tile_size: int,
) -> Path:
  cache_dir = _resolve_cache_dir(cfg, infer)
  cache_dir.mkdir(parents=True, exist_ok=True)
  cache_path = cache_dir / f"{slide_id}.zarr"

  ensure = bool(infer.get("ensure_cache", False))
  overwrite = bool(infer.get("overwrite_cache", False))
  tiles_key = str(infer.get("tiles_key", "tiles"))
  level = int(infer.get("level", 0))

  if cache_path.exists() and not overwrite:
    return cache_path
  if not ensure:
    raise RuntimeError(
      f"Missing wsidata cache for slide_id={slide_id}: {cache_path}. "
      "Set infer_ensem.ensure_cache=true to create caches on the fly."
    )

  process_slide(
    sample={"slide_id": slide_id, "image": str(wsi_path)},
    level=level,
    tile_size=tile_size,
    cache_dir=cache_dir,
    tiles_key=tiles_key,
    overwrite=overwrite,
  )
  return cache_path


def _iter_tiles_from_cache(
  *,
  slide_path: Path,
  cache_path: Path,
  tiles_key: str,
  transform: Any,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
  wsi = open_wsidata(str(slide_path), cache_path)
  try:
    found = False
    for tile in wsi.iter.tile_images(tiles_key):
      found = True
      arr = tile_image_to_array(tile)
      img = transform(image=arr)["image"]
      if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
      # Make best-effort CHW float
      if img.ndim == 3 and img.shape[0] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)
      img = img.float()
      tile_x, tile_y = extract_tile_xy(tile)
      coords = torch.tensor([tile_x, tile_y], dtype=torch.float32)
      yield img, coords
    if not found:
      raise RuntimeError(f"No tiles found in wsidata cache: {cache_path}")
  finally:
    if hasattr(wsi, "close"):
      wsi.close()


@torch.no_grad()
def _extract_feats_coords(
  *,
  model: Any,
  device: str,
  slide_path: Path,
  cache_path: Path,
  tiles_key: str,
  transform: Any,
  batch_size: int,
  use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  if not hasattr(model, "encode_tiles"):
    raise TypeError("MIL model must expose encode_tiles() for tile embedding extraction.")

  feats_chunks: list[torch.Tensor] = []
  coords_chunks: list[torch.Tensor] = []
  batch_imgs: list[torch.Tensor] = []
  batch_coords: list[torch.Tensor] = []
  n = 0

  def flush() -> None:
    nonlocal batch_imgs, batch_coords
    if not batch_imgs:
      return
    x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
    coords = torch.stack(batch_coords, dim=0).to(device, non_blocking=True)
    if use_amp:
      with torch.autocast(device_type="cuda"):
        feats = model.encode_tiles(x)
    else:
      feats = model.encode_tiles(x)
    feats_chunks.append(feats.detach().to("cpu"))
    coords_chunks.append(coords.detach().to("cpu"))
    batch_imgs = []
    batch_coords = []

  for img, coord in _iter_tiles_from_cache(
    slide_path=slide_path,
    cache_path=cache_path,
    tiles_key=tiles_key,
    transform=transform,
  ):
    batch_imgs.append(img)
    batch_coords.append(coord)
    n += 1
    if len(batch_imgs) >= batch_size:
      flush()
  flush()

  feats_all = torch.cat(feats_chunks, dim=0)
  coords_all = torch.cat(coords_chunks, dim=0)
  return feats_all, coords_all, n


@torch.no_grad()
def _predict_probs_from_feats(
  *,
  model: Any,
  device: str,
  feats: torch.Tensor,
  coords: torch.Tensor,
  use_amp: bool,
) -> torch.Tensor:
  if not hasattr(model, "encode_slide"):
    raise TypeError("MIL model must expose encode_slide() for slide aggregation.")
  feats_d = feats.to(device, non_blocking=True)
  coords_d = coords.to(device, non_blocking=True)

  if use_amp:
    with torch.autocast(device_type="cuda"):
      logits = model.encode_slide(feats_d, coords_d)
  else:
    logits = model.encode_slide(feats_d, coords_d)

  if logits.ndim == 1:
    logits = logits.unsqueeze(0)
  probs = torch.softmax(logits, dim=-1).squeeze(0).detach().to("cpu")
  return probs


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
  # Avoid importing sklearn (optional dependency in some envs)
  try:
    from sklearn.metrics import f1_score

    return float(f1_score(list(y_true), list(y_pred), average="macro"))
  except Exception:
    # Minimal macro-F1 for integer labels.
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
      tp = sum((yt == lab) and (yp == lab) for yt, yp in zip(y_true, y_pred))
      fp = sum((yt != lab) and (yp == lab) for yt, yp in zip(y_true, y_pred))
      fn = sum((yt == lab) and (yp != lab) for yt, yp in zip(y_true, y_pred))
      prec = tp / (tp + fp) if (tp + fp) else 0.0
      rec = tp / (tp + fn) if (tp + fn) else 0.0
      f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
      f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


def _roc_auc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
  try:
    from sklearn.metrics import roc_auc_score

    # roc_auc_score errors if only one class present
    if len(set(y_true)) < 2:
      return float("nan")
    return float(roc_auc_score(list(y_true), list(y_score)))
  except Exception:
    return float("nan")


def run_csv_ensem_inference(cfg: AppCfg, rr: RuntimeResolved, infer: Dict[str, Any]) -> Dict[str, Any]:
  csv_path = Path(infer["csv_path"])
  df = pd.read_csv(csv_path)

  slide_id_col = str(infer.get("slide_id_col", "SlideName"))
  gt_col = str(infer.get("gt_col", "GT"))
  split_col = str(infer.get("split_col", "ID"))
  test_value = str(infer.get("test_value", "test"))
  wsi_path_col = infer.get("wsi_path_col", None)

  if slide_id_col not in df.columns:
    raise ValueError(f"CSV missing slide_id_col={slide_id_col!r}. Columns={list(df.columns)}")
  if gt_col not in df.columns:
    log.warning("CSV missing gt_col=%r; metrics will be skipped.", gt_col)
  if split_col not in df.columns:
    log.warning("CSV missing split_col=%r; output will include all rows.", split_col)

  output_dir = Path(infer.get("output_dir") or cfg.run_dir or Path.cwd())
  output_dir.mkdir(parents=True, exist_ok=True)
  output_csv = output_dir / str(infer.get("output_csv", "submission.csv"))

  tile_size = int(infer.get("tile_size") or cfg.dataset.data.patch_size)
  tiles_key = str(infer.get("tiles_key", "tiles"))
  batch_size = int(infer.get("batch_size", 64))
  threshold = float(infer.get("threshold", 0.5))
  share_tile_encoder = bool(infer.get("share_tile_encoder", True))

  amp_cfg = str(infer.get("amp", "auto")).lower()
  use_amp = (rr.device == "cuda") if amp_cfg == "auto" else (amp_cfg == "true")

  weights_paths = _resolve_weights_paths(cfg, infer)
  log.info("[infer_ensem] weights=%s", [str(p) for p in weights_paths])

  # Build and load models once
  models = []
  for wp in weights_paths:
    m = build_model_mil(cfg)
    m = load_state_dict_generic(m, wp)
    m = m.to(rr.device)
    m.eval()
    models.append(m)

  _, val_tf = get_transforms(cfg)

  out_rows: list[dict[str, Any]] = []
  y_true: list[int] = []
  y_pred: list[int] = []
  y_score: list[float] = []

  for _, row in df.iterrows():
    slide_id = str(row[slide_id_col])
    if wsi_path_col and wsi_path_col in df.columns and pd.notna(row[wsi_path_col]):
      slide_path = Path(str(row[wsi_path_col]))
    else:
      if cfg.dataset.paths.wsi_dir is None:
        raise RuntimeError("dataset.paths.wsi_dir is required when wsi_path_col is not provided.")
      slide_path = Path(cfg.dataset.paths.wsi_dir) / f"{slide_id}{cfg.dataset.paths.wsi_ext}"

    cache_path = _ensure_cache_for_slide(
      cfg,
      infer,
      slide_id=slide_id,
      wsi_path=slide_path,
      tile_size=tile_size,
    )

    # Extract embeddings once (optionally shared across ensemble members)
    feats, coords, _n = _extract_feats_coords(
      model=models[0],
      device=rr.device,
      slide_path=slide_path,
      cache_path=cache_path,
      tiles_key=tiles_key,
      transform=val_tf,
      batch_size=batch_size,
      use_amp=use_amp,
    )

    probs_accum = None
    for mi, m in enumerate(models):
      if not share_tile_encoder and mi > 0:
        feats, coords, _n = _extract_feats_coords(
          model=m,
          device=rr.device,
          slide_path=slide_path,
          cache_path=cache_path,
          tiles_key=tiles_key,
          transform=val_tf,
          batch_size=batch_size,
          use_amp=use_amp,
        )
      probs = _predict_probs_from_feats(model=m, device=rr.device, feats=feats, coords=coords, use_amp=use_amp)
      probs_accum = probs if probs_accum is None else (probs_accum + probs)

    probs_mean = probs_accum / float(len(models))  # type: ignore[operator]
    pred = int(torch.argmax(probs_mean).item()) # type: ignore
    pred_prob = float(probs_mean[1].item()) if probs_mean.numel() > 1 else float(probs_mean[0].item()) # type: ignore

    gt_val = None
    if gt_col in df.columns and pd.notna(row.get(gt_col)):
      gt_val = int(row[gt_col])
      y_true.append(gt_val)
      y_pred.append(pred)
      y_score.append(pred_prob)

    split_val = str(row[split_col]) if (split_col in df.columns and pd.notna(row.get(split_col))) else None
    out_rows.append(
      {
        "ID": slide_id,
        "Predicted_Label": pred,
        "Predicted_Prob": pred_prob,
        "_GT": gt_val,
        "_split": split_val,
      }
    )

  # Metrics on rows with GT
  if y_true:
    macro_f1 = _macro_f1(y_true, y_pred)
    roc_auc = _roc_auc_binary(y_true, y_score)
  else:
    macro_f1 = float("nan")
    roc_auc = float("nan")

  print(f"macro_f1: {macro_f1}" if macro_f1 == macro_f1 else "macro_f1: n/a")
  print(f"ROC_AUC: {roc_auc}" if roc_auc == roc_auc else "ROC_AUC: n/a")

  # Output CSV for test rows (split_col == test_value). If split_col missing, export all.
  out_df = pd.DataFrame(out_rows)
  if split_col in df.columns:
    out_df = out_df[out_df["_split"].astype(str) == test_value].copy()
  out_df = out_df[["ID", "Predicted_Label", "Predicted_Prob"]]
  out_df.to_csv(output_csv, index=False)
  log.info("[infer_ensem] wrote: %s (%d rows)", output_csv, len(out_df))

  result = {
    "csv_path": str(csv_path),
    "output_csv": str(output_csv),
    "num_rows": int(len(df)),
    "num_scored": int(len(y_true)),
    "macro_f1": macro_f1,
    "roc_auc": roc_auc,
    "weights_paths": [str(p) for p in weights_paths],
    "share_tile_encoder": share_tile_encoder,
  }
  with open(output_dir / "metrics.json", "w") as f:
    json.dump(result, f, indent=2)
  return result


@hydra.main(version_base=None, config_path="conf", config_name="infer_ensem")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)

  infer = dict(hcfg.get("infer_ensem", {}))
  if "csv_path" not in infer:
    raise ValueError("Missing infer_ensem.csv_path in config.")

  run_csv_ensem_inference(cfg, rr, infer)


if __name__ == "__main__":
  main()
