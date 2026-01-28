from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from .config.load import CONFIG, RuntimeResolved, resolve_best_model_from_mlflow
from .config.schema import AppCfg
from .data.dataset_mil import MILDataset
from .modeling.factory_wsi import build_model_mil
from .utils.model_io import load_state_dict_generic

log = logging.getLogger(__name__)


def _original_cwd() -> Path:
  try:
    from hydra.utils import get_original_cwd

    return Path(get_original_cwd())
  except Exception:
    return Path.cwd()


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


def _resolve_weight_path(cfg: AppCfg, infer: Dict[str, Any]) -> Path:
  # Preferred: singular config key.
  weights_path = infer.get("weights_path")
  if weights_path:
    return Path(weights_path)

  # Back-compat: accept weights_paths but take the first.
  paths = infer.get("weights_paths") or []
  if isinstance(paths, (str, Path)):
    paths = [paths]
  if paths:
    log.warning("infer_ensem.weights_paths is deprecated; use infer_ensem.weights_path instead.")
    return Path(paths[0])

  # Default: best from MLflow if enabled, else fallback.
  if cfg.mlflow.enabled:
    try:
      model_name = cfg.mlflow.registry_model_name
      ckpt_path = resolve_best_model_from_mlflow(cfg, model_name)
      return Path(ckpt_path)
    except Exception as e:
      log.warning(
        "MLflow resolution failed; falling back to local weights. err=%s",
        e,
        exc_info=True,
      )
  return _resolve_fallback_weights(infer)


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
  split_col = str(infer.get("split_col", "split"))
  test_value = str(infer.get("test_value", "test"))

  if slide_id_col not in df.columns:
    raise ValueError(f"CSV missing slide_id_col={slide_id_col!r}. Columns={list(df.columns)}")
  if gt_col not in df.columns:
    log.warning("CSV missing gt_col=%r; metrics will be skipped.", gt_col)
  if split_col not in df.columns:
    log.warning("CSV missing split_col=%r; output will include all rows.", split_col)

  output_dir = Path(infer.get("output_dir") or cfg.run_dir or Path.cwd())
  output_dir.mkdir(parents=True, exist_ok=True)
  output_csv = output_dir / str(infer.get("output_csv", "submission.csv"))

  threshold = float(infer.get("threshold", 0.5))
  require_cached = bool(infer.get("require_cached_embeddings", True))

  if not (cfg.dataset.data.mil_cache.enabled and cfg.dataset.data.mil_cache.cache_tile_embeddings):
    raise RuntimeError(
      "infer_ensem requires cached MIL tile embeddings but "
      "dataset.data.mil_cache.enabled=false or cache_tile_embeddings=false. "
      "Enable them in dataset config (e.g. conf/dataset/wsi.yaml)."
    )
  if cfg.dataset.paths.cache_dir is None:
    raise RuntimeError(
      "infer_ensem requires dataset.paths.cache_dir to locate mil_embeds_cache."
    )

  amp_cfg = str(infer.get("amp", "auto")).lower()
  use_amp = (rr.device == "cuda") if amp_cfg == "auto" else (amp_cfg == "true")

  weights_path = _resolve_weight_path(cfg, infer)
  log.info("[infer_ensem] weights=%s", str(weights_path))

  # Build and load model once.
  model = build_model_mil(cfg)
  model = load_state_dict_generic(model, weights_path)
  model = model.to(rr.device)
  model.eval()

  # Create a minimal MIL dataframe so we can reuse MILDataset's embedding-cache reader.
  df_mil = pd.DataFrame(
    {
      "SlideName": df[slide_id_col].astype(str),
      "GT": df[gt_col] if gt_col in df.columns else 0,
      "split": df[split_col] if split_col in df.columns else test_value,
    }
  )
  # Ensure GT is non-null to satisfy MILDataset invariants (metrics are handled separately).
  df_mil["GT"] = pd.to_numeric(df_mil["GT"], errors="coerce").fillna(0).astype(int)
  ds = MILDataset(cfg, df_mil, transform=None)

  out_rows: list[dict[str, Any]] = []
  y_true: list[int] = []
  y_pred: list[int] = []
  y_score: list[float] = []

  processed = 0
  cache_miss = 0
  failed = 0

  pbar = tqdm(range(len(df)), desc="[infer_ensem]", unit="slide")
  for idx in pbar:
    row = df.iloc[idx]
    slide_id = str(row[slide_id_col])

    try:
      tile_stream, _y_dummy, _slide_name = ds[idx]
      cached = tile_stream.get_cached_tile_embeddings()
      if cached is None:
        cache_miss += 1
        if require_cached:
          raise RuntimeError(
            f"Missing cached tile embeddings for slide={slide_id}. "
            "Populate mil_embeds_cache by running MIL training once with "
            "data.mil_cache.enabled=true and data.mil_cache.cache_tile_embeddings=true "
            "(the training loop writes caches on cache-miss), or set "
            "infer_ensem.require_cached_embeddings=false."
          )
        # If we ever support fallback, it would go here.
        continue

      emb_np, coords_np = cached
      feats = torch.from_numpy(np.asarray(emb_np, dtype=np.float32))
      coords = torch.from_numpy(np.asarray(coords_np, dtype=np.float32))

      probs = _predict_probs_from_feats(model=model, device=rr.device, feats=feats, coords=coords, use_amp=use_amp)

      pred = int(torch.argmax(probs).item())
      pred_prob = float(probs[1].item()) if probs.numel() > 1 else float(probs[0].item())
      # Keep threshold for downstream consumers; for binary, this can override argmax if desired.
      if probs.numel() == 2:
        pred = int(pred_prob >= threshold)
    except Exception as exc:
      failed += 1
      log.error("[infer_ensem] slide=%s failed: %s", slide_id, exc, exc_info=True)
      pbar.set_postfix(processed=processed, cache_miss=cache_miss, failed=failed)
      continue

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
    processed += 1
    pbar.set_postfix(processed=processed, cache_miss=cache_miss, failed=failed)

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
    "weights_path": str(weights_path),
    "require_cached_embeddings": require_cached,
    "processed": processed,
    "cache_miss": cache_miss,
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
