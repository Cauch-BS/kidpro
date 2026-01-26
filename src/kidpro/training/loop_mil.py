from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# FIXED: Graceful fallback for torch.amp (PyTorch >= 1.9)
try:
  from torch.amp import GradScaler, autocast
  HAS_AMP = True
except ImportError:
  try:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    HAS_AMP = True
  except ImportError:
    HAS_AMP = False
    GradScaler = None  # type: ignore
    autocast = None  # type: ignore

from ..config.load import RuntimeResolved
from ..config.schema import AppCfg
from .early_stop import EarlyStopping

log = logging.getLogger(__name__)


def _find_best_threshold(
  y_true: list[int],
  y_prob: list[float],
  num_thresholds: int = 101,
) -> tuple[float, float]:
  """
  Sweep thresholds to find the one that maximizes macro F1.

  Args:
    y_true: Ground truth labels (0 or 1)
    y_prob: Predicted probabilities for positive class
    num_thresholds: Number of threshold values to try

  Returns:
    Tuple of (best_threshold, best_f1_score)
  """
  thresholds = np.linspace(0, 1, num_thresholds)
  best_thr, best_f1 = 0.5, 0.0
  y_true_arr = np.array(y_true)
  y_prob_arr = np.array(y_prob)

  for thr in thresholds:
    preds = (y_prob_arr >= thr).astype(int)
    f1 = f1_score(y_true_arr, preds, average="macro", zero_division=0)
    if f1 > best_f1:
      best_f1, best_thr = f1, float(thr)

  return best_thr, best_f1


def _is_skippable_tile_error(exc: RuntimeError) -> bool:
  msg = str(exc).lower()
  return (
    "no tiles available" in msg
    or "empty tile stream" in msg
    or "too many unreadable tiles" in msg
  )


def _unpack_mil_batch(
  batch: tuple[Any, ...],
) -> tuple[Any, torch.Tensor, torch.Tensor | None, str]:
  if len(batch) == 3:
    x, y, slide = batch
    return x, y, None, str(slide)
  if len(batch) == 4:
    x, y, coords, slide = batch
    return x, y, coords, str(slide)
  raise ValueError(f"Unexpected MIL batch format with {len(batch)} items.")


def _unwrap_singleton(value: Any) -> Any:
  if isinstance(value, list) and len(value) == 1:
    return value[0]
  return value


def _stream_slide_logits(
  model: nn.Module,
  tile_stream: Any,
  rr: RuntimeResolved,
  cfg: AppCfg,
  use_amp: bool,
  asynchrony: bool,
) -> tuple[torch.Tensor, int]:
  """
  Stream tiles through the model to get slide-level logits.

  Caches TILE EMBEDDINGS (output of tile_encoder) so that:
  - tile_encoder runs only once per slide (expensive, frozen)
  - slide_encoder + classifier run every epoch (trainable)
  """
  if not hasattr(tile_stream, "iter_batches"):
    raise ValueError("Expected a tile stream with iter_batches().")

  encode_tiles = getattr(model, "encode_tiles", None)
  encode_slide = getattr(model, "encode_slide", None)
  encode_slide_embedding = getattr(model, "encode_slide_embedding", None)

  # Check for cached TILE embeddings (not slide embeddings)
  cached_tile_emb = None
  if hasattr(tile_stream, "get_cached_tile_embeddings"):
    cached_tile_emb = tile_stream.get_cached_tile_embeddings()

  if cached_tile_emb is not None:
    # Load cached tile embeddings - skip tile_encoder
    feats_np, coords_np = cached_tile_emb
    feats_all = torch.from_numpy(feats_np).to(rr.device)
    coords_all = torch.from_numpy(coords_np).to(rr.device)
    tile_count = feats_all.shape[0]
  else:
    # Run tile_encoder and cache results
    chunk_size = cfg.dataset.data.mil_cache.chunk_size
    feats_list: list[torch.Tensor] = []
    coords_list: list[torch.Tensor] = []
    tile_count = 0

    for tiles, coords in tile_stream.iter_batches(chunk_size):
      tiles = tiles.to(rr.device, non_blocking=asynchrony)
      coords = coords.to(rr.device, non_blocking=asynchrony)
      tile_count += int(tiles.size(0))

      if use_amp and autocast is not None:
        with autocast(device_type="cuda"):
          feats = encode_tiles(tiles) if callable(encode_tiles) else model.tile_encoder(tiles)  # type: ignore
      else:
        feats = encode_tiles(tiles) if callable(encode_tiles) else model.tile_encoder(tiles)  # type: ignore

      feats_list.append(feats)
      coords_list.append(coords)

    if tile_count == 0:
      raise RuntimeError("Empty tile stream for slide.")

    feats_all = torch.cat(feats_list, dim=0)
    coords_all = torch.cat(coords_list, dim=0)

    # Cache tile embeddings for next epoch
    if hasattr(tile_stream, "set_cached_tile_embeddings"):
      try:
        tile_stream.set_cached_tile_embeddings(
          feats_all.detach().cpu().numpy(),
          coords_all.detach().cpu().numpy(),
        )
      except Exception as exc:
        log.warning("Failed to cache tile embeddings: %s", exc)

  # ALWAYS run slide_encoder + classifier (these are trainable)
  if use_amp and autocast is not None:
    with autocast(device_type="cuda"):
      if callable(encode_slide_embedding) and hasattr(model, "classify_slide_embedding"):
        emb = encode_slide_embedding(feats_all, coords_all)
        logits = model.classify_slide_embedding(emb)  # type: ignore
      else:
        logits = encode_slide(feats_all, coords_all) if callable(encode_slide) else model(feats_all, coords_all)
  else:
    if callable(encode_slide_embedding) and hasattr(model, "classify_slide_embedding"):
      emb = encode_slide_embedding(feats_all, coords_all)
      logits = model.classify_slide_embedding(emb)  # type: ignore
    else:
      logits = encode_slide(feats_all, coords_all) if callable(encode_slide) else model(feats_all, coords_all)

  return logits, tile_count


@torch.no_grad()
def evaluate_mil(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,
  loader: DataLoader,
  skipped_slides: Optional[set[str]] = None,
) -> Dict[str, Any]:
  """
  Evaluate MIL model on a validation/test set.

  Returns:
    Dictionary containing:
      - "acc": float (accuracy)
      - "macro_f1": float (macro F1 score)
      - "auc": Optional[float] (ROC AUC, None if only one class present)
      - "cm": np.ndarray of shape (2,2) (confusion matrix)
  """
  model.eval()
  use_amp = (rr.device == "cuda" and HAS_AMP)

  y_true: list[int] = []
  y_prob: list[float] = []
  y_pred: list[int] = []

  if skipped_slides is None:
    skipped_slides = set()

  for batch in tqdm(loader, desc="Eval", leave=False):
    x, y, coords, _slide = _unpack_mil_batch(batch)
    x = _unwrap_singleton(x)
    y = y.to(rr.device, non_blocking=True)  # (1,)

    if hasattr(x, "iter_batches"):
      try:
        logits, _tile_count = _stream_slide_logits(model, x, rr, cfg, use_amp, asynchrony=True)
      except RuntimeError as exc:
        if _is_skippable_tile_error(exc):
          slide_name = str(_slide)
          if slide_name not in skipped_slides:
            log.warning("Skipping slide %s during eval: %s", slide_name, exc)
            skipped_slides.add(slide_name)
          continue
        raise RuntimeError(f"Error during evaluation: {exc}")
    else:
      x = x.squeeze(0).to(rr.device, non_blocking=True)  # (N,C,H,W)
      coords_t = coords.squeeze(0).to(rr.device, non_blocking=True) if coords is not None else None
      if use_amp and autocast is not None:
        with autocast(device_type="cuda"):
          logits = model(x, coords_t)  # (1,2)
      else:
        logits = model(x, coords_t)

    prob = torch.softmax(logits, dim=1)[:, 1]
    pred = torch.argmax(logits, dim=1)

    y_true.append(int(y.item()))
    y_prob.append(float(prob.item()))
    y_pred.append(int(pred.item()))

  # FIXED: Handle empty validation set
  if not y_true:
    log.warning("[WARN] Empty validation set in evaluate_mil")
    return {
      "acc": 0.0,
      "macro_f1": 0.0,
      "auc": None,
      "cm": np.zeros((2, 2), dtype=int),
      "true_counts": {0: 0, 1: 0},
      "pred_counts": {0: 0, 1: 0},
      "pred_pos_rate": 0.0,
      "true_pos_rate": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "best_thr": 0.5,
      "f1_at_best_thr": 0.0,
      "f1_at_0.5": 0.0,
    }

  y_true_arr = np.array(y_true)
  y_pred_arr = np.array(y_pred)
  y_prob_arr = np.array(y_prob)

  # Basic metrics
  acc = accuracy_score(y_true, y_pred)
  macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

  # AUC (only if both classes present)
  auc: Optional[float] = None
  if len(set(y_true)) > 1:
    auc = float(roc_auc_score(y_true, y_prob))

  # Confusion matrix and counts
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
  true_counts = {0: int(np.sum(y_true_arr == 0)), 1: int(np.sum(y_true_arr == 1))}
  pred_counts = {0: int(np.sum(y_pred_arr == 0)), 1: int(np.sum(y_pred_arr == 1))}

  # Positive rates
  pred_pos_rate = float(np.mean(y_pred_arr == 1))
  true_pos_rate = float(np.mean(y_true_arr == 1))

  # Precision/recall for positive class
  precision = float(precision_score(y_true, y_pred, zero_division=0))
  recall = float(recall_score(y_true, y_pred, zero_division=0))

  # Threshold tuning
  best_thr, f1_at_best_thr = _find_best_threshold(y_true, y_prob)

  # F1 at fixed threshold 0.5
  preds_at_05 = (y_prob_arr >= 0.5).astype(int)
  f1_at_05 = float(f1_score(y_true, preds_at_05, average="macro", zero_division=0))

  return {
    "acc": float(acc),
    "macro_f1": float(macro_f1),
    "auc": auc,
    "cm": cm,
    "true_counts": true_counts,
    "pred_counts": pred_counts,
    "pred_pos_rate": pred_pos_rate,
    "true_pos_rate": true_pos_rate,
    "precision": precision,
    "recall": recall,
    "best_thr": best_thr,
    "f1_at_best_thr": f1_at_best_thr,
    "f1_at_0.5": f1_at_05,
  }


def fit_mil(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  criterion: nn.Module,
  optimizer: optim.Optimizer,
  scheduler: Optional[Any] = None,
) -> Path:
  """
  Train MIL model with early stopping and checkpointing on val_auc.

  Args:
    cfg: Application configuration
    rr: Runtime resolution (device, etc.)
    model: MIL model to train
    train_loader: Training data loader
    val_loader: Validation data loader
    criterion: Loss function
    optimizer: Optimizer
    scheduler: Optional learning rate scheduler

  Returns:
    Path to best model checkpoint
  """
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()
  best_path = run_dir / cfg.core.export.best_weights_name

  # FIXED: Proper amp handling
  use_amp = (rr.device == "cuda" and HAS_AMP)
  scaler = GradScaler(device="cuda", enabled=use_amp) if use_amp and GradScaler is not None else None

  # Early stopping with configurable metric
  es_metric = cfg.train.early_stopping.metric
  es_mode = "max" if es_metric in ("val_auc", "val_macro_f1") else "min"
  stopper = EarlyStopping(
    patience=cfg.train.early_stopping.patience,
    min_delta=cfg.train.early_stopping.min_delta,
    mode=es_mode,
  )
  log.info("[EARLY STOPPING] metric=%s, mode=%s, patience=%d", es_metric, es_mode, cfg.train.early_stopping.patience)

  # Gradient clipping
  gradient_clip = cfg.train.gradient_clip

  best_val_auc: float = -math.inf
  best_epoch: int = -1  # 1-based when reported

  asynchrony: bool = cfg.dataset.data.pin_memory
  skipped_train_slides: set[str] = set()
  skipped_val_slides: set[str] = set()
  skipped_eval_slides: set[str] = set()

  # For sanity check mode, limit epochs
  epochs = 1 if cfg.train.sanity_check else cfg.train.epochs

  for epoch in range(epochs):
    # -------------------------
    # Train
    # -------------------------
    model.train()
    train_losses: list[float] = []
    pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)

    for batch in pbar:
      x, y, coords, _slide = _unpack_mil_batch(batch)
      x = _unwrap_singleton(x)
      y = y.to(rr.device, non_blocking=asynchrony)             # (1,)

      optimizer.zero_grad(set_to_none=True)

      if hasattr(x, "iter_batches"):
        try:
          logits, tile_count = _stream_slide_logits(model, x, rr, cfg, use_amp, asynchrony)
        except RuntimeError as exc:
          if _is_skippable_tile_error(exc):
            slide_name = str(_slide)
            if slide_name not in skipped_train_slides:
              log.warning("Skipping slide %s during training: %s", slide_name, exc)
              skipped_train_slides.add(slide_name)
            continue
          raise RuntimeError(f"Error during training: {exc}")
        loss = criterion(logits, y)
      else:
        x = x.squeeze(0).to(rr.device, non_blocking=asynchrony)  # (N,C,H,W)
        coords_t = coords.squeeze(0).to(rr.device, non_blocking=asynchrony) if coords is not None else None
        tile_count = int(x.size(0))
        if use_amp and autocast is not None:
          with autocast(device_type="cuda"):
            logits = model(x, coords_t)  # (1,2)
            loss = criterion(logits, y)
        else:
          logits = model(x, coords_t)
          loss = criterion(logits, y)

      if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        if gradient_clip > 0:
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        if gradient_clip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

      # Step scheduler per batch
      if scheduler is not None:
        scheduler.step()

      train_losses.append(float(loss.item()))
      current_lr = optimizer.param_groups[0]["lr"]
      pbar.set_postfix(train_loss=f"{loss.item():.4f}", n_patches=tile_count, lr=f"{current_lr:.2e}")

    train_loss = float(np.mean(train_losses)) if train_losses else 0.0

    # -------------------------
    # Val loss
    # -------------------------
    model.eval()
    val_losses: list[float] = []
    with torch.no_grad():
      for batch in tqdm(val_loader, desc="ValLoss", leave=False):
        x, y, coords, _slide = _unpack_mil_batch(batch)
        x = _unwrap_singleton(x)
        y = y.to(rr.device, non_blocking=asynchrony)

        if hasattr(x, "iter_batches"):
          try:
            logits, _tile_count = _stream_slide_logits(model, x, rr, cfg, use_amp, asynchrony)
          except RuntimeError as exc:
            if _is_skippable_tile_error(exc):
              slide_name = str(_slide)
              if slide_name not in skipped_val_slides:
                log.warning("Skipping slide %s during val loss: %s", slide_name, exc)
                skipped_val_slides.add(slide_name)
              continue
            raise RuntimeError(f"Error during validation: {exc}")
          loss = criterion(logits, y)
        else:
          x = x.squeeze(0).to(rr.device, non_blocking=asynchrony)
          coords_t = coords.squeeze(0).to(rr.device, non_blocking=asynchrony) if coords is not None else None
          if use_amp and autocast is not None:
            with autocast(device_type="cuda"):
              logits = model(x, coords_t)
              loss = criterion(logits, y)
          else:
            logits = model(x, coords_t)
            loss = criterion(logits, y)

        val_losses.append(float(loss.item()))

    val_loss = float(np.mean(val_losses)) if val_losses else 0.0

    # -------------------------
    # Val metrics
    # -------------------------
    metrics = evaluate_mil(cfg, rr, model, val_loader, skipped_slides=skipped_eval_slides)
    auc = metrics.get("auc", None)
    val_auc = float(auc) if isinstance(auc, (float, int)) else -math.inf
    auc_str = f"{val_auc:.4f}" if math.isfinite(val_auc) else "None"

    # -------------------------
    # Checkpointing: maximize val_auc
    # -------------------------
    if val_auc > best_val_auc:
      best_val_auc = val_auc
      best_epoch = epoch + 1  # 1-based
      if cfg.core.export.save_best_weights:
        torch.save(model.state_dict(), best_path)

    # -------------------------
    # Logging
    # -------------------------
    best_auc_str = f"{best_val_auc:.4f}" if math.isfinite(best_val_auc) else "None"
    current_lr = optimizer.param_groups[0]["lr"]
    log.info(
      f"Epoch {epoch+1}/{epochs} | "
      f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
      f"val_acc={metrics['acc']:.4f} | val_macro_f1={metrics['macro_f1']:.4f} | val_auc={auc_str} | "
      f"best_val_auc={best_auc_str} | best_epoch={best_epoch} | "
      f"patience={stopper.counter}/{stopper.patience} | lr={current_lr:.2e}"
    )
    # Enhanced diagnostics
    true_counts = metrics.get("true_counts")
    pred_counts = metrics.get("pred_counts")
    if isinstance(true_counts, dict) and isinstance(pred_counts, dict):
      log.info(
        "Val true_counts=%s | pred_counts=%s | pred_pos_rate=%.3f | true_pos_rate=%.3f",
        true_counts, pred_counts, metrics["pred_pos_rate"], metrics["true_pos_rate"]
      )
      log.info(
        "Val precision=%.4f | recall=%.4f | f1@0.5=%.4f | best_thr=%.3f | f1@best_thr=%.4f",
        metrics["precision"], metrics["recall"], metrics["f1_at_0.5"],
        metrics["best_thr"], metrics["f1_at_best_thr"]
      )
      log.info("Val confusion_matrix:\n%s", metrics["cm"])

    # -------------------------
    # Early stopping with configurable metric
    # -------------------------
    if es_metric == "val_auc":
      stopper.step(val_auc if math.isfinite(val_auc) else -math.inf)
    elif es_metric == "val_macro_f1":
      stopper.step(metrics["macro_f1"])
    else:  # val_loss
      stopper.step(val_loss)

    if stopper.early_stop:
      log.info("[Early Stop] Training stopped (%s criterion).", es_metric)
      break

  # Load best weights (by val_auc)
  if cfg.core.export.save_best_weights and best_path.exists():
    log.info(f"[DONE] Best model saved to {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=rr.device))

  # Persist best summary
  summary = {
    "best_epoch": int(best_epoch),
    "best_val_auc": None if not math.isfinite(best_val_auc) else float(best_val_auc),
    "best_weights_path": str(best_path) if best_path.exists() else None,
  }
  try:
    with open(run_dir / "best_summary.json", "w") as f:
      json.dump(summary, f, indent=2)
  except Exception as e:
    log.warning(f"[WARN] Failed to write best_summary.json: {e}")

  return best_path
