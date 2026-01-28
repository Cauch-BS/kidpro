from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
from .rankmix import RankMixSampler, TileScorer, compute_rankmix_loss, rankmix

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

def _as_mil_samples(batch: Any) -> list[tuple[Any, ...]]:
  """
  Normalize dataloader output into a list of per-slide samples.

  - With a standard (batch_size=1) MIL collate, a batch is a single tuple.
  - With our MIL collate for batch_size>1, a batch is a list[tuple].
  """
  if isinstance(batch, list):
    return cast(list[tuple[Any, ...]], batch)
  if isinstance(batch, tuple):
    return [batch]
  raise ValueError(f"Unexpected MIL batch type: {type(batch)}")




def _get_tile_embeddings(
  model: nn.Module,
  tile_stream: Any,
  rr: RuntimeResolved,
  cfg: AppCfg,
  use_amp: bool,
  asynchrony: bool,
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
  """
  Extract tile embeddings from a slide without running the aggregator.

  This is useful for RankMix where we need embeddings from multiple slides
  before mixing them and passing through the aggregator.

  Returns:
    Tuple of (tile_embeddings, tile_coords, tile_count, cache_hit)
  """
  if not hasattr(tile_stream, "iter_batches"):
    raise ValueError("Expected a tile stream with iter_batches().")

  encode_tiles = getattr(model, "encode_tiles", None)

  # Check for cached TILE embeddings
  cached_tile_emb = None
  if hasattr(tile_stream, "get_cached_tile_embeddings"):
    cached_tile_emb = tile_stream.get_cached_tile_embeddings()

  if cached_tile_emb is not None:
    # Load cached tile embeddings - skip tile_encoder
    feats_np, coords_np = cached_tile_emb
    cache_hit = True
    feats_all = torch.from_numpy(feats_np).to(rr.device)
    coords_all = torch.from_numpy(coords_np).to(rr.device)
    tile_count = feats_all.shape[0]
  else:
    cache_hit = False
    # Run tile_encoder and cache results (cache miss)
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
      tile_stream.set_cached_tile_embeddings(
        feats_all.detach().cpu().numpy(),
        coords_all.detach().cpu().numpy(),
      )

  return feats_all, coords_all, tile_count, cache_hit


def _embeddings_to_logits(
  model: nn.Module,
  feats_all: torch.Tensor,
  coords_all: torch.Tensor,
  use_amp: bool,
) -> torch.Tensor:
  """
  Run aggregator and classifier on tile embeddings to get logits.

  Args:
    model: MIL model with encode_slide or encode_slide_embedding + classify_slide_embedding
    feats_all: Tile embeddings of shape (N, D)
    coords_all: Tile coordinates of shape (N, 2)
    use_amp: Whether to use automatic mixed precision

  Returns:
    Logits tensor of shape (1, num_classes)
  """
  encode_slide = getattr(model, "encode_slide", None)
  encode_slide_embedding = getattr(model, "encode_slide_embedding", None)

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

  return logits # type: ignore[no-any-return]


def _stream_slide_logits(
  model: nn.Module,
  tile_stream: Any,
  rr: RuntimeResolved,
  cfg: AppCfg,
  use_amp: bool,
  asynchrony: bool,
) -> tuple[torch.Tensor, int, bool]:
  """
  Stream tiles through the model to get slide-level logits.

  Caches TILE EMBEDDINGS (output of tile_encoder) so that:
  - tile_encoder runs only once per slide (expensive, frozen)
  - slide_encoder + classifier run every epoch (trainable)

  Returns:
    Tuple of (logits, tile_count, cache_hit)
  """
  feats_all, coords_all, tile_count, cache_hit = _get_tile_embeddings(
    model, tile_stream, rr, cfg, use_amp, asynchrony
  )
  logits = _embeddings_to_logits(model, feats_all, coords_all, use_amp)
  return logits, tile_count, cache_hit


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
  # Argmax predictions (kept for diagnostics).
  y_pred_argmax: list[int] = []
  pos_logits: list[float] = []
  margins: list[float] = []

  if skipped_slides is None:
    skipped_slides = set()

  for batch in tqdm(loader, desc="Eval", leave=False):
    for sample in _as_mil_samples(batch):
      x, y, _coords, _slide = _unpack_mil_batch(sample)
      y = y.to(rr.device, non_blocking=True)  # (1,)

      try:
        logits, _tile_count, _cache_hit = _stream_slide_logits(model, x, rr, cfg, use_amp, asynchrony=True)
      except RuntimeError as exc:
        if _is_skippable_tile_error(exc):
          slide_name = str(_slide)
          if slide_name not in skipped_slides:
            log.debug("Skipping slide %s during eval: %s", slide_name, exc)
            skipped_slides.add(slide_name)
          continue
        raise RuntimeError(f"Error during evaluation: {exc}")

      prob = torch.softmax(logits, dim=1)[:, 1]
      pred = torch.argmax(logits, dim=1)

      # Diagnostics: logits statistics (pos logit and pos-neg margin)
      with torch.no_grad():
        logits_1x2 = logits.detach().float().view(-1)
        if logits_1x2.numel() >= 2:
          neg_logit = float(logits_1x2[0].item())
          pos_logit = float(logits_1x2[1].item())
          pos_logits.append(pos_logit)
          margins.append(pos_logit - neg_logit)

      y_true.append(int(y.item()))
      y_prob.append(float(prob.item()))
      y_pred_argmax.append(int(pred.item()))

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
      "pr_auc": None,
      "prob_min": None,
      "prob_mean": None,
      "prob_max": None,
      "prob_lt_001_rate": None,
      "prob_gt_099_rate": None,
      "pos_logit_min": None,
      "pos_logit_mean": None,
      "pos_logit_max": None,
      "margin_min": None,
      "margin_mean": None,
      "margin_max": None,
      "best_thr": 0.5,
      "f1_at_best_thr": 0.0,
      "f1_at_0.5": 0.0,
    }

  y_true_arr = np.array(y_true)
  y_pred_argmax_arr = np.array(y_pred_argmax)
  y_prob_arr = np.array(y_prob)

  # AUC (only if both classes present)
  auc: Optional[float] = None
  if len(set(y_true)) > 1:
    auc = float(roc_auc_score(y_true, y_prob))

  # PR-AUC (average precision; only if both classes present)
  pr_auc: Optional[float] = None
  if len(set(y_true)) > 1:
    pr_auc = float(average_precision_score(y_true, y_prob))

  # Threshold tuning (optimize macro-F1 on val probabilities).
  best_thr, f1_at_best_thr = _find_best_threshold(y_true, y_prob)

  # Primary eval metrics: computed using tuned threshold (binary).
  y_pred_best_arr = (y_prob_arr >= best_thr).astype(int)
  acc = float(accuracy_score(y_true_arr, y_pred_best_arr))
  macro_f1 = float(f1_score(y_true_arr, y_pred_best_arr, average="macro", zero_division=0))
  cm = confusion_matrix(y_true_arr, y_pred_best_arr, labels=[0, 1])
  true_counts = {0: int(np.sum(y_true_arr == 0)), 1: int(np.sum(y_true_arr == 1))}
  pred_counts = {0: int(np.sum(y_pred_best_arr == 0)), 1: int(np.sum(y_pred_best_arr == 1))}
  pred_pos_rate = float(np.mean(y_pred_best_arr == 1))
  true_pos_rate = float(np.mean(y_true_arr == 1))
  precision = float(precision_score(y_true_arr, y_pred_best_arr, zero_division=0))
  recall = float(recall_score(y_true_arr, y_pred_best_arr, zero_division=0))

  # Argmax diagnostics (kept for debugging; not used for early stopping/checkpointing).
  acc_argmax = float(accuracy_score(y_true_arr, y_pred_argmax_arr))
  macro_f1_argmax = float(f1_score(y_true_arr, y_pred_argmax_arr, average="macro", zero_division=0))
  cm_argmax = confusion_matrix(y_true_arr, y_pred_argmax_arr, labels=[0, 1])

  # Score distribution diagnostics
  prob_min = float(np.min(y_prob_arr))
  prob_mean = float(np.mean(y_prob_arr))
  prob_max = float(np.max(y_prob_arr))
  prob_lt_001_rate = float(np.mean(y_prob_arr < 0.01))
  prob_gt_099_rate = float(np.mean(y_prob_arr > 0.99))

  pos_logits_arr = np.array(pos_logits, dtype=np.float32) if pos_logits else np.array([], dtype=np.float32)
  margins_arr = np.array(margins, dtype=np.float32) if margins else np.array([], dtype=np.float32)
  pos_logit_min = float(np.min(pos_logits_arr)) if pos_logits_arr.size else None
  pos_logit_mean = float(np.mean(pos_logits_arr)) if pos_logits_arr.size else None
  pos_logit_max = float(np.max(pos_logits_arr)) if pos_logits_arr.size else None
  margin_min = float(np.min(margins_arr)) if margins_arr.size else None
  margin_mean = float(np.mean(margins_arr)) if margins_arr.size else None
  margin_max = float(np.max(margins_arr)) if margins_arr.size else None

  # F1 at fixed threshold 0.5
  preds_at_05 = (y_prob_arr >= 0.5).astype(int)
  f1_at_05 = float(f1_score(y_true, preds_at_05, average="macro", zero_division=0))

  return {
    "acc": float(acc),
    "macro_f1": float(macro_f1),
    "acc_argmax": acc_argmax,
    "macro_f1_argmax": macro_f1_argmax,
    "auc": auc,
    "pr_auc": pr_auc,
    "cm": cm,
    "cm_argmax": cm_argmax,
    "true_counts": true_counts,
    "pred_counts": pred_counts,
    "pred_pos_rate": pred_pos_rate,
    "true_pos_rate": true_pos_rate,
    "precision": precision,
    "recall": recall,
    "prob_min": prob_min,
    "prob_mean": prob_mean,
    "prob_max": prob_max,
    "prob_lt_001_rate": prob_lt_001_rate,
    "prob_gt_099_rate": prob_gt_099_rate,
    "pos_logit_min": pos_logit_min,
    "pos_logit_mean": pos_logit_mean,
    "pos_logit_max": pos_logit_max,
    "margin_min": margin_min,
    "margin_mean": margin_mean,
    "margin_max": margin_max,
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
  # RankMix components (optional, for two-stage training)
  rankmix_scorer: Optional[TileScorer] = None,
  rankmix_sampler: Optional[RankMixSampler] = None,
  train_dataset: Optional[Any] = None,
) -> Path:
  """
  Train MIL model with early stopping and checkpointing.

  Supports optional RankMix data augmentation for class imbalance:
  - Stage 1 (epochs 1 to stage1_epochs): Standard MIL training
  - Stage 2 (epochs stage1_epochs+1 onwards): RankMix augmentation

  Args:
    cfg: Application configuration
    rr: Runtime resolution (device, etc.)
    model: MIL model to train
    train_loader: Training data loader
    val_loader: Validation data loader
    criterion: Loss function
    optimizer: Optimizer
    scheduler: Optional learning rate scheduler
    rankmix_scorer: Optional TileScorer for RankMix (required if rankmix enabled)
    rankmix_sampler: Optional RankMixSampler for partner slide selection
    train_dataset: Optional training dataset for RankMix partner access

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
  es_mode = "max" if es_metric in ("val_auc", "val_pr_auc", "val_macro_f1") else "min"
  stopper = EarlyStopping(
    patience=cfg.train.early_stopping.patience,
    min_delta=cfg.train.early_stopping.min_delta,
    mode=es_mode,
  )
  log.info("[EARLY STOPPING] metric=%s, mode=%s, patience=%d", es_metric, es_mode, cfg.train.early_stopping.patience)

  # Gradient clipping
  gradient_clip = cfg.train.gradient_clip

  # Track best checkpoint using the SAME metric as early stopping.
  # This avoids stopping on one objective but saving weights optimized for another.
  best_score: float = -math.inf if es_mode == "max" else math.inf
  best_score_str: str = "None"
  best_epoch: int = -1  # 1-based when reported
  best_val_threshold: Optional[float] = None

  asynchrony: bool = cfg.dataset.data.pin_memory
  skipped_train_slides: set[str] = set()
  skipped_val_slides: set[str] = set()
  skipped_eval_slides: set[str] = set()

  # For sanity check mode, limit epochs
  epochs = 1 if cfg.train.sanity_check else cfg.train.epochs

  # RankMix configuration
  rankmix_cfg = cfg.train.rankmix
  rankmix_enabled = rankmix_cfg.enabled and rankmix_scorer is not None and rankmix_sampler is not None

  if rankmix_enabled:
    log.info(
      "[RANKMIX] Stage 2 Training: alpha=%.2f, minority_ratio=%.2f",
      rankmix_cfg.alpha,
      rankmix_cfg.minority_sampling_ratio,
    )
  elif rankmix_cfg.enabled:
    log.warning("[RANKMIX] Config enabled but missing scorer/sampler - running standard training")

  for epoch in range(epochs):
    # -------------------------
    # Train
    # -------------------------
    model.train()
    if rankmix_scorer is not None:
      rankmix_scorer.train()

    train_losses: list[float] = []
    train_y_true: list[int] = []
    train_y_prob: list[float] = []
    train_y_pred: list[int] = []
    # Cache statistics for this epoch
    cache_hits = 0
    cache_misses = 0
    # RankMix statistics
    rankmix_count = 0
    rankmix_avg_lambda = 0.0
    pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)

    for batch in pbar:
      optimizer.zero_grad(set_to_none=True)
      loss_sum: torch.Tensor | None = None
      n_effective = 0
      tile_count_total = 0

      for sample in _as_mil_samples(batch):
        x, y, _coords, _slide = _unpack_mil_batch(sample)
        y = y.to(rr.device, non_blocking=asynchrony)  # (1,)
        y_val = int(y.item())

        try:
          # Get tile embeddings for current slide
          feats_a, coords_a, tile_count, cache_hit = _get_tile_embeddings(
            model, x, rr, cfg, use_amp, asynchrony
          )
        except RuntimeError as exc:
          if _is_skippable_tile_error(exc):
            slide_name = str(_slide)
            if slide_name not in skipped_train_slides:
              log.debug("Skipping slide %s during training: %s", slide_name, exc)
              skipped_train_slides.add(slide_name)
            continue
          raise RuntimeError(f"Error during training: {exc}")

        # Track cache statistics
        if cache_hit:
          cache_hits += 1
        else:
          cache_misses += 1

        # -------------------------
        # RankMix: Mix with partner slide (Stage 2 training)
        # -------------------------
        if rankmix_enabled and rankmix_sampler is not None and rankmix_scorer is not None and train_dataset is not None:
          try:
            # Sample a partner slide (biased toward minority class)
            partner_idx = rankmix_sampler.sample_partner_idx()
            partner_stream, partner_y, _partner_slide = train_dataset[partner_idx]
            partner_y_val = int(partner_y.item())

            # Get embeddings for partner slide
            feats_b, coords_b, _, partner_cache_hit = _get_tile_embeddings(
              model, partner_stream, rr, cfg, use_amp, asynchrony
            )
            if partner_cache_hit:
              cache_hits += 1
            else:
              cache_misses += 1

            # Score tiles for both slides
            with torch.no_grad():
              scores_a = rankmix_scorer(feats_a.detach())
              scores_b = rankmix_scorer(feats_b.detach())

            # Mix ranked embeddings
            feats_mixed, y_mixed, lam = rankmix(
              feats_a, feats_b, scores_a, scores_b,
              y_val, partner_y_val, alpha=rankmix_cfg.alpha
            )

            # Use mixed coordinates (average or from slide A)
            # For simplicity, we use coords from slide A since RankMix preserves order
            k = feats_mixed.shape[0]
            coords_mixed = coords_a[:k] if k <= coords_a.shape[0] else coords_a

            # Get logits from mixed embeddings
            logits = _embeddings_to_logits(model, feats_mixed, coords_mixed, use_amp)

            # Compute loss with soft label
            loss = compute_rankmix_loss(logits, y_mixed)

            # Track RankMix statistics
            rankmix_count += 1
            rankmix_avg_lambda = (rankmix_avg_lambda * (rankmix_count - 1) + lam) / rankmix_count

            # For metrics, use the mixed label rounded to nearest class
            y_for_metrics = int(round(y_mixed))

          except Exception as exc:
            # Fall back to standard training if RankMix fails for this sample
            log.debug("RankMix failed for slide %s, falling back to standard: %s", _slide, exc)
            logits = _embeddings_to_logits(model, feats_a, coords_a, use_amp)
            loss = criterion(logits.float(), y)
            y_for_metrics = y_val
        else:
          # -------------------------
          # Standard training (Stage 1 or RankMix disabled)
          # -------------------------
          logits = _embeddings_to_logits(model, feats_a, coords_a, use_amp)
          loss = criterion(logits.float(), y)
          y_for_metrics = y_val

        # Training metrics (slide-level) - detach so metrics don't backprop.
        with torch.no_grad():
          prob_pos = torch.softmax(logits.detach().float(), dim=1)[:, 1]
          pred = torch.argmax(logits.detach(), dim=1)
          train_y_true.append(y_for_metrics)
          train_y_prob.append(float(prob_pos.view(-1)[0].item()))
          train_y_pred.append(int(pred.view(-1)[0].item()))

        tile_count_total += int(tile_count)
        n_effective += 1
        loss_sum = loss if loss_sum is None else (loss_sum + loss)

      if n_effective == 0 or loss_sum is None:
        continue

      loss_mean = loss_sum / float(n_effective)

      if use_amp and scaler is not None:
        scaler.scale(loss_mean).backward()
        if gradient_clip > 0:
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
      else:
        loss_mean.backward()
        if gradient_clip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

      # Step scheduler per batch
      if scheduler is not None:
        scheduler.step()

      train_losses.append(float(loss_mean.item()))
      current_lr = optimizer.param_groups[0]["lr"]
      pbar.set_postfix(train_loss=f"{loss_mean.item():.4f}", n_patches=tile_count_total, lr=f"{current_lr:.2e}")

    train_loss = float(np.mean(train_losses)) if train_losses else 0.0

    # Train metrics (epoch-level)
    train_acc = accuracy_score(train_y_true, train_y_pred) if train_y_true else 0.0
    train_macro_f1 = f1_score(train_y_true, train_y_pred, average="macro", zero_division=0) if train_y_true else 0.0
    train_auc: Optional[float] = None
    train_pr_auc: Optional[float] = None
    if len(set(train_y_true)) > 1:
      train_auc = float(roc_auc_score(train_y_true, train_y_prob))
      train_pr_auc = float(average_precision_score(train_y_true, train_y_prob))
    train_cm = confusion_matrix(train_y_true, train_y_pred, labels=[0, 1]) if train_y_true else np.zeros((2, 2), dtype=int)

    # -------------------------
    # Val loss
    # -------------------------
    model.eval()
    val_losses: list[float] = []
    with torch.no_grad():
      for batch in tqdm(val_loader, desc="ValLoss", leave=False):
        for sample in _as_mil_samples(batch):
          x, y, _coords, _slide = _unpack_mil_batch(sample)
          y = y.to(rr.device, non_blocking=asynchrony)

          try:
            logits, _tile_count, _cache_hit = _stream_slide_logits(model, x, rr, cfg, use_amp, asynchrony)
          except RuntimeError as exc:
            if _is_skippable_tile_error(exc):
              slide_name = str(_slide)
              if slide_name not in skipped_val_slides:
                log.debug("Skipping slide %s during val loss: %s", slide_name, exc)
                skipped_val_slides.add(slide_name)
              continue
            raise RuntimeError(f"Error during validation: {exc}")
          loss = criterion(logits.float(), y)

          val_losses.append(float(loss.item()))

    val_loss = float(np.mean(val_losses)) if val_losses else 0.0

    # -------------------------
    # Val metrics
    # -------------------------
    metrics = evaluate_mil(cfg, rr, model, val_loader, skipped_slides=skipped_eval_slides)
    val_best_thr = float(metrics.get("best_thr", 0.5))
    auc = metrics.get("auc", None)
    val_auc = float(auc) if isinstance(auc, (float, int)) else -math.inf
    auc_str = f"{val_auc:.4f}" if math.isfinite(val_auc) else "None"

    # Train macro-F1 computed using the (val) best threshold for this epoch.
    if train_y_true:
      train_prob_arr = np.array(train_y_prob, dtype=np.float32)
      train_pred_at_val_best = (train_prob_arr >= val_best_thr).astype(int)
      train_macro_f1 = float(f1_score(train_y_true, train_pred_at_val_best, average="macro", zero_division=0))
    else:
      train_macro_f1 = 0.0

    # -------------------------
    # Checkpointing: follow early-stopping metric
    # -------------------------
    current_score: float
    current_score_str: str
    if es_metric == "val_auc":
      current_score = val_auc if math.isfinite(val_auc) else (-math.inf if es_mode == "max" else math.inf)
      current_score_str = auc_str
    elif es_metric == "val_pr_auc":
      pr_auc_val = metrics.get("pr_auc", None)
      pr_auc_num = float(pr_auc_val) if isinstance(pr_auc_val, (float, int)) else -math.inf
      current_score = pr_auc_num
      current_score_str = f"{pr_auc_num:.4f}" if math.isfinite(pr_auc_num) else "None"
    elif es_metric == "val_macro_f1":
      current_score = float(metrics["macro_f1"])
      current_score_str = f"{current_score:.4f}"
    else:  # val_loss
      current_score = val_loss
      current_score_str = f"{val_loss:.4f}"

    improved = (current_score > best_score) if es_mode == "max" else (current_score < best_score)
    if improved:
      best_score = current_score
      best_score_str = current_score_str
      best_epoch = epoch + 1  # 1-based
      best_val_threshold = val_best_thr
      if cfg.core.export.save_best_weights:
        torch.save(model.state_dict(), best_path)

    # -------------------------
    # Logging
    # -------------------------
    current_lr = optimizer.param_groups[0]["lr"]
    train_auc_str = f"{train_auc:.4f}" if isinstance(train_auc, (float, int)) else "None"
    train_pr_auc_str = f"{train_pr_auc:.4f}" if isinstance(train_pr_auc, (float, int)) else "None"
    val_pr_auc = metrics.get("pr_auc", None)
    val_pr_auc_str = f"{float(val_pr_auc):.4f}" if isinstance(val_pr_auc, (float, int)) else "None"

    log.info(
      f"Epoch {epoch+1}/{epochs} | "
      f"train_loss={train_loss:.4f} | "
      f"train_acc={float(train_acc):.4f} | train_macro_f1={float(train_macro_f1):.4f} | "
      f"train_auc={train_auc_str} | train_pr_auc={train_pr_auc_str} | "
      f"best_{es_metric}={best_score_str} | best_epoch={best_epoch} | "
      f"patience={stopper.counter}/{stopper.patience} | lr={current_lr:.2e}"
    )
    skipped_train_n = len(skipped_train_slides)
    skipped_val_n = len(skipped_val_slides)
    log.info(
      f"Epoch {epoch+1}/{epochs} | "
      f"val_loss={val_loss:.4f} | "
      f"val_acc={metrics['acc']:.4f} | val_macro_f1={metrics['macro_f1']:.4f} | "
      f"val_auc={auc_str} | val_pr_auc={val_pr_auc_str} | val_best_thr={val_best_thr:.3f}"
    )
    # Log cache statistics
    total_cache_ops = cache_hits + cache_misses
    cache_hit_rate = cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
    log.info(
      f"Epoch {epoch+1}/{epochs} | [CACHE STATS] hits={cache_hits} misses={cache_misses} "
      f"hit_rate={cache_hit_rate:.1%}"
    )
    # Log RankMix statistics if active
    if rankmix_enabled and rankmix_count > 0:
      log.info(
        f"Epoch {epoch+1}/{epochs} | [RANKMIX STATS] samples={rankmix_count} "
        f"avg_lambda={rankmix_avg_lambda:.3f}"
      )
    log.debug("Skipped slides this epoch: train=%d val=%d", skipped_train_n, skipped_val_n)
    log.info("Train confusion_matrix:\n%s", train_cm)
    log.info("Val confusion_matrix:\n%s", metrics["cm"])

    # -------------------------
    # Early stopping with configurable metric
    # -------------------------
    if es_metric == "val_auc":
      stopper.step(val_auc if math.isfinite(val_auc) else -math.inf)
    elif es_metric == "val_pr_auc":
      pr_auc_val = metrics.get("pr_auc", None)
      pr_auc_num = float(pr_auc_val) if isinstance(pr_auc_val, (float, int)) else -math.inf
      stopper.step(pr_auc_num)
    elif es_metric == "val_macro_f1":
      stopper.step(metrics["macro_f1"])
    else:  # val_loss
      stopper.step(val_loss)

    if stopper.early_stop:
      log.info("[Early Stop] Training stopped (%s criterion).", es_metric)
      break

  # Load best weights (by early-stopping metric)
  if cfg.core.export.save_best_weights and best_path.exists():
    log.info(f"[DONE] Best model saved to {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=rr.device))

  # Persist best summary
  summary = {
    "best_epoch": int(best_epoch),
    "best_metric": str(es_metric),
    "best_score": None if (best_epoch < 0 or not math.isfinite(best_score)) else float(best_score),
    "best_weights_path": str(best_path) if best_path.exists() else None,
    "best_val_threshold": best_val_threshold,
  }
  try:
    with open(run_dir / "best_summary.json", "w") as f:
      json.dump(summary, f, indent=2)
  except Exception as e:
    log.warning(f"[WARN] Failed to write best_summary.json: {e}")

  return best_path
