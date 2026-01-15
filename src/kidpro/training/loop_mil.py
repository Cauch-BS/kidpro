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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
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


def _unpack_mil_batch(
  batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
  if len(batch) == 3:
    x, y, slide = batch
    return x, y, None, str(slide)
  if len(batch) == 4:
    x, y, coords, slide = batch
    return x, y, coords, str(slide)
  raise ValueError(f"Unexpected MIL batch format with {len(batch)} items.")


@torch.no_grad()
def evaluate_mil(rr: RuntimeResolved, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
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

  for batch in tqdm(loader, desc="Eval", leave=False):
    x, y, coords, _slide = _unpack_mil_batch(batch)
    x = x.squeeze(0).to(rr.device, non_blocking=True)  # (N,C,H,W)
    y = y.to(rr.device, non_blocking=True)             # (1,)
    coords_t = coords.squeeze(0).to(rr.device, non_blocking=True) if coords is not None else None

    # FIXED: Proper amp context handling
    if use_amp and autocast is not None:
      with autocast(device_type="cuda"):
        logits = model(x, coords_t)  # (1,2)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)
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
    return {"acc": 0.0, "macro_f1": 0.0, "auc": None, "cm": np.zeros((2, 2), dtype=int)}

  acc = accuracy_score(y_true, y_pred)
  macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

  auc: Optional[float] = None
  if len(set(y_true)) > 1:
    auc = float(roc_auc_score(y_true, y_prob))

  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
  return {"acc": float(acc), "macro_f1": float(macro_f1), "auc": auc, "cm": cm}


def fit_mil(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  criterion: nn.Module,
  optimizer: optim.Optimizer,
) -> Path:
  """
  Train MIL model with early stopping on val_loss and checkpointing on val_auc.

  Args:
    cfg: Application configuration
    rr: Runtime resolution (device, etc.)
    model: MIL model to train
    train_loader: Training data loader
    val_loader: Validation data loader
    criterion: Loss function
    optimizer: Optimizer

  Returns:
    Path to best model checkpoint
  """
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()
  best_path = run_dir / cfg.export.best_weights_name

  # FIXED: Proper amp handling
  use_amp = (rr.device == "cuda" and HAS_AMP)
  scaler = GradScaler(device="cuda", enabled=use_amp) if use_amp and GradScaler is not None else None

  # Early stopping on loss (minimize)
  stopper_loss = EarlyStopping(
    patience=cfg.train.early_stopping.patience,
    min_delta=cfg.train.early_stopping.min_delta,
    mode="min",
  )

  best_val_auc: float = -math.inf
  best_epoch: int = -1  # 1-based when reported

  asynchrony: bool = cfg.dataset.data.pin_memory

  for epoch in range(cfg.train.epochs):
    # -------------------------
    # Train
    # -------------------------
    model.train()
    train_losses: list[float] = []
    pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.train.epochs}", leave=False)

    for batch in pbar:
      x, y, coords, _slide = _unpack_mil_batch(batch)
      x = x.squeeze(0).to(rr.device, non_blocking=asynchrony)  # (N,C,H,W)
      y = y.to(rr.device, non_blocking=asynchrony)             # (1,)
      coords_t = coords.squeeze(0).to(rr.device, non_blocking=asynchrony) if coords is not None else None

      optimizer.zero_grad(set_to_none=True)

      # FIXED: Proper amp handling
      if use_amp and autocast is not None:
        with autocast(device_type="cuda"):
          logits = model(x, coords_t)  # (1,2)
          loss = criterion(logits, y)
      else:
        logits = model(x, coords_t)
        loss = criterion(logits, y)

      if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()

      train_losses.append(float(loss.item()))
      pbar.set_postfix(train_loss=f"{loss.item():.4f}", n_patches=int(x.size(0)))

    train_loss = float(np.mean(train_losses)) if train_losses else 0.0

    # -------------------------
    # Val loss
    # -------------------------
    model.eval()
    val_losses: list[float] = []
    with torch.no_grad():
      for batch in tqdm(val_loader, desc="ValLoss", leave=False):
        x, y, coords, _slide = _unpack_mil_batch(batch)
        x = x.squeeze(0).to(rr.device, non_blocking=asynchrony)
        y = y.to(rr.device, non_blocking=asynchrony)
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
    metrics = evaluate_mil(rr, model, val_loader)
    auc = metrics.get("auc", None)
    val_auc = float(auc) if isinstance(auc, (float, int)) else -math.inf
    auc_str = f"{val_auc:.4f}" if math.isfinite(val_auc) else "None"

    # -------------------------
    # Checkpointing: maximize val_auc
    # -------------------------
    if val_auc > best_val_auc:
      best_val_auc = val_auc
      best_epoch = epoch + 1  # 1-based
      if cfg.export.save_best_weights:
        torch.save(model.state_dict(), best_path)

    # -------------------------
    # Logging
    # -------------------------
    best_auc_str = f"{best_val_auc:.4f}" if math.isfinite(best_val_auc) else "None"
    log.info(
      f"Epoch {epoch+1}/{cfg.train.epochs} | "
      f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
      f"val_acc={metrics['acc']:.4f} | val_macro_f1={metrics['macro_f1']:.4f} | val_auc={auc_str} | "
      f"best_val_auc={best_auc_str} | best_epoch={best_epoch} | "
      f"patience={stopper_loss.counter}/{stopper_loss.patience}"
    )

    # -------------------------
    # Early stopping: minimize val_loss
    # -------------------------
    stopper_loss.step(val_loss)
    if stopper_loss.early_stop:
      log.info("[Early Stop] Training stopped (val_loss criterion).")
      break

  # Load best weights (by val_auc)
  if cfg.export.save_best_weights and best_path.exists():
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
