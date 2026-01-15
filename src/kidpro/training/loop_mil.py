from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.load import RuntimeResolved
from ..config.schema import AppCfg
from .early_stop import EarlyStopping

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_mil(rr: RuntimeResolved, model: nn.Module, loader: DataLoader) -> Dict[str, object]:
  model.eval()
  use_amp = (rr.device == "cuda")

  y_true: list[int] = []
  y_prob: list[float] = []
  y_pred: list[int] = []

  for x, y, _slide in tqdm(loader, desc="Eval", leave=False):
    x = x.squeeze(0).to(rr.device)  # (N,C,H,W)
    y = y.to(rr.device)             # (1,)

    # Modern AMP autocast
    with autocast(device_type="cuda", enabled=use_amp):
      logits = model(x)  # (1,2)
      prob = torch.softmax(logits, dim=1)[:, 1]
      pred = torch.argmax(logits, dim=1)

    y_true.append(int(y.item()))
    y_prob.append(float(prob.item()))
    y_pred.append(int(pred.item()))

  acc = accuracy_score(y_true, y_pred) if y_true else 0.0
  macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0

  auc: Optional[float] = None
  if len(set(y_true)) > 1:
    auc = float(roc_auc_score(y_true, y_prob))

  cm = confusion_matrix(y_true, y_pred) if y_true else np.zeros((2, 2), dtype=int)
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
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()
  best_path = run_dir / cfg.export.best_weights_name

  use_amp = (rr.device == "cuda")
  scaler = GradScaler(device="cuda", enabled=use_amp)

  stopper = EarlyStopping(
    patience=cfg.train.early_stopping.patience,
    min_delta=cfg.train.early_stopping.min_delta,
    mode=cfg.train.early_stopping.mode,
  )

  for epoch in range(cfg.train.epochs):
    # -------------------------
    # Train
    # -------------------------
    model.train()
    train_losses: list[float] = []
    pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.train.epochs}", leave=False)

    for x, y, _slide in pbar:
      x = x.squeeze(0).to(rr.device)  # (N,C,H,W)
      y = y.to(rr.device)             # (1,)

      optimizer.zero_grad(set_to_none=True)

      with autocast(device_type="cuda", enabled=use_amp):
        logits = model(x)             # (1,2)
        loss = criterion(logits, y)

      if use_amp:
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
      for x, y, _slide in tqdm(val_loader, desc="ValLoss", leave=False):
        x = x.squeeze(0).to(rr.device)
        y = y.to(rr.device)

        with autocast(device_type="cuda", enabled=use_amp):
          logits = model(x)
          loss = criterion(logits, y)

        val_losses.append(float(loss.item()))

    val_loss = float(np.mean(val_losses)) if val_losses else 0.0

    # Metrics (also autocast inside evaluate_mil)
    metrics = evaluate_mil(rr, model, val_loader)
    auc_str = "None" if metrics["auc"] is None else f"{metrics['auc']:.4f}"

    is_best = stopper.step(val_loss)
    if is_best and cfg.export.save_best_weights:
      torch.save(model.state_dict(), best_path)

    log.info(
      f"Epoch {epoch+1}/{cfg.train.epochs} | "
      f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
      f"val_acc={metrics['acc']:.4f} | val_macro_f1={metrics['macro_f1']:.4f} | val_auc={auc_str} | "
      f"best={stopper.best_score:.4f} | patience={stopper.counter}/{stopper.patience}"
    )

    if stopper.early_stop:
      log.info("[Early Stop] Training stopped.")
      break

  if cfg.export.save_best_weights and best_path.exists():
    log.info(f"[DONE] Best model saved to {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=rr.device))

  return best_path
