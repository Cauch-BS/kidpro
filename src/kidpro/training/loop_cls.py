from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.load import RuntimeResolved
from ..config.schema import AppCfg
from .early_stop import EarlyStopping

log = logging.getLogger(__name__)


def train_one_epoch_cls(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: optim.Optimizer,
) -> float:
  model.train()
  losses: list[float] = []

  pbar = tqdm(loader, desc="Train", leave=False)
  for x, y in pbar:
    x, y = x.to(rr.device), y.to(rr.device)

    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(float(loss.item()))
    pbar.set_postfix(train_loss=f"{loss.item():.4f}")

  return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_cls(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
) -> Tuple[float, float]:
  model.eval()

  total_loss = 0.0
  correct = 0
  n = 0

  for x, y in tqdm(loader, desc="Val", leave=False):
    x, y = x.to(rr.device), y.to(rr.device)

    logits = model(x)
    loss = criterion(logits, y)

    bs = x.size(0)
    total_loss += float(loss.item()) * bs
    n += bs

    pred = torch.argmax(logits, dim=1)
    correct += int((pred == y).sum().item())

  val_loss = total_loss / max(n, 1)
  val_acc = correct / max(n, 1)
  return val_loss, val_acc


def fit_cls(
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

  stopper = EarlyStopping(
    patience=cfg.train.early_stopping.patience,
    min_delta=cfg.train.early_stopping.min_delta,
    mode=cfg.train.early_stopping.mode,
  )

  for epoch in range(cfg.train.epochs):
    train_loss = train_one_epoch_cls(cfg, rr, model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_cls(cfg, rr, model, val_loader, criterion)

    is_best = stopper.step(val_loss)
    if is_best and cfg.export.save_best_weights:
      torch.save(model.state_dict(), best_path)

    log.info(
      f"Epoch {epoch + 1}/{cfg.train.epochs} | "
      f"train_loss={train_loss:.4f} | "
      f"val_loss={val_loss:.4f} | "
      f"val_acc={val_acc:.4f} | "
      f"best={stopper.best_score:.4f} | "
      f"patience={stopper.counter}/{stopper.patience}"
    )

    if stopper.early_stop:
      log.info("[Early Stop] Training stopped.")
      break

  if cfg.export.save_best_weights and best_path.exists():
    log.info(f"[DONE] Best model saved to {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=rr.device))

  return best_path
