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


def train_one_epoch(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,  # Annotated
  loader: DataLoader,  # Annotated
  criterion: nn.Module,  # Annotated
  optimizer: optim.Optimizer,  # Annotated
) -> float:
  model.train()
  losses = []

  pbar = tqdm(loader, desc="Train", leave=False)
  for img, mask in pbar:
    img, mask = img.to(rr.device), mask.to(rr.device)
    pred = model(img)

    if cfg.dataset.task.type == "binary":
      loss = criterion(pred.squeeze(1), mask)
    else:
      loss = criterion(pred, mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix(train_loss=f"{loss.item():.4f}")

  return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,  # Annotated
  loader: DataLoader,  # Annotated
  criterion: nn.Module,  # Annotated
) -> Tuple[float, float, float]:
  model.eval()
  total_loss, n = 0.0, 0

  dice_list = []
  iou_list = []

  for img, mask in tqdm(loader, desc="Val", leave=False):
    img, mask = img.to(rr.device), mask.to(rr.device)
    pred = model(img)

    if cfg.dataset.task.type == "binary":
      loss = criterion(pred.squeeze(1), mask)
      prob = torch.sigmoid(pred)
      pred_bin = (prob > 0.5).float()
      gt = mask.unsqueeze(1)
    else:
      loss = criterion(pred, mask)
      prob = torch.softmax(pred, dim=1)
      pred_bin = torch.argmax(prob, dim=1)
      gt = mask

    total_loss += loss.item() * img.size(0)
    n += img.size(0)

    if cfg.dataset.task.type == "binary":
      p = pred_bin.view(pred_bin.size(0), -1)
      g = gt.view(gt.size(0), -1)

      inter = (p * g).sum(dim=1)
      union = p.sum(dim=1) + g.sum(dim=1)

      dice = (2 * inter + 1e-6) / (union + 1e-6)
      iou = (inter + 1e-6) / (p.sum(dim=1) + g.sum(dim=1) - inter + 1e-6)

      dice_list.append(dice.mean().item())
      iou_list.append(iou.mean().item())
    else:
      num_classes = pred.shape[1]
      dice_per_class = []
      iou_per_class = []
      for c in range(1, num_classes):  # exclude background
        p = (pred_bin == c).float().view(pred_bin.size(0), -1)
        g = (gt == c).float().view(gt.size(0), -1)
        inter = (p * g).sum(dim=1)
        union = p.sum(dim=1) + g.sum(dim=1)
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        iou = (inter + 1e-6) / (p.sum(dim=1) + g.sum(dim=1) - inter + 1e-6)
        dice_per_class.append(dice.mean())
        iou_per_class.append(iou.mean())
      dice_list.append(torch.stack(dice_per_class).mean().item())
      iou_list.append(torch.stack(iou_per_class).mean().item())

  val_loss = total_loss / max(n, 1)
  val_dice = float(np.mean(dice_list)) if dice_list else 0.0
  val_iou = float(np.mean(iou_list)) if iou_list else 0.0
  return val_loss, val_dice, val_iou


def fit(
  cfg: AppCfg,
  rr: RuntimeResolved,
  model: nn.Module,  # Annotated
  train_loader: DataLoader,  # Annotated
  val_loader: DataLoader,  # Annotated
  criterion: nn.Module,  # Annotated
  optimizer: optim.Optimizer,  # Annotated
) -> Path:
  # FIX: Handle Optional[Path]
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()

  best_path = run_dir / cfg.export.best_weights_name

  stopper = EarlyStopping(
    patience=cfg.train.early_stopping.patience,
    min_delta=cfg.train.early_stopping.min_delta,
    mode=cfg.train.early_stopping.mode,
  )

  for epoch in range(cfg.train.epochs):
    train_loss = train_one_epoch(cfg, rr, model, train_loader, criterion, optimizer)
    val_loss, val_dice, val_iou = validate(cfg, rr, model, val_loader, criterion)

    if cfg.mlflow.enabled:
      try:
        import mlflow
      except Exception:
        mlflow = None
      if mlflow is not None and mlflow.active_run():
        mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
        mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
        mlflow.log_metric("val_dice", val_dice, step=epoch + 1)
        mlflow.log_metric("val_iou", val_iou, step=epoch + 1)

    is_best = stopper.step(val_loss)
    if is_best and cfg.export.save_best_weights:
      torch.save(model.state_dict(), best_path)

    log.info(
      f"Epoch {epoch + 1}/{cfg.train.epochs} | "
      f"train_loss={train_loss:.4f} | "
      f"val_loss={val_loss:.4f} | "
      f"val_dice={val_dice:.4f} | "
      f"val_iou={val_iou:.4f} | "
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
