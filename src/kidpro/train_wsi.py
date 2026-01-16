from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.load import CONFIG, CONFIG_EXPORT
from .data.dataset_mil import MILDataset
from .data.split_mil import build_mil_split_csv
from .data.transform import get_transforms
from .modeling.factory_wsi import build_model_mil
from .training.loop_mil import fit_mil

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config_wsi")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)

  if not cfg.model.freeze_backbone:
    raise ValueError("train_wsi requires model.freeze_backbone=true.")

  CONFIG_EXPORT(cfg, rr)

  # Build slide-level CSV (SlideName / GT / split)
  df = build_mil_split_csv(cfg)

  df_tr = df[df["split"] == "train"].reset_index(drop=True)
  df_va = df[df["split"] == "val"].reset_index(drop=True)

  train_tf, val_tf = get_transforms(cfg)

  ds_tr = MILDataset(cfg, df_tr, transform=train_tf)
  ds_va = MILDataset(cfg, df_va, transform=val_tf)

  dl_tr = torch.utils.data.DataLoader(
    ds_tr,
    batch_size=1,
    shuffle=True,
    num_workers=cfg.dataset.data.num_workers,
    pin_memory=cfg.dataset.data.pin_memory,
    persistent_workers=(cfg.dataset.data.num_workers > 0),
    prefetch_factor=4 if cfg.dataset.data.num_workers > 0 else None,
  )
  dl_va = torch.utils.data.DataLoader(
    ds_va,
    batch_size=1,
    shuffle=False,
    num_workers=cfg.dataset.data.num_workers,
    pin_memory=cfg.dataset.data.pin_memory,
    persistent_workers=(cfg.dataset.data.num_workers > 0),
    prefetch_factor=4 if cfg.dataset.data.num_workers > 0 else None,
  )

  model = build_model_mil(cfg).to(rr.device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

  best_path = fit_mil(cfg, rr, model, dl_tr, dl_va, criterion, optimizer)
  log.info(f"[RUN COMPLETE] run_dir={run_dir} best={best_path}")


if __name__ == "__main__":
  main()
