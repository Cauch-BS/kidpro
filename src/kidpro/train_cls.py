from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.load import CONFIG, CONFIG_EXPORT
from .data.dataset_cls import PatchClsDataset
from .data.split_cls import build_cls_dataset_csv
from .data.transform_cls import get_transforms_cls
from .modeling.factory_cls import build_loss_cls, build_model_cls, build_optimizer_cls
from .training.loop_cls import fit_cls

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config_cls")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)
  CONFIG_EXPORT(cfg, rr)

  df = build_cls_dataset_csv(cfg)
  df_tr = df[df["split"] == "train"]
  df_va = df[df["split"] == "val"]

  train_tf, val_tf = get_transforms_cls(cfg)

  ds_tr = PatchClsDataset(df_tr, train_tf)
  ds_va = PatchClsDataset(df_va, val_tf)

  dl_tr = torch.utils.data.DataLoader(
    ds_tr,
    batch_size=cfg.train.batch_size,
    shuffle=True,
    num_workers=cfg.dataset.data.num_workers,
    pin_memory=cfg.dataset.data.pin_memory,
  )
  dl_va = torch.utils.data.DataLoader(
    ds_va,
    batch_size=cfg.train.batch_size,
    shuffle=False,
    num_workers=cfg.dataset.data.num_workers,
    pin_memory=cfg.dataset.data.pin_memory,
  )

  model = build_model_cls(cfg).to(rr.device)
  criterion = build_loss_cls(cfg)
  optimizer = build_optimizer_cls(cfg, model)

  best_path = fit_cls(cfg, rr, model, dl_tr, dl_va, criterion, optimizer)
  log.info(f"[RUN COMPLETE] run_dir={run_dir} best={best_path}")


if __name__ == "__main__":
  main()
