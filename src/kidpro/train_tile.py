from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.load import CONFIG, CONFIG_EXPORT
from .data.dataset import SegDataset
from .data.split import build_dataset_csv
from .data.transform import get_transforms
from .modeling.factory_tile import build_loss, build_model, build_optimizer
from .training.loop import fit

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(hcfg: DictConfig) -> None:
  run_dir = Path.cwd()  # Hydra job.chdir=true => cwd is the unique run directory
  cfg, rr = CONFIG(hcfg, run_dir=run_dir)

  # Export config + env capture into the run directory
  CONFIG_EXPORT(cfg, rr)

  # Build CSV in run dir (so the run is self-contained)
  df = build_dataset_csv(cfg)

  # Split dataframes
  df_tr = df[df["split"] == "train"]
  df_va = df[df["split"] == "val"]

  # Transforms
  train_tf, val_tf = get_transforms(cfg)

  # Datasets & loaders
  ds_tr = SegDataset(cfg, df_tr, train_tf)
  ds_va = SegDataset(cfg, df_va, val_tf)

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

  # Model / loss / optim
  model = build_model(cfg).to(rr.device)
  criterion = build_loss(cfg)
  optimizer = build_optimizer(cfg, model)

  # Train
  if cfg.mlflow.enabled:
    try:
      import mlflow
      import mlflow.pytorch
    except Exception as e:
      raise RuntimeError("MLflow is enabled but could not be imported.") from e

    if cfg.mlflow.tracking_uri:
      mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    with mlflow.start_run(run_name=run_dir.name):
      mlflow.log_params(
        {
          "model_name": cfg.model.name,
          "dataset_root_dir": str(cfg.dataset.paths.root_dir),
          "dataset_csv_name": cfg.dataset.paths.csv_name,
          "patch_size": cfg.dataset.data.patch_size,
          "batch_size": cfg.train.batch_size,
          "epochs": cfg.train.epochs,
          "lr": cfg.train.lr,
        }
      )
      best_path = fit(cfg, rr, model, dl_tr, dl_va, criterion, optimizer)
      if cfg.export.save_best_weights and best_path.exists():
        mlflow.log_artifact(str(best_path), artifact_path="weights")
      mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name=cfg.mlflow.registry_model_name,
      )
  else:
    best_path = fit(cfg, rr, model, dl_tr, dl_va, criterion, optimizer)
  log.info(f"[RUN COMPLETE] run_dir={run_dir} best={best_path}")


if __name__ == "__main__":
  main()
