from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

TaskType = Literal["binary", "categorical"]
DeviceChoice = Literal["auto", "cpu", "cuda"]


class PathsCfg(BaseModel):
  root_dir: Path
  runs_root: Path
  csv_name: str = "dataset.csv"


class TaskCfg(BaseModel):
  type: TaskType
  layer_ids: List[int] = Field(default_factory=list)


class ModelCfg(BaseModel):
  name: Literal["unet"] = "unet"
  encoder_name: str = "resnet50"
  encoder_weights: Optional[str] = "imagenet"
  in_channels: int = 3
  activation: Optional[str] = None


class DataCfg(BaseModel):
  patch_size: int = 512
  test_ratio: float = 0.2
  val_ratio: float = 0.2
  num_workers: int = 4
  pin_memory: bool = True


class EarlyStoppingCfg(BaseModel):
  patience: int = 5
  min_delta: float = 1e-5
  mode: Literal["min", "max"] = "min"


class TrainCfg(BaseModel):
  batch_size: int = 16
  epochs: int = 50
  lr: float = 1e-4
  seed: int = 0
  early_stopping: EarlyStoppingCfg = EarlyStoppingCfg()


class RuntimeCfg(BaseModel):
  device: DeviceChoice = "auto"
  cudnn_benchmark: bool = True
  cudnn_deterministic: bool = False


class ExportCfg(BaseModel):
  save_resolved_config: bool = True
  resolved_config_name: str = "config_resolved.yaml"

  save_env_json: bool = True
  env_json_name: str = "training_env.json"

  save_best_weights: bool = True
  best_weights_name: str = "best_model.pt"


class DatasetCfg(BaseModel):
  paths: PathsCfg
  task: TaskCfg
  data: DataCfg


class AppCfg(BaseModel):
  model: ModelCfg
  dataset: DatasetCfg
  train: TrainCfg
  runtime: RuntimeCfg
  export: ExportCfg

  # Hydra will create a unique working directory; we treat it as the run dir.
  run_dir: Optional[Path] = None

  @model_validator(mode="after")
  def _validate_invariants(self) -> "AppCfg":
    task = self.dataset.task
    data = self.dataset.data

    if task.type == "binary" and len(task.layer_ids) != 1:
      raise ValueError("task.type='binary' requires task.layer_ids to have length 1.")
    if data.patch_size <= 0:
      raise ValueError("data.patch_size must be > 0.")
    if not (0.0 < data.test_ratio < 1.0):
      raise ValueError("data.test_ratio must be in (0, 1).")
    if not (0.0 <= data.val_ratio < 1.0):
      raise ValueError("data.val_ratio must be in [0, 1).")
    if self.train.lr <= 0:
      raise ValueError("train.lr must be > 0.")
    if self.train.epochs <= 0:
      raise ValueError("train.epochs must be > 0.")
    if self.train.batch_size <= 0:
      raise ValueError("train.batch_size must be > 0.")
    return self
