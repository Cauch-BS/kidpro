from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# -------------------------
# Core enums / literals
# -------------------------
DeviceChoice = Literal["auto", "cpu", "cuda"]


# -------------------------
# Paths
# -------------------------
class PathsCfg(BaseModel):
  root_dir: Path
  runs_root: Path
  csv_name: str = "dataset.csv"

  # Classification-only
  label_csv: Optional[Path] = None


# -------------------------
# Task configs (discriminated union)
# -------------------------
class SegTaskCfg(BaseModel):
  # "binary" => one foreground class, single layer_id required
  # "categorical" => multi-layer segmentation, layer_ids >= 1 required
  type: Literal["binary", "categorical"]
  layer_ids: list[int] = Field(default_factory=list)

  @model_validator(mode="after")
  def _validate(self) -> "SegTaskCfg":
    if len(self.layer_ids) < 1:
      raise ValueError("Segmentation task requires at least one layer_id.")
    if self.type == "binary" and len(self.layer_ids) != 1:
      raise ValueError("Binary segmentation requires exactly one layer_id.")
    # categorical can be 1+; background is handled downstream
    return self


class ClassificationTaskCfg(BaseModel):
  type: Literal["classification"]
  num_classes: int = 2

  # Optional; useful if your labeling logic wants a canonical positive label name.
  # Your current split_cls.py hard-codes: target == "m1" -> 1 else 0
  # Keeping this field makes that rule configurable later.
  positive_label: Optional[str] = "m1"

  @model_validator(mode="after")
  def _validate(self) -> "ClassificationTaskCfg":
    if self.num_classes <= 1:
      raise ValueError("Classification requires num_classes >= 2.")
    return self

class MILTaskCfg(BaseModel):
    type: Literal["mil"]
    num_classes: int = 2
    top_k: int = 10
    max_patches: int = 300
    sample_mode: Literal["random", "first"] = "random"

    @model_validator(mode="after")
    def _validate(self) -> "ClassificationTaskCfg":
      if self.num_classes <= 1:
        raise ValueError("Classification requires num_classes >= 2.")
      return self

TaskCfg = Annotated[
    Union[SegTaskCfg, ClassificationTaskCfg, MILTaskCfg],
    Field(discriminator="type"),
]

# -------------------------
# Model configs
# -------------------------
class ModelCfg(BaseModel):
  # Two supported model families in your repo today:
  # - segmentation_models_pytorch Unet ("unet")
  # - timm classifiers ("timm")
  name: Literal["unet", "timm"] = "unet"
  init_ckpt: Optional[Path]

  # Unet-specific
  encoder_name: str = "resnet50"
  encoder_weights: Optional[str] = "imagenet"
  activation: Optional[str] = None

  # timm-specific
  arch: Optional[str] = None          # e.g. "resnet50"
  pretrained: bool = False
  num_classes: Optional[int] = None   # head dimension (should match task for classification)

  # shared
  in_channels: int = 3

  @model_validator(mode="after")
  def _validate(self) -> "ModelCfg":
    if self.in_channels <= 0:
      raise ValueError("model.in_channels must be > 0.")

    if self.name == "timm":
      if not self.arch:
        raise ValueError("model.arch is required when model.name='timm'.")
      # num_classes can be set either here or derived from task; we validate later in AppCfg.
    return self


# -------------------------
# Data / Train / Runtime / Export
# -------------------------
class DataCfg(BaseModel):
  patch_size: int = 512
  train_ratio: float = 0.6
  test_ratio: float = 0.2
  val_ratio: float = 0.2
  num_workers: int = 4
  pin_memory: bool = True

  @model_validator(mode="after")
  def _validate(self) -> "DataCfg":
    if self.patch_size <= 0:
      raise ValueError("data.patch_size must be > 0.")
    if not (0.0 < self.train_ratio < 1.0):
      raise ValueError("data.train_ratio must be in (0,1).")
    if not (0.0 < self.test_ratio < 1.0):
      raise ValueError("data.test_ratio must be in (0, 1).")
    if not (0.0 <= self.val_ratio < 1.0):
      raise ValueError("data.val_ratio must be in [0, 1).")
    if abs(self.train_ratio + self.test_ratio + self.val_ratio - 1) >= 1e-6:
      raise ValueError("train, test and validation ratios must sum to 1")
    return self


class EarlyStoppingCfg(BaseModel):
  patience: int = 5
  min_delta: float = 1e-5
  mode: Literal["min", "max"] = "min"

  @model_validator(mode="after")
  def _validate(self) -> "EarlyStoppingCfg":
    if self.patience <= 0:
      raise ValueError("train.early_stopping.patience must be > 0.")
    if self.min_delta < 0:
      raise ValueError("train.early_stopping.min_delta must be >= 0.")
    return self


class TrainCfg(BaseModel):
  batch_size: int = 16
  epochs: int = 50
  lr: float = 1e-4
  seed: int = 0
  early_stopping: EarlyStoppingCfg = Field(default_factory=EarlyStoppingCfg)

  @model_validator(mode="after")
  def _validate(self) -> "TrainCfg":
    if self.batch_size <= 0:
      raise ValueError("train.batch_size must be > 0.")
    if self.epochs <= 0:
      raise ValueError("train.epochs must be > 0.")
    if self.lr <= 0:
      raise ValueError("train.lr must be > 0.")
    return self


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

# -------------------------
# App config (cross-field validation)
# -------------------------
class AppCfg(BaseModel):
  model: ModelCfg
  dataset: DatasetCfg
  train: TrainCfg
  runtime: RuntimeCfg
  export: ExportCfg

  # Hydra run dir is injected by CONFIG()
  run_dir: Optional[Path] = None

  @model_validator(mode="after")
  def _validate_cross_fields(self) -> "AppCfg":
    task = self.dataset.task
    paths = self.dataset.paths
    model = self.model

    # Classification requirements
    if isinstance(task, ClassificationTaskCfg) or isinstance(task, MILTaskCfg):
      if paths.label_csv is None:
        raise ValueError("dataset.paths.label_csv is required for classification or MIL tasks.")

      # For classification, ensure we use timm (for now) or at least align head dims
      if model.name != "timm":
        raise ValueError("For classification or MIL tasks, set model.name='timm' (current implementation).")

      # If model.num_classes provided, must match task.num_classes
      if model.num_classes is not None and model.num_classes != task.num_classes:
        raise ValueError(
          f"model.num_classes ({model.num_classes}) must equal dataset.task.num_classes ({task.num_classes})."
        )

    # Segmentation requirements
    if isinstance(task, SegTaskCfg):
      if model.name != "unet":
        raise ValueError("For segmentation tasks, set model.name='unet' (current implementation).")

    return self
