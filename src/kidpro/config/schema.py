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
    def _validate(self) -> "MILTaskCfg":
      if self.num_classes <= 1:
        raise ValueError("MIL (multi-instance learning) requires num_classes >= 2.")
      return self

TaskCfg = Annotated[
    Union[SegTaskCfg, ClassificationTaskCfg, MILTaskCfg],
    Field(discriminator="type"),
]

# -------------------------
# Model configs
# -------------------------

class WeightsCfg(BaseModel):
  # "local" => load from local_path
  # "hf_cache" => load from hf_cache_path (already downloaded / pre-populated)
  source: Literal["local", "hf_cache"] = "local"
  local_path: Optional[Path] = None
  hf_cache_path: Optional[Path] = Path("/home/khdp-user/workspace/huggingface_cache")

  @model_validator(mode="after")
  def _validate(self) -> "WeightsCfg":
    if self.source == "local" and self.local_path is None:
      raise ValueError("model.weights.local_path is required when weights.source='local'")
    if self.source == "hf_cache" and self.hf_cache_path is None:
      raise ValueError("model.weights.hf_cache_path is required when weights.source='hf_cache'")
    return self

class ModelCfg(BaseModel):
  name: Literal["unet", "timm", "prov_gigapath", "uni2_h", "virchow2"] = "unet"
  # Common
  in_channels: int = 3
  num_classes: Optional[int] = None

  # Optional weights (used by timm/prov/uni2_h as appropriate)
  weights: Optional[WeightsCfg] = None

  # Unet-specific
  encoder_name: str = "resnet50"
  encoder_weights: Optional[str] = "imagenet"
  activation: Optional[str] = None

  # timm-specific (also used as a generic "arch" string for other backbones)
  arch: Optional[str] = None
  pretrained: bool = False

  # --- foundation model backbones ---
  freeze_backbone: bool = True

  # Adapter projection: foundation_dim -> mil_dim
  foundation_dim: Optional[int] = None
  mil_dim: Optional[int] = None
  adapter_activation: Literal["relu", "gelu", "tanh"] = "gelu"
  adapter_dropout: float = 0.25

  # Attention head config
  num_heads: int = 4
  attn_hidden_dim: int = 128
  attn_dropout: float = 0.25
  input_size: Optional[int] = 256

  @model_validator(mode="after")
  def _validate(self) -> "ModelCfg":
    if self.in_channels <= 0:
      raise ValueError("model.in_channels must be > 0.")

    # For timm / prov_gigapath / uni2_h, arch is expected (even if it's just an identifier)
    if self.name in {"timm", "prov_gigapath", "uni2_h", "virchow2"}:
      if not self.arch:
        raise ValueError(f"model.arch is required when model.name='{self.name}'.")

    if self.adapter_dropout < 0 or self.adapter_dropout >= 1:
      raise ValueError("model.adapter_dropout must be in [0, 1).")

    if self.num_heads <= 0:
      raise ValueError("model.num_heads must be > 0.")

    if self.attn_hidden_dim <= 0:
      raise ValueError("model.attn_hidden_dim must be > 0.")

    if self.attn_dropout < 0 or self.attn_dropout >= 1:
      raise ValueError("model.attn_dropout must be in [0, 1).")

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
    data = self.dataset.data
    task = self.dataset.task
    paths = self.dataset.paths
    model = self.model

    #Classification requirements
    if isinstance(task, ClassificationTaskCfg):
      if paths.label_csv is None:
        raise ValueError("dataset.paths.label_csv is required for classification or MIL tasks.")

      if model.name != "timm":
        raise ValueError("For classification tasks, set model.name='timm' (current implementation).")

      # If model.num_classes provided, must match task.num_classes
      if model.num_classes is not None and model.num_classes != task.num_classes:
        raise ValueError(
          f"model.num_classes ({model.num_classes}) must equal dataset.task.num_classes ({task.num_classes})."
      )

    #MIL Task Requirements
    if isinstance(task, MILTaskCfg):
      if model.name not in ("timm", "prov_gigapath", "uni2_h", "virchow2"):
        raise ValueError("For MIL tasks, set model.name in {'timm','prov_gigapath','uni2_h','virchow2'}.")

      # If model.num_classes provided, must match task.num_classes
      if model.num_classes is not None and model.num_classes != task.num_classes:
        raise ValueError(
          f"model.num_classes ({model.num_classes}) must equal dataset.task.num_classes ({task.num_classes})."
        )

      # If model.input_size is provided, must match data.patch_size
      if model.input_size is not None and model.input_size != data.patch_size:
          raise ValueError(
            f"model.input_size ({model.input_size}) must equal data.patch_size ({data.patch_size})."
          )


    # Segmentation requirements
    if isinstance(task, SegTaskCfg):
      if model.name != "unet":
        raise ValueError("For segmentation tasks, set model.name='unet' (current implementation).")

    return self

# -------------------------
# Patch config (Hydra patch CLI)
# -------------------------
class PatchPathsCfg(BaseModel):
  svs_dir: Path
  xml_dir: Path
  out_home: Path


class PatchParamsCfg(BaseModel):
  target_mag: float = 10.0
  mask_mag: float = 2.5
  patch_size: int = 512
  stride: int = 512
  overlap_th: float = 0.05
  num_workers: int = 16
  allowed_mags: list[float] = Field(default_factory=lambda: [2.5, 10.0, 40.0])

  @model_validator(mode="after")
  def _validate(self) -> "PatchParamsCfg":
    if self.patch_size <= 0:
      raise ValueError("patch.params.patch_size must be > 0.")
    if self.stride <= 0:
      raise ValueError("patch.params.stride must be > 0.")
    if self.num_workers <= 0:
      raise ValueError("patch.params.num_workers must be > 0.")
    if not (0.0 <= self.overlap_th <= 1.0):
      raise ValueError("patch.params.overlap_th must be in [0, 1].")
    if not self.allowed_mags:
      raise ValueError("patch.params.allowed_mags must be non-empty.")
    if self.target_mag not in self.allowed_mags:
      raise ValueError("patch.params.target_mag must be in patch.params.allowed_mags.")
    if self.mask_mag not in self.allowed_mags:
      raise ValueError("patch.params.mask_mag must be in patch.params.allowed_mags.")
    return self


class PatchLoggingCfg(BaseModel):
  level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
  log_file: str = "process.log"


class PatchCfg(BaseModel):
  paths: PatchPathsCfg
  segmentation_type: str
  layer_ids: list[int] = Field(default_factory=list)
  output_map: dict[str, str] = Field(default_factory=dict)
  params: PatchParamsCfg = Field(default_factory=PatchParamsCfg)
  logging: PatchLoggingCfg = Field(default_factory=PatchLoggingCfg)

  @model_validator(mode="after")
  def _validate(self) -> "PatchCfg":
    if not self.layer_ids:
      raise ValueError("patch.layer_ids must include at least one layer id.")
    if self.segmentation_type not in self.output_map:
      raise ValueError("patch.segmentation_type must exist in patch.output_map.")
    return self
