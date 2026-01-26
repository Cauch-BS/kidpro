from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union, cast

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

  # MIL-only
  label_csv: Optional[Path] = None
  cache_dir: Optional[Path] = None
  wsi_cache_dir: Optional[Path] = None
  tiles_key: str = "tiles"
  wsi_dir: Optional[Path] = None
  wsi_ext: str = ".svs"


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


class MILTaskCfg(BaseModel):
    type: Literal["mil"]
    num_classes: int = 2
    top_k: int = 10

    @model_validator(mode="after")
    def _validate(self) -> "MILTaskCfg":
      if self.num_classes <= 1:
        raise ValueError("MIL (multi-instance learning) requires num_classes >= 2.")
      return self

TaskCfg = Annotated[
    Union[SegTaskCfg, MILTaskCfg],
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


class LoraCfg(BaseModel):
  enabled: bool = True
  r: int = 8
  alpha: int = 16
  dropout: float = 0.05
  bias: Literal["none", "all", "lora_only"] = "none"
  apply_to: list[Literal["backbone", "longnet"]] = Field(
    default_factory=lambda: cast(list[Literal["backbone", "longnet"]], ["backbone"])
  )
  target_modules: list[str] = Field(
    default_factory=lambda: [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "out_proj",
      "proj",
      "qkv",
    ]
  )

  @model_validator(mode="after")
  def _validate(self) -> "LoraCfg":
    if self.r <= 0:
      raise ValueError("model.lora.r must be > 0.")
    if self.alpha <= 0:
      raise ValueError("model.lora.alpha must be > 0.")
    if self.dropout < 0 or self.dropout >= 1:
      raise ValueError("model.lora.dropout must be in [0, 1).")
    if self.enabled and not self.apply_to:
      raise ValueError("model.lora.apply_to must be non-empty when lora is enabled.")
    if self.enabled and not self.target_modules:
      raise ValueError("model.lora.target_modules must be non-empty when lora is enabled.")
    return self


class ModelCfg(BaseModel):
  name: Literal["unet", "timm", "prov_gigapath", "uni2_h", "virchow2"] = "prov_gigapath"
  # Common
  in_channels: int = 3
  num_classes: Optional[int] = None

  # Optional weights (used by timm/prov/uni2_h as appropriate)
  weights: Optional[WeightsCfg] = None
  longnet_pretrained: bool = False
  longnet_weights: Optional[WeightsCfg] = None
  lora: LoraCfg = Field(default_factory=LoraCfg)

  # Unet-specific
  encoder_name: str = "resnet50"
  encoder_weights: Optional[str] = "imagenet"
  activation: Optional[str] = None

  # timm-specific (also used as a generic "arch" string for other backbones)
  arch: Optional[str] = None
  pretrained: bool = False

  # --- foundation model backbones ---
  freeze_backbone: bool = True

  # Foundation output dimension (used by MIL head)
  foundation_dim: Optional[int] = None

  # LongNet MIL head config
  longnet_dim: int = 1536
  longnet_depth: int = 2
  # Grid resolution for slide-level positional embeddings.
  # Default 1000 creates a 1000×1000 grid (1M positions), covering slides up to
  # ~256K×256K pixels with 256px tiles. Must match pretrained weights if using
  # longnet_pretrained=true. Adjust for very large/small WSIs or memory constraints.
  longnet_slide_ngrids: int = 1000
  longnet_max_wsi_size: int = 262144
  longnet_dropout: float = 0.25
  input_size: Optional[int] = 256

  # Aggregator selection and input conditioning
  aggregator_type: Literal["longnet", "mean_pool", "max_pool"] = "longnet"
  longnet_input_norm: bool = True
  longnet_input_dropout: float = 0.1

  @model_validator(mode="after")
  def _validate(self) -> "ModelCfg":
    if self.in_channels <= 0:
      raise ValueError("model.in_channels must be > 0.")

    # For timm / prov_gigapath / uni2_h, arch is expected (even if it's just an identifier)
    if self.name in {"timm", "prov_gigapath", "uni2_h", "virchow2"}:
      if not self.arch:
        raise ValueError(f"model.arch is required when model.name='{self.name}'.")

    if self.longnet_pretrained and self.longnet_weights is None:
      raise ValueError("model.longnet_weights is required when model.longnet_pretrained is true.")

    if self.longnet_dim <= 0:
      raise ValueError("model.longnet_dim must be > 0.")
    if self.longnet_depth <= 0:
      raise ValueError("model.longnet_depth must be > 0.")
    if self.longnet_slide_ngrids <= 0:
      raise ValueError("model.longnet_slide_ngrids must be > 0.")
    if self.longnet_max_wsi_size <= 0:
      raise ValueError("model.longnet_max_wsi_size must be > 0.")
    if self.longnet_dropout < 0 or self.longnet_dropout >= 1:
      raise ValueError("model.longnet_dropout must be in [0, 1).")
    if self.longnet_input_dropout < 0 or self.longnet_input_dropout >= 1:
      raise ValueError("model.longnet_input_dropout must be in [0, 1).")

    return self

# -------------------------
# Data / Train / Runtime / Export
# -------------------------
class MILCacheCfg(BaseModel):
  enabled: bool = False
  cache_subdir: str = "mil_tiles_cache"
  chunk_size: int = 64
  compression: Literal["none", "blosc"] = "none"
  memory_max_slides: int = 0
  cache_pooled_embeddings: bool = False
  pooled_embeddings_key: str = "pooled_emb"
  pooled_embeddings_tag: Optional[str] = None

  @model_validator(mode="after")
  def _validate(self) -> "MILCacheCfg":
    if not self.cache_subdir:
      raise ValueError("data.mil_cache.cache_subdir must be non-empty.")
    if self.chunk_size <= 0:
      raise ValueError("data.mil_cache.chunk_size must be > 0.")
    if self.memory_max_slides < 0:
      raise ValueError("data.mil_cache.memory_max_slides must be >= 0.")
    if not self.pooled_embeddings_key:
      raise ValueError("data.mil_cache.pooled_embeddings_key must be non-empty.")
    return self


class DataCfg(BaseModel):
  patch_size: int = 256
  train_ratio: float = 0.6
  test_ratio: float = 0.2
  val_ratio: float = 0.2
  num_workers: int = 4
  pin_memory: bool = True
  mil_cache: MILCacheCfg = Field(default_factory=MILCacheCfg)

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


class PreprocessDataCfg(BaseModel):
  patch_size: int = 256

  @model_validator(mode="after")
  def _validate(self) -> "PreprocessDataCfg":
    if self.patch_size <= 0:
      raise ValueError("data.patch_size must be > 0.")
    return self


class PreprocessPathsCfg(BaseModel):
  run_root: Path
  cache_dir: Path
  label_csv: Path
  wsi_dir: Optional[Path] = None
  wsi_ext: str = ".svs"


class PreprocessCfg(BaseModel):
  level: int = 0
  margin: int = 0
  occupancy_threshold: float = 0.1
  foreground_threshold: Optional[float] = None
  hsv_s_threshold: Optional[float] = 0.05
  overwrite: bool = False
  tiles_key: str = "tiles"

  @model_validator(mode="after")
  def _validate(self) -> "PreprocessCfg":
    if self.level < 0:
      raise ValueError("preprocess.level must be >= 0.")
    if self.margin < 0:
      raise ValueError("preprocess.margin must be >= 0.")
    if not (0.0 <= self.occupancy_threshold <= 1.0):
      raise ValueError("preprocess.occupancy_threshold must be in [0, 1].")
    if self.hsv_s_threshold is not None and not (0.0 <= self.hsv_s_threshold <= 1.0):
      raise ValueError("preprocess.hsv_s_threshold must be in [0, 1] or None.")
    if not self.tiles_key:
      raise ValueError("preprocess.tiles_key must be non-empty.")
    return self


class PreprocessLoggingCfg(BaseModel):
  level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
  log_file: str = "preprocess.log"


class EarlyStoppingCfg(BaseModel):
  patience: int = 5
  min_delta: float = 1e-5
  mode: Literal["min", "max"] = "min"
  metric: Literal["val_loss", "val_auc", "val_macro_f1"] = "val_auc"

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

  # Class imbalance handling
  # Default to a single mechanism to avoid "double compensation".
  use_class_weights: bool = False
  use_balanced_sampling: bool = True

  # Optimizer settings
  weight_decay: float = 0.01
  head_lr_multiplier: float = 3.0

  # Scheduler settings
  warmup_epochs: int = 5
  scheduler_type: Literal["cosine", "step", "none"] = "cosine"
  gradient_clip: float = 1.0

  # Sanity check mode
  sanity_check: bool = False
  sanity_check_samples: int = 10

  @model_validator(mode="after")
  def _validate(self) -> "TrainCfg":
    if self.batch_size <= 0:
      raise ValueError("train.batch_size must be > 0.")
    if self.epochs <= 0:
      raise ValueError("train.epochs must be > 0.")
    if self.lr <= 0:
      raise ValueError("train.lr must be > 0.")
    if self.weight_decay < 0:
      raise ValueError("train.weight_decay must be >= 0.")
    if self.head_lr_multiplier <= 0:
      raise ValueError("train.head_lr_multiplier must be > 0.")
    if self.warmup_epochs < 0:
      raise ValueError("train.warmup_epochs must be >= 0.")
    if self.gradient_clip < 0:
      raise ValueError("train.gradient_clip must be >= 0.")
    if self.sanity_check_samples <= 0:
      raise ValueError("train.sanity_check_samples must be > 0.")
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


class InferenceCfg(BaseModel):
  wsi_path: Path
  slide_id: Optional[str] = None
  cache_dir: Optional[Path] = None
  patch_dir: Optional[Path] = None
  output_dir: Optional[Path] = None
  output_json: str = "prediction.json"
  tile_size: Optional[int] = None
  cleanup_tiles: bool = False
  preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
  fallback_weights: Optional[Path] = None


class MlflowCfg(BaseModel):
  enabled: bool = True
  registry_model_name: str
  selection_metric: str = "val_iou"
  load_selection_metric: Optional[str] = None
  tracking_uri: Optional[str] = "http://localhost:5000"


class DatasetCfg(BaseModel):
  paths: PathsCfg
  task: TaskCfg
  data: DataCfg

class CoreCfg(BaseModel):
  runtime: RuntimeCfg
  export: ExportCfg

# -------------------------
# App config (cross-field validation)
# -------------------------
class AppCfg(BaseModel):
  model: ModelCfg
  dataset: DatasetCfg
  train: TrainCfg
  core: CoreCfg
  mlflow: MlflowCfg
  inference: Optional[InferenceCfg] = None

  # Hydra run dir is injected by CONFIG()
  run_dir: Optional[Path] = None

  @model_validator(mode="after")
  def _validate_cross_fields(self) -> "AppCfg":
    data = self.dataset.data
    task = self.dataset.task
    paths = self.dataset.paths
    model = self.model

    # MIL Task Requirements
    if isinstance(task, MILTaskCfg):
      if paths.label_csv is None:
        raise ValueError("dataset.paths.label_csv is required for MIL tasks.")
      if model.name not in ("timm", "prov_gigapath", "uni2_h", "virchow2"):
        raise ValueError("For MIL tasks, set model.name in {'timm','prov_gigapath','uni2_h','virchow2'}.")

      # If model.num_classes provided, must match task.num_classes
      if model.num_classes is not None and model.num_classes != task.num_classes:
        raise ValueError(
          f"model.num_classes ({model.num_classes}) must equal dataset.task.num_classes ({task.num_classes})."
        )

      # If model.input_size is provided, it must not exceed the tile size.
      if model.input_size is not None and model.input_size > data.patch_size:
        raise ValueError(
          f"model.input_size ({model.input_size}) must be <= data.patch_size ({data.patch_size})."
        )


    # Segmentation requirements
    if isinstance(task, SegTaskCfg):
      if model.name not in ("unet", "timm", "prov_gigapath", "uni2_h", "virchow2"):
        raise ValueError(
          "For segmentation tasks, set model.name in "
          "{'unet','timm','prov_gigapath','uni2_h','virchow2'}."
        )

    return self


class PreprocessAppCfg(BaseModel):
  paths: PreprocessPathsCfg
  data: PreprocessDataCfg
  preprocess: PreprocessCfg
  logging: PreprocessLoggingCfg = Field(default_factory=PreprocessLoggingCfg)


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
  patch_size: int = 256
  stride: int = 256
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
