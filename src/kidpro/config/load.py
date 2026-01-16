from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from ..utils.seed import seed_everything
from .schema import AppCfg, MILTaskCfg, PatchCfg, PreprocessAppCfg, SegTaskCfg


@dataclass(frozen=True)
class RuntimeResolved:
  device: str  # "cpu" or "cuda"
  cuda_available: bool


def resolve_device(choice: str) -> RuntimeResolved:
  cuda_avail = torch.cuda.is_available()
  if choice == "auto":
    dev = "cuda" if cuda_avail else "cpu"
  elif choice == "cuda":
    dev = "cuda" if cuda_avail else "cpu"
  else:
    dev = "cpu"
  return RuntimeResolved(device=dev, cuda_available=cuda_avail)


def CONFIG(hcfg: DictConfig, run_dir: Path) -> Tuple[AppCfg, RuntimeResolved]:
  cfg_dict = OmegaConf.to_container(hcfg, resolve=True)

  # Inject run_dir (Hydra CWD)
  cfg_dict["run_dir"] = str(run_dir) # type: ignore

  cfg = AppCfg.model_validate(cfg_dict)
  rr = resolve_device(cfg.runtime.device)

  torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark
  torch.backends.cudnn.deterministic = cfg.runtime.cudnn_deterministic

  seed_everything(cfg.train.seed, cuda=rr.cuda_available)
  return cfg, rr


def PATCH_CONFIG(hcfg: DictConfig) -> PatchCfg:
  cfg_dict = OmegaConf.to_container(hcfg, resolve=True)
  if not isinstance(cfg_dict, dict) or "patch" not in cfg_dict:
    raise ValueError("Patch config must contain a top-level 'patch' key.")
  patch_cfg = cfg_dict["patch"]
  if not isinstance(patch_cfg, dict):
    raise ValueError("Patch config must be a mapping under the 'patch' key.")
  return PatchCfg.model_validate(patch_cfg) # type: ignore


def PREPROCESS_CONFIG(hcfg: DictConfig) -> PreprocessAppCfg:
  cfg_dict = OmegaConf.to_container(hcfg, resolve=True)
  if not isinstance(cfg_dict, dict):
    raise ValueError("Preprocess config must be a mapping.")
  return PreprocessAppCfg.model_validate(cfg_dict) # type: ignore


def _git_info() -> Dict[str, Any]:
  def run(cmd: list[str]) -> str | None:
    try:
      return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
      return None

  commit = run(["git", "rev-parse", "HEAD"])
  dirty = run(["git", "status", "--porcelain"])
  return {
    "git_commit": commit,
    "git_dirty": bool(dirty),
  }


def CONFIG_EXPORT(cfg: AppCfg, rr: RuntimeResolved) -> None:
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()

  if cfg.export.save_resolved_config:
    resolved = cfg.model_dump(mode="python")
    OmegaConf.save(
      config=OmegaConf.create(resolved),
      f=str(run_dir / cfg.export.resolved_config_name),
    )

  if cfg.export.save_env_json:
    task = cfg.dataset.task
    data = cfg.dataset.data
    paths = cfg.dataset.paths
    model = cfg.model

    env: Dict[str, Any] = {
      # Task / dataset identity
      "task_type": task.type,
      "dataset_root_dir": str(paths.root_dir),
      "dataset_csv_name": paths.csv_name,
      "runs_root": str(paths.runs_root),

      # Data / training
      "patch_size": data.patch_size,
      "batch_size": cfg.train.batch_size,
      "epochs": cfg.train.epochs,
      "lr": cfg.train.lr,
      "test_ratio": data.test_ratio,
      "val_ratio": data.val_ratio,
      "num_workers": data.num_workers,
      "pin_memory": data.pin_memory,

      # Runtime
      "device": rr.device,
      "cuda_available": rr.cuda_available,
      "torch_version": torch.__version__,
      "python_version": platform.python_version(),

      # Model (useful for both seg + mil)
      "model_name": model.name,
      "model_arch": getattr(model, "arch", None),
      "model_encoder_name": getattr(model, "encoder_name", None),
      "model_encoder_weights": getattr(model, "encoder_weights", None),
      "model_in_channels": model.in_channels,
      "model_num_classes": getattr(model, "num_classes", None),
      "model_pretrained": getattr(model, "pretrained", None),

      # Provenance
      **_git_info(),
    }

    # --- Task-specific fields (safe, no assumptions) ---
    if isinstance(task, SegTaskCfg):
      env["layer_ids"] = task.layer_ids
    elif isinstance(task, MILTaskCfg):
      env["task_num_classes"] = task.num_classes

    # Dataset-specific metadata (e.g., classification label CSV)
    if getattr(paths, "label_csv", None) is not None:
      env["label_csv"] = str(paths.label_csv)

    with open(run_dir / cfg.export.env_json_name, "w") as f:
      json.dump(env, f, indent=2)
