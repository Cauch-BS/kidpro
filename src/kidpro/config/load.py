from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from ..config.schema import AppCfg
from ..utils.seed import seed_everything


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
  cfg_dict["run_dir"] = str(run_dir)

  cfg = AppCfg.model_validate(cfg_dict)
  rr = resolve_device(cfg.runtime.device)

  torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark
  torch.backends.cudnn.deterministic = cfg.runtime.cudnn_deterministic

  seed_everything(cfg.train.seed, cuda=rr.cuda_available)
  return cfg, rr


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
    env = {
      "task_type": cfg.dataset.task.type,
      "layer_ids": cfg.dataset.task.layer_ids,
      "patch_size": cfg.dataset.data.patch_size,
      "batch_size": cfg.train.batch_size,
      "epochs": cfg.train.epochs,
      "lr": cfg.train.lr,
      "test_ratio": cfg.dataset.data.test_ratio,
      "val_ratio": cfg.dataset.data.val_ratio,
      "device": rr.device,
      "cuda_available": rr.cuda_available,
      "torch_version": torch.__version__,
      "python_version": platform.python_version(),
      **_git_info(),
    }
    with open(run_dir / cfg.export.env_json_name, "w") as f:
      json.dump(env, f, indent=2)
