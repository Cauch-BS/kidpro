from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from ...config.schema import AppCfg


# -------------------------
# Public return type
# -------------------------
@dataclass(frozen=True)
class FoundationBackbone:
    backbone: nn.Module
    feat_dim: int


# -------------------------
# Shared utils
# -------------------------
def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def load_state_dict_generic(model: nn.Module, ckpt_path: Path) -> None:
    """
    Load a checkpoint into `model` with common head-stripping.
    Works with .pt/.pth (torch.load) and .safetensors if installed.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state: dict[str, torch.Tensor]
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install with: pip install safetensors"
            ) from e
        state = load_file(str(ckpt_path))
    else:
        try:
            obj = torch.load(str(ckpt_path), map_location="cpu")
            state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {ckpt_path}. "
                "The file may be corrupted or not a valid PyTorch checkpoint."
            ) from e

    # Strip typical classifier / head parameters
    head_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
    filtered = {k: v for k, v in state.items() if not k.startswith(head_prefixes)}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("[FND CKPT]", "missing:", missing[:8], "unexpected:", unexpected[:8])


def resolve_weights_path(cfg: AppCfg) -> Optional[Path]:
    """
    Priority:
      1) model.weights.*
    """
    m = cfg.model

    w = getattr(m, "weights", None)
    if w is not None:
        if w.source == "local":
            return Path(w.local_path)  # type: ignore[arg-type]
        if w.source == "hf_cache":
            return Path(w.hf_cache_path)  # type: ignore[arg-type]
        raise ValueError(f"Unknown weights.source={w.source!r}")

    return None


def infer_feat_dim(backbone: nn.Module, cfg: AppCfg) -> int:
    """
    Prefer explicit config for non-standard backbones.
    Fallback to backbone.num_features if present.
    """
    if getattr(cfg.model, "foundation_dim", None) is not None:
        return int(cfg.model.foundation_dim)  # type: ignore[arg-type]
    nf = getattr(backbone, "num_features", None)
    if nf is None:
        raise ValueError(
            "Cannot infer foundation feature dim. "
            "Set model.foundation_dim in config, or use a backbone exposing .num_features."
        )
    return int(nf)


def discover_foundation_builders(
    addon_paths: Optional[list[str]] = None,
) -> dict[str, Callable[[AppCfg], FoundationBackbone]]:
    """
    Discovers builders from:
      1) built-in modules in kidpro.modeling.sources.*
      2) optional external addon module files

    Contract per module:
      - FOUNDATION_NAME: str
      - build(cfg: AppCfg) -> FoundationBackbone
    """
    import importlib
    import os
    import pkgutil
    import sys

    from . import __name__, __path__

    builders: dict[str, Callable[[AppCfg], FoundationBackbone]] = {}

    def scan_module(mod: Any) -> None:
        fnd_name = getattr(mod, "FOUNDATION_NAME", None)
        fn = getattr(mod, "build", None)
        if isinstance(fnd_name, str) and callable(fn):
            if fnd_name in builders:
                raise RuntimeError(
                    f"Duplicate FOUNDATION_NAME={fnd_name!r} from module {mod.__name__}. "
                    "Each source must provide a unique FOUNDATION_NAME."
                )
            builders[fnd_name] = fn

    # 1) built-in modules
    for modinfo in pkgutil.iter_modules(__path__):
        if modinfo.name.startswith("_"):
            continue
        modname = f"{__name__}.{modinfo.name}"
        mod = importlib.import_module(modname)
        scan_module(mod)

    # 2) addon modules (optional)
    addon_paths = addon_paths or []
    loaded = set()
    for filepath in addon_paths:
        abspath = os.path.abspath(filepath)
        if abspath in loaded:
            continue
        loaded.add(abspath)

        modname = os.path.splitext(os.path.basename(abspath))[0]
        dirname = os.path.dirname(abspath)
        if dirname not in sys.path:
            sys.path.insert(0, dirname)

        mod = importlib.import_module(modname)
        scan_module(mod)

    return builders


# Eager registry (built-ins only). You can switch to lazy if you prefer.
BUILDER_REGISTRY = discover_foundation_builders(addon_paths=[])


def available_foundations() -> list[str]:
    return sorted(BUILDER_REGISTRY.keys())


def build_foundation(cfg: AppCfg) -> FoundationBackbone:
    """
    Uses cfg.model.name as the registry key (FOUNDATION_NAME).
    """
    name = cfg.model.name
    if name not in BUILDER_REGISTRY:
        raise ValueError(
            f"Unknown foundation model.name={name!r}. "
            f"Available={available_foundations()}. "
            "If you added a new module under modeling/sources, ensure it defines "
            "FOUNDATION_NAME and build(cfg)->FoundationBackbone."
        )
    return BUILDER_REGISTRY[name](cfg)
