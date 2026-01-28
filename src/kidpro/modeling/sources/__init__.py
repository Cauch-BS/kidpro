from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch.nn as nn

from ...config.schema import AppCfg
from ...utils.model_io import (
    freeze_module,
    load_state_dict_generic,
    resolve_weights_path,
)

logger = logging.getLogger(__name__)


# -------------------------
# Public return type
# -------------------------
@dataclass(frozen=True)
class FoundationBackbone:
    backbone: nn.Module
    feat_dim: int


__all__ = [
    "FoundationBackbone",
    "freeze_module",
    "load_state_dict_generic",
    "resolve_weights_path",
    "infer_feat_dim",
    "discover_foundation_builders",
    "available_foundations",
    "build_foundation",
]


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
