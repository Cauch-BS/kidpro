from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys
from typing import Any, Callable, Optional

import torch.nn as nn

from ...config.schema import AppCfg

logger = logging.getLogger(__name__)


from ._types import MILTemplate, SlideEncoderBackbone

__all__ = [
    "MILTemplate",
    "SlideEncoderBackbone",
    "build_slide_encoder",
    "build_mil_model",
    "available_slide_encoders",
    "discover_slide_encoder_builders",
    "SLIDE_ENCODER_REGISTRY",
]


def discover_slide_encoder_builders(
    addon_paths: Optional[list[str]] = None,
) -> dict[str, Callable[..., Any]]:
    """
    Discovers builders from:
      1) built-in modules in kidpro.modeling.agg.*
      2) optional external addon module files

    Contract per module:
      - AGGREGATOR_NAME: str
      - build_mil(cfg: AppCfg, tile_encoder: nn.Module, num_classes: int) -> MILTemplate (preferred)
      - OR build(cfg: AppCfg) -> SlideEncoderBackbone (for aggregators)
      - OR build(cfg: AppCfg, tile_encoder: nn.Module, num_classes: int) -> MILTemplate (for complete models)
    """
    builders: dict[str, Callable[..., Any]] = {}

    def scan_module(mod: Any) -> None:
        agg_name = getattr(mod, "AGGREGATOR_NAME", None)
        # Prefer build_mil if it exists, otherwise fall back to build
        fn = getattr(mod, "build_mil", None) or getattr(mod, "build", None)
        if isinstance(agg_name, str) and callable(fn):
            if agg_name in builders:
                raise RuntimeError(
                    f"Duplicate AGGREGATOR_NAME={agg_name!r} from module {mod.__name__}. "
                    "Each source must provide a unique AGGREGATOR_NAME."
                )
            builders[agg_name] = fn

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
SLIDE_ENCODER_REGISTRY = discover_slide_encoder_builders(addon_paths=[])


def available_slide_encoders() -> list[str]:
    return sorted(SLIDE_ENCODER_REGISTRY.keys())


# Mapping from legacy/alias names to registry names
_AGGREGATOR_ALIASES: dict[str, str] = {
    "mean_pool": "simple",
    "max_pool": "simple",
}


def build_slide_encoder(cfg: AppCfg) -> SlideEncoderBackbone:
    """
    Uses cfg.model.aggregator_type as the registry key (AGGREGATOR_NAME).

    Supports aliases:
      - "mean_pool" -> "simple" (with pool_type=mean)
      - "max_pool" -> "simple" (with pool_type=max)
    """
    name = cfg.model.aggregator_type
    registry_name = _AGGREGATOR_ALIASES.get(name, name)

    if registry_name not in SLIDE_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown slide encoder aggregator_type={name!r}. "
            f"Available={available_slide_encoders() + list(_AGGREGATOR_ALIASES.keys())}. "
            "If you added a new module under modeling/agg, ensure it defines "
            "AGGREGATOR_NAME and build(cfg)->SlideEncoderBackbone."
        )
    return SLIDE_ENCODER_REGISTRY[registry_name](cfg) # type: ignore[no-any-return]


def build_mil_model(
    cfg: AppCfg,
    tile_encoder: nn.Module,
    num_classes: int,
) -> MILTemplate:
    """
    Build a complete MIL model with configurable slide aggregator.

    Uses cfg.model.aggregator_type to select the aggregator from the registry.
    All aggregators must provide build_mil() that returns a complete MILTemplate.

    Args:
        cfg: Application configuration
        tile_encoder: The tile encoder module (foundation backbone)
        num_classes: Number of output classes for the classifier

    Returns:
        Complete MIL model (MILTemplate) ready for training/inference
    """
    name = cfg.model.aggregator_type
    registry_name = _AGGREGATOR_ALIASES.get(name, name)

    if registry_name not in SLIDE_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown slide encoder aggregator_type={name!r}. "
            f"Available={available_slide_encoders() + list(_AGGREGATOR_ALIASES.keys())}. "
            "If you added a new module under modeling/agg, ensure it defines "
            "AGGREGATOR_NAME and build_mil(cfg, tile_encoder, num_classes)->MILTemplate."
        )

    builder = SLIDE_ENCODER_REGISTRY[registry_name]
    return builder(cfg, tile_encoder, num_classes) # type: ignore[no-any-return]
