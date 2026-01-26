from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
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


def _as_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in obj.state_dict().items()}

    if isinstance(obj, Mapping):
        if "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            return dict(obj["state_dict"])
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)

    raise TypeError(
        f"Unsupported checkpoint payload type: {type(obj)}. "
        "Expected state_dict-like mapping or {'state_dict': ...}."
    )

def load_state_dict_generic(model: nn.Module, ckpt_path: Path) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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
            try:
                obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(str(ckpt_path), map_location="cpu")
        except (pickle.UnpicklingError, RuntimeError) as e:
            warnings.warn(
                "Checkpoint requires pickle deserialization (full object). "
                "Retrying with weights_only=False. Do this only for trusted checkpoints.",
                RuntimeWarning,
            )
            obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {ckpt_path}: {e}"
            ) from e

        state = _as_state_dict(obj)

    head_prefixes = ("fc.", "classifier.", "head.", "last_linear.")
    filtered = {k: v for k, v in state.items() if not k.startswith(head_prefixes)}

    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "tile_encoder.",
        "tile_encoder.base_model.",
        "tile_encoder.base_model.model.",
        "base_model.",
        "base_model.model.",
    )
    stripped: dict[str, torch.Tensor] = {}
    collisions: list[tuple[str, str]] = []
    dropped_lora = 0

    def _normalize_key(key: str) -> Optional[str]:
        if "lora_" in key or "modules_to_save" in key:
            return None
        out_k = key
        while True:
            matched = False
            for prefix in prefixes:
                if out_k.startswith(prefix):
                    out_k = out_k[len(prefix):]
                    matched = True
                    break
            if not matched:
                break
        out_k = out_k.replace(".base_layer.", ".")
        return out_k

    for k, v in filtered.items():
        out_k = _normalize_key(k)
        if out_k is None:
            dropped_lora += 1
            continue
        if out_k in stripped and stripped[out_k] is not v:
            collisions.append((out_k, k))
        stripped[out_k] = v

    if collisions:
        raise RuntimeError(
            "Key collision(s) after prefix stripping, e.g. "
            + ", ".join([f"{out_k} <- {src}" for out_k, src in collisions[:5]])
            + (" ..." if len(collisions) > 5 else "")
        )

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(
        "[FND CKPT]",
        f"dropped_lora={dropped_lora}",
        f"missing={len(missing)} (showing up to 8): {missing[:8]}",
        f"unexpected={len(unexpected)} (showing up to 8): {unexpected[:8]}",
    )


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
