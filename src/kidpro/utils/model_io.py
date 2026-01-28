from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn as nn

from ..config.schema import AppCfg

log = logging.getLogger(__name__)


def freeze_module(module: nn.Module) -> None:
  for p in module.parameters():
    p.requires_grad = False


def _as_state_dict(obj: object) -> dict[str, torch.Tensor]:
  if isinstance(obj, nn.Module):
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in obj.state_dict().items()}

  if isinstance(obj, Mapping):
    for key in ("state_dict", "model_state_dict", "model", "module", "net", "weights"):
      if key in obj and isinstance(obj[key], Mapping):
        return dict(obj[key])
      if key in obj and isinstance(obj[key], nn.Module):
        return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in obj[key].state_dict().items()}
    if all(isinstance(v, torch.Tensor) for v in obj.values()):
      return dict(obj)
    tensor_only = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if tensor_only:
      warnings.warn(
        "Checkpoint contains non-tensor metadata; using tensor entries only.",
        RuntimeWarning,
      )
      return tensor_only

  raise TypeError(
    f"Unsupported checkpoint payload type: {type(obj)}. "
    "Expected state_dict-like mapping or {'state_dict': ...}."
  )


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  if state and all(k.startswith("module.") for k in state):
    return {k[len("module."):]: v for k, v in state.items()}
  return dict(state)


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
  if not prefix:
    return dict(state)
  return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}


def _has_lora_keys(state: Mapping[str, torch.Tensor]) -> bool:
  for k in state.keys():
    if "lora_" in k or "modules_to_save" in k:
      return True
  return False


def load_state_dict_generic(
  model: nn.Module,
  ckpt_path: Path,
  *,
  drop_heads: bool = True,
  include_prefixes: tuple[str, ...] | None = None,
  exclude_prefixes: tuple[str, ...] | None = None,
  ckpt_prefix: str | None = None,
) -> nn.Module:
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
    except (pickle.UnpicklingError, RuntimeError):
      log.warning(
        "[WARNING] Checkpoint requires pickle deserialization (full object). "
        "Retrying with weights_only=False. Do this only for trusted checkpoints."
      )
      obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
      raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}: {e}") from e

    state = _as_state_dict(obj)

  state = _strip_module_prefix(state)
  if ckpt_prefix:
    state = {k: v for k, v in state.items() if k.startswith(ckpt_prefix)}
    state = _strip_prefix(state, ckpt_prefix)
  if exclude_prefixes:
    state = {k: v for k, v in state.items() if not k.startswith(exclude_prefixes)}
  has_lora = _has_lora_keys(state)

  def _format_ckpt_msg(
    *,
    missing: list[str],
    unexpected: list[str],
    dropped_lora: int = 0,
    peft_merge: bool = False,
  ) -> str:
    # If caller restricted loading to specific prefixes, ignore unrelated missing keys
    # to reduce noise (e.g. loading only slide_encoder.* + classifier.* into a MIL model).
    missing_rel = (
      [k for k in missing if k.startswith(include_prefixes)]  # type: ignore[arg-type]
      if include_prefixes
      else missing
    )
    base = "[FND CKPT] "
    if peft_merge:
      base += "peft_merge=1 "
    else:
      base += f"dropped_lora={dropped_lora} "
    base += (
      f"missing={len(missing_rel)} (showing up to 8): {missing_rel[:8]} "
      f"unexpected={len(unexpected)} (showing up to 8): {unexpected[:8]}"
    )
    return base

  if has_lora and callable(getattr(model, "merge_and_unload", None)):
    # Some checkpoints store weights under a full model path; normalize.
    state = _strip_prefix(state, "backbone.tile_encoder.")
    if ckpt_prefix:
      state = {k: v for k, v in state.items() if k.startswith(ckpt_prefix)}
      state = _strip_prefix(state, ckpt_prefix)
    if exclude_prefixes:
      state = {k: v for k, v in state.items() if not k.startswith(exclude_prefixes)}
    if drop_heads:
      # Drop task-specific heads/decoders for backbone-only loads (tile encoders).
      head_prefixes = (
        "fc.",
        "classifier.",
        "head.",
        "last_linear.",
        "decoder.",
        "decode_head.",
        "seg_head.",
        "segmentation_head.",
      )
      state = {k: v for k, v in state.items() if not k.startswith(head_prefixes)}
    if include_prefixes:
      state = {k: v for k, v in state.items() if k.startswith(include_prefixes)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    merged = model.merge_and_unload()  # type: ignore[operator]
    if merged is None:
      merged = model
    msg = _format_ckpt_msg(missing=missing, unexpected=unexpected, peft_merge=True)
    log.info(msg)
    return cast(nn.Module, merged)

  if has_lora:
    warnings.warn(
      "LoRA weights found in checkpoint, but model is not a PEFT model. Dropping LoRA adapters.",
      RuntimeWarning,
    )

  # Many checkpoints in this repo are *backbone-only* (e.g. timm encoders) and we
  # intentionally drop classifier heads when loading them. For full-model
  # checkpoints (e.g. WSI MIL), callers must set drop_heads=False.
  if drop_heads:
    head_prefixes = (
      "fc.",
      "classifier.",
      "head.",
      "last_linear.",
      # Common decoder-style heads (e.g. segmentation) that we never want when
      # loading a backbone/tile encoder checkpoint.
      "decoder.",
      "decode_head.",
      "seg_head.",
      "segmentation_head.",
    )
    filtered = {k: v for k, v in state.items() if not k.startswith(head_prefixes)}
  else:
    filtered = dict(state)

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

  if include_prefixes:
    stripped = {k: v for k, v in stripped.items() if k.startswith(include_prefixes)}

  if collisions:
    raise RuntimeError(
      "Key collision(s) after prefix stripping, e.g. "
      + ", ".join([f"{out_k} <- {src}" for out_k, src in collisions[:5]])
      + (" ..." if len(collisions) > 5 else "")
    )

  missing, unexpected = model.load_state_dict(stripped, strict=False)
  msg = _format_ckpt_msg(missing=missing, unexpected=unexpected, dropped_lora=dropped_lora, peft_merge=False)
  log.info(msg)
  return model


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
