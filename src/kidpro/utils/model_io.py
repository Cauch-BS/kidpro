from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn as nn

from ..config.schema import AppCfg

log = logging.getLogger(__name__)

# --- Checkpoint diagnostics -------------------------------------------------

def _top_level_prefixes(keys: Sequence[str], *, limit: int = 24) -> list[str]:
  """
  Return a short list of "top-level" key prefixes for debugging, e.g.:
    slide_encoder, classifier, tile_encoder, backbone, model, base_model, ...
  """
  prefixes: set[str] = set()
  for k in keys:
    if not k:
      continue
    prefixes.add(k.split(".", 1)[0])
  out = sorted(prefixes)
  if len(out) > limit:
    return out[:limit] + ["..."]
  return out


def _raise_empty_prefix_filter(*, ckpt_path: Path, ckpt_prefix: str, keys: Sequence[str]) -> None:
  tops = _top_level_prefixes(keys)
  sample = sorted(keys)[:8]
  raise RuntimeError(
    "Checkpoint prefix filter matched 0 keys. "
    f"ckpt_path={ckpt_path} ckpt_prefix={ckpt_prefix!r}. "
    f"Top-level prefixes seen: {tops}. "
    f"Sample keys: {sample}. "
    "This usually means you pointed at the wrong checkpoint (or a checkpoint saved without module prefixes)."
  )


# Task-specific heads/decoders that should be ignored when loading a backbone/tile encoder checkpoint.
_HEAD_PREFIXES: tuple[str, ...] = (
  "fc.",
  "classifier.",
  "head.",
  "last_linear.",
  # Common decoder-style heads (e.g. segmentation).
  "decoder.",
  "decode_head.",
  "seg_head.",
  "segmentation_head.",
)


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


def _replace_base_layer(
  state: Mapping[str, torch.Tensor],
  model: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
  """
  Normalize `.base_layer.` key drift between checkpoints and models.

  Depending on the LoRA implementation, modules may expose parameters as:
    - `...proj.weight`                 (plain module)
    - `...proj.base_layer.weight`      (wrapped module exposing base_layer)

  If `model` is provided, we remap keys *per-key* to best match the model's
  `state_dict()` keys (preferring identity, then dropping, then adding
  `.base_layer.`). If `model` is not provided (legacy behavior), we drop
  `.base_layer.`.
  """
  # Legacy behavior: always drop `.base_layer.`
  if model is None:
    return {k.replace(".base_layer.", "."): v for k, v in state.items()}

  try:
    model_keys = set(model.state_dict().keys())
  except Exception:
    # If we can't inspect the model keys, preserve legacy behavior.
    return {k.replace(".base_layer.", "."): v for k, v in state.items()}

  # Only insert `.base_layer` before common parameter leaves.
  param_leaves = {
    "weight",
    "bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
  }

  def _skip_key(k: str) -> bool:
    return ("lora_" in k) or ("modules_to_save" in k)

  def _drop(k: str) -> str:
    return k.replace(".base_layer.", ".")

  def _add(k: str) -> str:
    if ".base_layer." in k:
      return k
    parts = k.split(".")
    if len(parts) < 2 or parts[-1] not in param_leaves:
      return k
    parts.insert(-1, "base_layer")
    return ".".join(parts)

  remapped: dict[str, torch.Tensor] = {}
  collisions: list[tuple[str, str]] = []
  changed = 0
  for k, v in state.items():
    if _skip_key(k):
      kk = k
    else:
      # Prefer identity; otherwise try drop/add variants if they match model keys.
      kd = _drop(k)
      ka = _add(k)
      if k in model_keys:
        kk = k
      elif kd in model_keys:
        kk = kd
      elif ka in model_keys:
        kk = ka
      else:
        kk = k
    if kk != k:
      changed += 1
    if kk in remapped and remapped[kk] is not v:
      collisions.append((kk, k))
    remapped[kk] = v

  if collisions:
    raise RuntimeError(
      "Key collision(s) after base_layer normalization, e.g. "
      + ", ".join([f"{out_k} <- {src}" for out_k, src in collisions[:5]])
      + (" ..." if len(collisions) > 5 else "")
    )

  if changed:
    log.info("[FND CKPT] base_layer_norm changed=%d", changed)
  return remapped


def _remap_encoder_variant(
  state: Mapping[str, torch.Tensor],
  model: nn.Module,
) -> dict[str, torch.Tensor]:
  """
  Best-effort remap for a common structural drift between checkpoints and code:

  Some backbones are saved with an extra `.encoder.` segment (or without it),
  e.g.:
    - ckpt:  base_model.model.cls_token
    - model: base_model.model.encoder.cls_token

  We score a small set of candidate transforms against `model.state_dict()` keys
  and apply the best one only if it significantly improves matches.
  """
  state = dict(state)
  if not state:
    return state

  try:
    model_keys = set(model.state_dict().keys())
  except Exception:
    # Some wrappers may not expose state_dict reliably; don't remap in that case.
    return state

  if not model_keys:
    return state

  def _baseline_matches(keys: list[str]) -> int:
    return sum(1 for k in keys if k in model_keys)

  rules: list[tuple[str, str]] = [
    ("base_model.model.", "base_model.model.encoder."),
    ("model.", "model.encoder."),
    ("tile_encoder.", "tile_encoder.encoder."),
  ]

  def _add_encoder(k: str) -> str:
    out = k
    for plain, enc in rules:
      if plain and out.startswith(plain) and not out.startswith(enc):
        return enc + out[len(plain):]
    return out

  def _drop_encoder(k: str) -> str:
    out = k
    for plain, enc in rules:
      if out.startswith(enc):
        return plain + out[len(enc):]
    return out

  keys = list(state.keys())
  base = _baseline_matches(keys)
  add = _baseline_matches([_add_encoder(k) for k in keys])
  drop = _baseline_matches([_drop_encoder(k) for k in keys])

  best_name = "identity"
  best_fn = lambda k: k
  best = base
  if add > best:
    best = add
    best_name = "add_encoder"
    best_fn = _add_encoder
  if drop > best:
    best = drop
    best_name = "drop_encoder"
    best_fn = _drop_encoder

  # Require a meaningful improvement; avoid flapping on tiny diffs.
  # Absolute threshold handles small models; relative threshold handles large.
  improve = best - base
  if best_name == "identity" or (improve < 32 and improve < int(0.05 * max(1, base))):
    return state

  remapped: dict[str, torch.Tensor] = {}
  collisions: list[tuple[str, str]] = []
  for k, v in state.items():
    kk = best_fn(k)
    if kk in remapped and remapped[kk] is not v:
      collisions.append((kk, k))
    remapped[kk] = v

  if collisions:
    raise RuntimeError(
      "Key collision(s) after encoder remap, e.g. "
      + ", ".join([f"{out_k} <- {src}" for out_k, src in collisions[:5]])
      + (" ..." if len(collisions) > 5 else "")
    )

  log.info("[FND CKPT] key_remap=%s matched=%d->%d", best_name, base, best)
  return remapped


def load_state_dict_with_remap(
  model: nn.Module,
  ckpt_path: Path,
  *,
  strict: bool = True,
  drop_heads: bool = False,
  include_prefixes: tuple[str, ...] | None = None,
  exclude_prefixes: tuple[str, ...] | None = None,
  ckpt_prefix: str | None = None,
) -> nn.Module:
  """
  Load a *full model* checkpoint with key remapping, without the "backbone-only"
  prefix-stripping logic in `load_state_dict_generic`.

  This is intended for cases like loading an older `best_model.pt` into a newer
  model class where wrapper structure changed (e.g. `tile_encoder.*` vs
  `tile_encoder.encoder.*`, or `.base_layer.` drift).
  """
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
    keys_before = list(state.keys())
    state = {k: v for k, v in state.items() if k.startswith(ckpt_prefix)}
    if not state:
      _raise_empty_prefix_filter(ckpt_path=ckpt_path, ckpt_prefix=str(ckpt_prefix), keys=keys_before)
    state = _strip_prefix(state, ckpt_prefix)
  if exclude_prefixes:
    state = {k: v for k, v in state.items() if not k.startswith(exclude_prefixes)}
  if drop_heads:
    state = {k: v for k, v in state.items() if not k.startswith(_HEAD_PREFIXES)}
  if include_prefixes:
    state = {k: v for k, v in state.items() if k.startswith(include_prefixes)}

  # Remap known structural drift against the *current model*.
  state = _replace_base_layer(state, model=model)
  state = _remap_encoder_variant(state, model)

  model.load_state_dict(state, strict=strict)
  return model


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
    keys_before = list(state.keys())
    state = {k: v for k, v in state.items() if k.startswith(ckpt_prefix)}
    if not state:
      _raise_empty_prefix_filter(ckpt_path=ckpt_path, ckpt_prefix=str(ckpt_prefix), keys=keys_before)
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
    state = _replace_base_layer(state, model=model)
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

    # Handle common `.encoder.` vs no-`.encoder.` drift for PEFT-wrapped models.
    state = _remap_encoder_variant(state, model)

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

  # Same `.encoder.` drift can occur for non-PEFT backbones too (e.g. wrapper refactors).
  stripped = _remap_encoder_variant(stripped, model)

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
