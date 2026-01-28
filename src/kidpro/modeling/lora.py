from __future__ import annotations

import inspect
import logging
import types
from typing import Any, cast

import torch.nn as nn

from ..config.schema import AppCfg
from .sources import freeze_module

log = logging.getLogger(__name__)


def _has_lora_targets(module: nn.Module, target_modules: list[str]) -> bool:
  for name, _ in module.named_modules():
    for target in target_modules:
      if name == target or name.endswith(f".{target}"):
        return True
  return False


def apply_lora(cfg: AppCfg, encoder: nn.Module, freeze_base: bool) -> Any:
  lora_cfg = cfg.model.lora
  if not lora_cfg.enabled:
    return encoder

  if freeze_base:
    freeze_module(encoder)

  if not _has_lora_targets(encoder, lora_cfg.target_modules):
    log.warning(
      "LoRA enabled but no target modules matched. "
      "Skipping LoRA wrap; encoder remains frozen."
    )
    return encoder

  _patch_forward_for_peft(encoder)

  from peft import LoraConfig, TaskType, get_peft_model

  peft_cfg = LoraConfig(
    r=lora_cfg.r,
    lora_alpha=lora_cfg.alpha,
    lora_dropout=lora_cfg.dropout,
    bias=lora_cfg.bias,
    target_modules=lora_cfg.target_modules,
    task_type=TaskType.FEATURE_EXTRACTION,
  )

  return get_peft_model(cast(Any, encoder), peft_cfg)

def _patch_forward_for_peft(encoder: nn.Module) -> None:
  if getattr(encoder, "_kidpro_peft_forward_patched", False):
    return

  try:
    sig = inspect.signature(encoder.forward)
  except (TypeError, ValueError):
    return

  params = sig.parameters
  if "input_ids" in params:
    return
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
    return

  original_forward = encoder.forward

  def _forward(  # type: ignore[no-untyped-def]
    self,
    input_ids=None,
    x=None,
    pixel_values=None,
    **kwargs,
  ):
    if x is None:
      if input_ids is not None:
        x = input_ids
      elif pixel_values is not None:
        x = pixel_values
      elif "inputs_embeds" in kwargs:
        x = kwargs["inputs_embeds"]
    if x is None:
      raise TypeError(
        "Patched encoder forward requires an image tensor (input_ids, x, or pixel_values)."
      )
    return original_forward(x)

  encoder.forward = types.MethodType(_forward, encoder)
  setattr(encoder, "_kidpro_peft_forward_patched", True)
