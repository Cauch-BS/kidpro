from __future__ import annotations

import inspect
import logging
from typing import Any, cast

import torch.nn as nn

from ..config.schema import AppCfg
from ..utils.model_io import freeze_module

log = logging.getLogger(__name__)


class _PeftInputAdapter(nn.Module):
  """
  Adapter to make non-HF encoders compatible with PEFT's
  `TaskType.FEATURE_EXTRACTION` wrapper, which forwards `input_ids`/`pixel_values`.

  This avoids monkey-patching `encoder.forward` while still letting call sites do
  `encoder(x)` (positional) or `encoder(pixel_values=x)` (keyword).
  """

  def __init__(self, encoder: nn.Module) -> None:
    super().__init__()
    self.encoder = encoder
    setattr(self, "_kidpro_peft_input_adapter", True)

  def forward(  # type: ignore[no-untyped-def]
    self,
    input_ids=None,
    x=None,
    pixel_values=None,
    inputs_embeds=None,
    **kwargs,
  ):
    if x is None:
      if input_ids is not None:
        x = input_ids
      elif pixel_values is not None:
        x = pixel_values
      elif inputs_embeds is not None:
        x = inputs_embeds
      elif "inputs_embeds" in kwargs:
        x = kwargs["inputs_embeds"]
    if x is None:
      raise TypeError(
        "LoRA encoder expected an image tensor via input_ids, x, pixel_values, or inputs_embeds."
      )
    return self.encoder(x)

  def forward_features(self, x: Any) -> Any:
    ff = getattr(self.encoder, "forward_features", None)
    if callable(ff):
      return ff(x)
    return self.encoder(x)

  def __getattr__(self, name: str) -> Any:
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.encoder, name)


def _needs_peft_input_adapter(encoder: nn.Module) -> bool:
  if getattr(encoder, "_kidpro_peft_input_adapter", False):
    return False
  try:
    sig = inspect.signature(encoder.forward)
  except (TypeError, ValueError):
    # If we can't introspect, don't wrap; let runtime errors surface naturally.
    return False

  params = sig.parameters
  if "input_ids" in params:
    return False
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
    return False
  return True


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
      "Skipping LoRA wrap; encoder left unchanged."
    )
    return encoder

  if _needs_peft_input_adapter(encoder):
    encoder = _PeftInputAdapter(encoder)

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
