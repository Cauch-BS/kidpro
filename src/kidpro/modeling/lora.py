from __future__ import annotations

import logging
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


def apply_lora(cfg: AppCfg, encoder: nn.Module, freeze_base: bool) -> nn.Module:
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
