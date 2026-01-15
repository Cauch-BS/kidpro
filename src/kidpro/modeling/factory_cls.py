from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from ..config.schema import AppCfg


def build_model_cls(cfg: AppCfg) -> Module:
  if cfg.dataset.task.type != "classification":
    raise ValueError(
      f"build_model_cls called with dataset.task.type={cfg.dataset.task.type!r} "
      "(expected 'classification')"
    )

  # With your schema.py, classification implies model.name == "timm"
  if cfg.model.name != "timm":
    raise ValueError("Classification requires model.name='timm'.")

  # schema.py guarantees arch exists when model.name == 'timm'
  arch = cfg.model.arch
  assert arch is not None  # for type-checkers; schema enforces this

  # schema cross-field validation ensures consistency if model.num_classes is set
  num_classes = cfg.model.num_classes or cfg.dataset.task.num_classes

  model = timm.create_model(
    cfg.model.arch, # type: ignore
    pretrained=False,  # IMPORTANT
    num_classes=num_classes,
    in_chans=cfg.model.in_channels,
  )

  ckpt = getattr(cfg.model, "init_ckpt", None)

  if ckpt:
    from safetensors.torch import load_file
    state = load_file(ckpt)
    head_keys = [
      "fc.weight",
      "fc.bias",
      "classifier.weight",
      "classifier.bias",
      "head.weight",
      "head.bias",
    ]

    filtered = {
        k: v for k, v in state.items()
        if k not in head_keys
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("[CKPT]", "missing:", missing[:5], "unexpected:", unexpected[:5])

  return model


def build_loss_cls(cfg: AppCfg) -> nn.Module:
  # CrossEntropyLoss expects class indices (0..C-1)
  if cfg.dataset.task.type != "classification":
    raise ValueError(
      f"build_loss_cls called with dataset.task.type={cfg.dataset.task.type!r} "
      "(expected 'classification')"
    )
  return nn.CrossEntropyLoss()


def build_optimizer_cls(cfg: AppCfg, model: Module) -> Optimizer:
  return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
