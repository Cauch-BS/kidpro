from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from ...config.schema import AppCfg
from . import (
    FoundationBackbone,
    infer_feat_dim,
    load_state_dict_generic,
    resolve_weights_path,
)

FOUNDATION_NAME = "prov_gigapath"


class ProvGigaPathBackbone(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.tile_encoder = tile_encoder

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        return cast(torch.Tensor, self.tile_encoder(x))


def build(cfg: AppCfg) -> FoundationBackbone:
    import timm

    arch = cfg.model.arch
    if not arch:
        raise ValueError("model.arch is required for prov_gigapath foundation.")
    if arch == "prov_gigapath":
        arch = "vit_giant_patch14_dinov2"

    tile_encoder = timm.create_model(
        arch,
        pretrained=False,
        num_classes=0,
        global_pool="token",
        in_chans=cfg.model.in_channels,
    )

    ckpt = resolve_weights_path(cfg)
    if ckpt is not None:
        if ckpt.exists():
            load_state_dict_generic(tile_encoder, ckpt)
        elif ckpt.name == "model.pt":
            candidate = ckpt.with_name("pytorch_model.bin")
            if candidate.exists():
                load_state_dict_generic(tile_encoder, candidate)

    backbone = ProvGigaPathBackbone(
        tile_encoder=tile_encoder,
    )

    feat_dim = infer_feat_dim(tile_encoder, cfg)

    return FoundationBackbone(backbone=backbone, feat_dim=feat_dim)
