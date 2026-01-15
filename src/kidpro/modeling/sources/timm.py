from __future__ import annotations

from ...config.schema import AppCfg
from . import (
    FoundationBackbone,
    infer_feat_dim,
    load_state_dict_generic,
    resolve_weights_path,
)

FOUNDATION_NAME = "timm"

def build(cfg: AppCfg) -> FoundationBackbone:
    import timm

    arch = cfg.model.arch
    if not arch:
        raise ValueError("model.arch is required for timm foundation (e.g., resnet50).")

    backbone = timm.create_model(
        arch,
        pretrained=False,  # IMPORTANT: no download
        num_classes=0,     # backbone only
        global_pool="avg",
        in_chans=cfg.model.in_channels,
    )

    ckpt = resolve_weights_path(cfg)
    if ckpt is not None:
        load_state_dict_generic(backbone, ckpt)

    feat_dim = infer_feat_dim(backbone, cfg)
    return FoundationBackbone(backbone=backbone, feat_dim=feat_dim)
