from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class SlideEncoderBackbone:
    """Public return type for slide encoder builders."""
    encoder: nn.Module
    embed_dim: int
