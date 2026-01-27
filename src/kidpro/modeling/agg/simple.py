from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...config.schema import AppCfg
    from . import SlideEncoderBackbone


AGGREGATOR_NAME = "simple"


class SimpleAggregator(nn.Module):
    """
    Simple baseline aggregator using mean/max pooling.
    Useful for debugging to compare against LongNet.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        pool_type: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.pool_type = pool_type

        self.input_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        all_layer_embed: bool = False,
        **kwargs: Any,  # Accept extra kwargs from PEFT wrapper (e.g., input_ids)
    ) -> list[torch.Tensor]:
        """
        Args:
            x: Tile embeddings of shape (batch, num_tiles, in_dim) or (num_tiles, in_dim)
            coords: Tile coordinates (accepted but ignored for simple pooling)
            all_layer_embed: Whether to return all layer embeddings (ignored, returns single embedding)
            **kwargs: Extra keyword arguments (ignored, for PEFT compatibility)

        Returns:
            List containing single pooled embedding of shape (batch, embed_dim)
        """
        if x is None:
            x = kwargs.get("x", None)
        _ = coords if coords is not None else kwargs.get("coords", None)
        if x is None:
            raise TypeError("SimpleAggregator.forward requires x (either as arg or kwarg).")
        # Normalize input
        x = self.input_norm(x)
        x = self.dropout(x)

        # Project to embed_dim
        x = self.proj(x)  # (..., embed_dim)

        # Pool over tiles
        if x.ndim == 2:
            # (num_tiles, embed_dim) -> (1, embed_dim)
            if self.pool_type == "mean":
                pooled = x.mean(dim=0, keepdim=True)
            elif self.pool_type == "max":
                pooled = x.max(dim=0, keepdim=True)[0]
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        else:
            # (batch, num_tiles, embed_dim) -> (batch, embed_dim)
            if self.pool_type == "mean":
                pooled = x.mean(dim=1)
            elif self.pool_type == "max":
                pooled = x.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return [self.norm(pooled)]


def build(cfg: "AppCfg") -> "SlideEncoderBackbone":
    """Build a SimpleAggregator from config."""
    from . import SlideEncoderBackbone

    in_chans = int(getattr(cfg.model, "foundation_dim", 1536))
    dim = cfg.model.longnet_dim
    pool_type = "mean" if cfg.model.aggregator_type == "mean_pool" else "max"
    dropout = cfg.model.longnet_input_dropout

    encoder = SimpleAggregator(
        in_dim=in_chans,
        embed_dim=dim,
        pool_type=pool_type,
        dropout=dropout,
    )
    return SlideEncoderBackbone(encoder=encoder, embed_dim=dim)
