from __future__ import annotations

import logging
from typing import Any, Callable, Union, cast

import torch
import torch.nn as nn

from ..torchscale.model.LongNet import make_longnet_from_name
from .pos_embed import get_2d_sincos_pos_embed

log = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """Slide Patch Embedding."""

    def __init__(
        self,
        in_chans: int = 1536,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nn.Module] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


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
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        all_layer_embed: bool = False,
    ) -> list[torch.Tensor]:
        """
        Args:
            x: Tile embeddings of shape (batch, num_tiles, in_dim) or (num_tiles, in_dim)
            coords: Tile coordinates (ignored for simple pooling)
            all_layer_embed: Whether to return all layer embeddings (ignored, returns single embedding)

        Returns:
            List containing single pooled embedding of shape (batch, embed_dim)
        """
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


class LongNetViT(nn.Module):
    def __init__(
        self,
        in_chans: int = 1536,
        embed_dim: int = 256,
        depth: int = 12,
        slide_ngrids: int = 1000,
        tile_size: int = 256,
        max_wsi_size: int = 262144,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        global_pool: bool = False,
        dropout: float = 0.25,
        drop_path_rate: float = 0.1,
        input_norm: bool = True,
        input_dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        # Input conditioning: normalize and dropout before patch embedding
        self.input_norm = nn.LayerNorm(in_chans) if input_norm else nn.Identity()
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        log.info("[LongNetViT] input_norm=%s, input_dropout=%.2f", input_norm, input_dropout)

        self.patch_embed = PatchEmbed(in_chans, embed_dim, norm_layer=None)
        self.tile_size = tile_size
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, num_patches + 1, embed_dim),
            persistent=False,
        )

        self.encoder_name = f"LongNet_{depth}_layers_{embed_dim}_dim"
        if kwargs.get("mlp_ratio", 4.0) != 4.0:
            self.encoder_name += f"_mlp{kwargs.get('mlp_ratio')}"

        max_seq_len = (max_wsi_size // tile_size) ** 2
        segment_length = torch.linspace(
            torch.log2(torch.tensor(1024.0)),
            torch.log2(torch.tensor(float(max_seq_len))),
            steps=5,
        )
        segment_list = torch.pow(2.0, segment_length).to(torch.int).tolist()
        self.encoder = make_longnet_from_name(
            self.encoder_name,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            segment_length=str(segment_list),
        )
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool
        pos_embed = get_2d_sincos_pos_embed(
            int(self.pos_embed.shape[-1]), # type: ignore
            self.slide_ngrids,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0)) # type: ignore
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)
        nn.init.normal_(self.cls_token, std=0.02)

    def coords_to_pos(self, coords: torch.Tensor, tile_size: int = 256) -> torch.Tensor:
        """
        Convert pixel coordinates to position indices for position embedding lookup.

        Args:
            coords: Tile coordinates of shape (..., 2) where coords[..., 0] is x (col) and
                    coords[..., 1] is y (row) in pixel space.
            tile_size: Size of each tile in pixels.

        Returns:
            Position indices of shape (...) for looking up in pos_embed.
            Uses row-major ordering: pos = row * ngrids + col + 1 (offset by 1 for CLS token).
        """
        coords_ = torch.floor(coords / tile_size)

        # Bounds checking assertions
        x_coords = coords_[..., 0]
        y_coords = coords_[..., 1]
        if not ((x_coords >= 0).all() and (x_coords < self.slide_ngrids).all()):
            raise ValueError(
                f"x coordinates out of bounds: min={x_coords.min().item()}, "
                f"max={x_coords.max().item()}, expected [0, {self.slide_ngrids})"
            )
        if not ((y_coords >= 0).all() and (y_coords < self.slide_ngrids).all()):
            raise ValueError(
                f"y coordinates out of bounds: min={y_coords.min().item()}, "
                f"max={y_coords.max().item()}, expected [0, {self.slide_ngrids})"
            )

        # Row-major ordering: pos = y * ngrids + x (where y=row, x=col)
        pos = y_coords * self.slide_ngrids + x_coords
        pos = pos.long() + 1  # Offset by 1 for CLS token at position 0

        # Final bounds check
        max_pos = self.slide_ngrids ** 2
        if not ((pos >= 1).all() and (pos <= max_pos).all()):
            raise ValueError(
                f"Position indices out of bounds: min={pos.min().item()}, "
                f"max={pos.max().item()}, expected [1, {max_pos}]"
            )

        return pos

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        all_layer_embed: bool = False,
    ) -> list[torch.Tensor]:
        # Apply input conditioning
        x = self.input_norm(x)
        x = self.input_dropout(x)

        x = self.patch_embed(x)
        pos = self.coords_to_pos(coords, self.tile_size)
        x = x + self.pos_embed[:, pos, :].squeeze(0) # type: ignore

        cls_token = self.cls_token + self.pos_embed[:, :1, :] # type: ignore
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if all_layer_embed:
            out = self.encoder(
                src_tokens=None,
                token_embeddings=x,
                return_all_hiddens=all_layer_embed,
            )["encoder_states"]
        else:
            out = [self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]]

        outcomes: list[torch.Tensor] = []
        for o in out:
            if self.global_pool:
                pooled = o[:, 1:, :].mean(dim=1)
                outcomes.append(self.norm(pooled))
            else:
                outcomes.append(self.norm(o)[:, 0])
        return outcomes


SlideEncoder = Union[LongNetViT, SimpleAggregator]


class LongNetMIL(nn.Module):
    """
    MIL model combining tile encoder, slide aggregator, and classifier.

    Supports different slide encoders:
    - LongNetViT: Full LongNet encoder with positional embeddings
    - SimpleAggregator: Simple mean/max pooling baseline
    """

    def __init__(
        self,
        tile_encoder: nn.Module,
        slide_encoder: SlideEncoder,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder
        self.classifier = nn.Linear(slide_encoder.embed_dim, num_classes)

        # Log aggregator type
        if isinstance(slide_encoder, SimpleAggregator):
            log.info("[LongNetMIL] Using SimpleAggregator (pool_type=%s)", slide_encoder.pool_type)
        else:
            log.info("[LongNetMIL] Using LongNetViT encoder")

    def encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.tile_encoder(x))

    def encode_slide_embedding(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode tile features into a slide-level embedding.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates of shape (num_tiles, 2). Required for LongNetViT,
                    optional for SimpleAggregator.

        Returns:
            Slide embedding of shape (1, embed_dim)
        """
        if isinstance(self.slide_encoder, SimpleAggregator):
            # SimpleAggregator doesn't need coords
            slide_out = self.slide_encoder(feats.unsqueeze(0), coords)[-1]
        else:
            # LongNetViT requires coords
            if coords is None:
                raise ValueError("coords are required for LongNetViT slide encoder.")
            slide_out = self.slide_encoder(feats.unsqueeze(0), coords.unsqueeze(0))[-1]
        return cast(torch.Tensor, slide_out)

    def classify_slide_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(embedding))

    def encode_slide(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        slide_out = self.encode_slide_embedding(feats, coords)
        return self.classify_slide_embedding(slide_out)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.encode_tiles(x)
        return self.encode_slide(feats, coords)
