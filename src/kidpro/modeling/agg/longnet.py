from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import torch
import torch.nn as nn

from ...torchscale.model.LongNet import make_longnet_from_name
from ...utils.pos_embed import get_2d_sincos_pos_embed
from .simple import SimpleAggregator

if TYPE_CHECKING:
    from ...config.schema import AppCfg

from ._types import MILTemplate, SlideEncoderBackbone

log = logging.getLogger(__name__)


AGGREGATOR_NAME = "longnet"


def _infer_stride_1d(vals_int: torch.Tensor) -> torch.Tensor:
    """
    Infer the most likely stride from 1D integer coordinates.

    Robust to non-zero origins and MPP rounding effects.
    """
    if vals_int.numel() < 2:
        return torch.tensor(1, device=vals_int.device, dtype=torch.long)

    uniq = torch.unique(vals_int)
    if uniq.numel() < 2:
        return torch.tensor(1, device=vals_int.device, dtype=torch.long)

    uniq, _ = torch.sort(uniq)
    diffs = uniq[1:] - uniq[:-1]
    diffs = diffs[diffs > 0]
    if diffs.numel() == 0:
        return torch.tensor(1, device=vals_int.device, dtype=torch.long)

    diff_vals, counts = torch.unique(diffs, return_counts=True)
    stride = diff_vals[counts.argmax()].to(torch.long)
    if stride.item() <= 0:
        return torch.tensor(1, device=vals_int.device, dtype=torch.long)
    return stride # type: ignore[no-any-return]


def _coords_pixel_to_grid(coords_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert level-0 pixel coords (x,y) into integer grid coords (col,row).

    Returns:
        coords_grid: (N,2) int64 tensor as (col,row)
        stride_x: scalar int64 tensor
        stride_y: scalar int64 tensor
    """
    if coords_xy.ndim != 2 or coords_xy.shape[-1] != 2:
        raise ValueError(f"Expected coords of shape (N,2), got {tuple(coords_xy.shape)}")

    coords_int = torch.round(coords_xy).to(torch.int64)
    x = coords_int[:, 0]
    y = coords_int[:, 1]
    stride_x = _infer_stride_1d(x)
    stride_y = _infer_stride_1d(y)

    col = torch.div(x, stride_x, rounding_mode="floor")
    row = torch.div(y, stride_y, rounding_mode="floor")
    return torch.stack([col, row], dim=-1), stride_x, stride_y


class _PatchEmbed(nn.Module):
    """Slide Patch Embedding (internal implementation)."""

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


class _LongNetViT(nn.Module):
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
        log.info("[_LongNetViT] input_norm=%s, input_dropout=%.2f", input_norm, input_dropout)

        self.patch_embed = _PatchEmbed(in_chans, embed_dim, norm_layer=None)
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
        end_log2 = torch.floor(torch.log2(torch.tensor(float(max_seq_len)))).to(torch.long)
        segment_length = torch.linspace(
            torch.log2(torch.tensor(1024.0)),
            end_log2.to(torch.float32),
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
        Convert grid coordinates (col,row) to position indices for pos-embed lookup.

        Args:
            coords: Tile coordinates of shape (..., 2) as integer grid indices (col,row).
                These are derived from pixel-space coordinates (x,y) at call sites.
            tile_size: Unused (kept for backwards compatibility).

        Returns:
            Position indices of shape (...) for looking up in pos_embed.
            IMPORTANT: This follows the *reference slide encoder* convention:
              pos = col * ngrids + row + 1
            (offset by 1 for CLS token at position 0).
        """
        coords_ = coords.to(torch.long)

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

        # Reference convention: pos = x * ngrids + y
        # NOTE: This is not the usual row-major (y * ngrids + x); it's intentional
        # to match `reference-slide-encode.py`.
        pos = x_coords * self.slide_ngrids + y_coords
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
        x: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        all_layer_embed: bool = False,
        **kwargs: Any,  # Accept extra kwargs from PEFT wrapper (e.g., input_ids)
    ) -> list[torch.Tensor]:
        # Apply input conditioning
        if x is None:
            x = kwargs.get("x", None)
        if coords is None:
            coords = kwargs.get("coords", None)
        if x is None or coords is None:
            raise TypeError("_LongNetViT.forward requires x and coords (either as args or kwargs).")
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


SlideEncoder = Union[_LongNetViT, SimpleAggregator]


class LongNetMIL(MILTemplate):
    """
    MIL model combining tile encoder, slide aggregator, and classifier.

    Supports different slide encoders:
    - _LongNetViT: Full LongNet encoder with positional embeddings
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
        self._coord_log_emitted = False
        self._coord_clamp_warned = False

        # Log aggregator type
        if isinstance(slide_encoder, SimpleAggregator):
            log.info("[LongNetMIL] Using SimpleAggregator (pool_type=%s)", slide_encoder.pool_type)
        else:
            log.info("[LongNetMIL] Using _LongNetViT encoder")

    def encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.tile_encoder(x))

    def encode_slide_embedding(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode tile features into a slide-level embedding.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates of shape (num_tiles, 2). Required for _LongNetViT,
                    optional for SimpleAggregator.

        Returns:
            Slide embedding of shape (1, embed_dim)
        """
        if isinstance(self.slide_encoder, SimpleAggregator):
            # SimpleAggregator doesn't need coords
            slide_out = self.slide_encoder(x = feats.unsqueeze(0))[-1]
        else:
            # _LongNetViT requires coords
            if coords is None:
                raise ValueError("coords are required for _LongNetViT slide encoder.")

            # Convert pixel-space (x,y) coords to grid-space (col,row) indices.
            coords_grid, stride_x, stride_y = _coords_pixel_to_grid(coords)

            ngrids = int(getattr(self.slide_encoder, "slide_ngrids", 0))
            if ngrids <= 0:
                raise ValueError("slide_encoder.slide_ngrids must be > 0 for _LongNetViT.")

            col = coords_grid[:, 0]
            row = coords_grid[:, 1]
            col_min, col_max = int(col.min().item()), int(col.max().item())
            row_min, row_max = int(row.min().item()), int(row.max().item())

            if not self._coord_log_emitted:
                log.debug("[LongNetMIL] coord stride inferred")
                self._coord_log_emitted = True

            oob = (col_min < 0) or (row_min < 0) or (col_max >= ngrids) or (row_max >= ngrids)
            if oob:
                if not self._coord_clamp_warned:
                    log.debug("[LongNetMIL] Grid coords out of bounds")
                    self._coord_clamp_warned = True
                coords_grid = torch.stack(
                    [
                        col.clamp(0, ngrids - 1),
                        row.clamp(0, ngrids - 1),
                    ],
                    dim=-1,
                )

            # Keep dtype consistent with existing forward signature; slide encoder will cast to long.
            coords_grid = coords_grid.to(device=coords.device, dtype=coords.dtype)
            slide_out = self.slide_encoder(x = feats.unsqueeze(0), coords = coords_grid.unsqueeze(0))[-1]
        return cast(torch.Tensor, slide_out)

    def classify_slide_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(embedding))

    # encode_slide and forward are inherited from MILTemplate


def _resolve_longnet_weights_path(cfg: "AppCfg") -> Path:
    weights = cfg.model.longnet_weights
    if weights is None:
        raise ValueError("model.longnet_weights is required to preload LongNet.")
    if weights.source == "local":
        return Path(weights.local_path)  # type: ignore[arg-type]
    if weights.source == "hf_cache":
        return Path(weights.hf_cache_path)  # type: ignore[arg-type]
    raise ValueError(f"Unknown longnet_weights.source={weights.source!r}")


def build(cfg: "AppCfg") -> SlideEncoderBackbone:
    """Build a _LongNetViT slide encoder from config."""
    from ...utils.model_io import load_state_dict_generic
    from ..lora import apply_lora

    in_chans = int(getattr(cfg.model, "foundation_dim", 1536))
    dim = cfg.model.longnet_dim

    encoder = _LongNetViT(
        in_chans=in_chans,
        embed_dim=dim,
        depth=cfg.model.longnet_depth,
        slide_ngrids=cfg.model.longnet_slide_ngrids,
        tile_size=cfg.dataset.data.patch_size,
        max_wsi_size=cfg.model.longnet_max_wsi_size,
        global_pool=False,
        dropout=cfg.model.longnet_dropout,
        input_norm=cfg.model.longnet_input_norm,
        input_dropout=cfg.model.longnet_input_dropout,
    )

    if cfg.model.longnet_pretrained:
        ckpt_path = _resolve_longnet_weights_path(cfg)
        log.info("Loading LongNet weights from %s", ckpt_path)
        encoder = load_state_dict_generic(encoder, ckpt_path)  # type: ignore[assignment]

    lora_cfg = cfg.model.lora
    apply_to = set(lora_cfg.apply_to)
    if lora_cfg.enabled and "longnet" in apply_to:
        encoder = apply_lora(cfg, encoder, freeze_base=True)

    return SlideEncoderBackbone(encoder=encoder, embed_dim=dim)


def build_mil(cfg: "AppCfg", tile_encoder: nn.Module, num_classes: int) -> MILTemplate:
    """
    Build a complete LongNetMIL model from config.

    This returns a complete MILTemplate, not just a slide encoder.
    """
    slide_encoder_result = build(cfg)
    slide_encoder = cast(SlideEncoder, slide_encoder_result.encoder)

    model = LongNetMIL(
        tile_encoder=tile_encoder,
        slide_encoder=slide_encoder,
        num_classes=num_classes,
    )

    return model
