from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from ..torchscale.model.LongNet import make_longnet_from_name
from .pos_embed import get_2d_sincos_pos_embed


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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
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
        coords_ = torch.floor(coords / tile_size)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        all_layer_embed: bool = False,
    ) -> list[torch.Tensor]:
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


class LongNetMIL(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
        slide_encoder: LongNetViT,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder
        self.classifier = nn.Linear(slide_encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords are required for LongNetMIL.")
        feats = self.tile_encoder(x)
        slide_out = self.slide_encoder(feats.unsqueeze(0), coords.unsqueeze(0))[-1]
        return self.classifier(slide_out) # type: ignore
