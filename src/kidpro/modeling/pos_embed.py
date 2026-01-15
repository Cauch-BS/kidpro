from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def get_1d_sincos_pos_embed(
    embed_dim: int,
    pos: NDArray[Any],
) -> NDArray[np.float32]:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos position embedding.")
    omega: NDArray[np.float32] = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = cast(NDArray[np.float32], np.einsum("m,d->md", pos, omega))
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)
    return cast(NDArray[np.float32], emb)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> NDArray[np.float32]:
    grid_h: NDArray[np.float32] = np.arange(grid_size, dtype=np.float32)
    grid_w: NDArray[np.float32] = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(
        np.meshgrid(grid_w, grid_h, indexing="xy"),
        axis=0,
    ).astype(np.float32)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, pos_h)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, pos_w)
    emb = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token:
        cls = np.zeros([1, embed_dim], dtype=np.float32)
        emb = np.concatenate([cls, emb], axis=0)
    return cast(NDArray[np.float32], emb.astype(np.float32))
