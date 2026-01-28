#!/usr/bin/env python3
"""
Standalone WSI inference script for test submission.

Pipeline:
1) Read .svs WSI with wsidata/lazyslide and extract tiles.
2) Encode tiles with a tile encoder model loaded from a path.
3) Aggregate tile features with a slide encoder model loaded from a path.
4) Write CSV with columns: ID, Predicted_Label, Predicted_Prob.
"""

from __future__ import annotations

import argparse
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")

warnings.filterwarnings(
    "ignore",
    message="The legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Error fetching version info.*",
    category=UserWarning,
)

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, roc_auc_score

try:
    from wsidata import WSIData, open_wsi  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "wsidata is required. Install with: pip install wsidata"
    ) from exc

try:
    import lazyslide as zs  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "lazyslide is required. Install with: pip install lazyslide"
    ) from exc


try:
    import timm  # type: ignore
except Exception:
    timm = None

# Try to import LongNet from long_net package
try:
    from long_net import DilatedAttention  # type: ignore
    from long_net.model import LongNetTransformer  # type: ignore
    HAS_LONGNET = True
except Exception:
    HAS_LONGNET = False
    LongNetTransformer = None
    DilatedAttention = None


_OPENSLIDE_QUIETED = False


def _quiet_openslide_logs() -> None:
    global _OPENSLIDE_QUIETED
    if _OPENSLIDE_QUIETED:
        return
    try:
        from openslide import lowlevel as openslide_lowlevel
    except Exception:
        _OPENSLIDE_QUIETED = True
        return

    def _null_handler(*_args: object, **_kwargs: object) -> None:
        return

    try:
        openslide_lowlevel.set_error_handler(_null_handler)
        openslide_lowlevel.set_warning_handler(_null_handler)
    except Exception:
        pass
    _OPENSLIDE_QUIETED = True


@dataclass
class TileConfig:
    tile_size: int = 512
    level: int = 0
    max_tiles: int = 2048
    tiles_key: str = "tiles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WSI inference for test submission (standalone)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wsi", help="Path to input .svs file (single-slide mode).")
    mode.add_argument("--csv", help="Path to input CSV (batch mode).")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--wsi-dir",
        default=None,
        help="Optional directory to resolve WSI paths when CSV lacks image paths.",
    )
    parser.add_argument(
        "--wsi-ext",
        default=".svs",
        help="WSI extension when resolving from --wsi-dir (default: .svs).",
    )
    parser.add_argument(
        "--csv-wsi-col",
        default=None,
        help="CSV column containing WSI path (auto-detected if not provided).",
    )
    parser.add_argument(
        "--csv-slide-col",
        default=None,
        help="CSV column containing slide identifier (auto-detected if not provided).",
    )
    parser.add_argument(
        "--csv-gt-col",
        default="GT",
        help="CSV column containing ground-truth labels (default: GT).",
    )
    parser.add_argument(
        "--csv-split-col",
        default=None,
        help="CSV column indicating split (train/val/test). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--csv-test-value",
        default="test",
        help="Value in split column to export submission rows (default: test).",
    )
    parser.add_argument(
        "--tile-encoder-path",
        required=True,
        help="Path to tile encoder checkpoint (TorchScript or nn.Module).",
    )
    parser.add_argument(
        "--slide-encoder-path",
        required=True,
        help="Path to slide encoder checkpoint (TorchScript or nn.Module).",
    )
    parser.add_argument(
        "--tile-encoder-arch",
        default=None,
        help="Optional timm model name if tile encoder is a state_dict.",
    )
    parser.add_argument(
        "--slide-encoder-arch",
        default=None,
        choices=["attention_mil", "mean_pool"],
        help="Optional slide encoder arch if slide encoder is a state_dict.",
    )
    parser.add_argument(
        "--longnet-dim",
        type=int,
        default=768,
        help="LongNet embedding dimension (must match training config).",
    )
    parser.add_argument(
        "--longnet-depth",
        type=int,
        default=12,
        help="LongNet depth/number of layers (must match training config).",
    )
    parser.add_argument(
        "--longnet-slide-ngrids",
        type=int,
        default=1000,
        help="LongNet slide grid size (must match training config).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--input-size",
        type=int,
        default=0,
        help="Optional model input size for center-crop (0 disables).",
    )
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def default_transforms(tile_size: int, input_size: int) -> A.Compose:
    crop_size = input_size if input_size > 0 and input_size != tile_size else None
    return A.Compose(
        [
            A.Resize(tile_size, tile_size, interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(crop_size, crop_size) if crop_size else A.NoOp(),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def open_wsidata(slide_path: Path) -> "WSIData":
    slide_path = Path(slide_path)
    if not slide_path.exists():
        raise FileNotFoundError(f"WSI file not found: {slide_path}")
    if not slide_path.is_file():
        raise FileNotFoundError(f"WSI path is not a file: {slide_path}")
    _quiet_openslide_logs()
    return open_wsi(slide_path)


@contextmanager
def _silence_stderr() -> Iterator[None]:
    if os.environ.get("KIDPRO_SUPPRESS_TIFF_ERRORS", "1") == "0":
        yield
        return
    try:
        old_stderr = os.dup(2)
    except Exception:
        yield
        return
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        try:
            os.dup2(old_stderr, 2)
        finally:
            os.close(old_stderr)


def tile_image_to_array(tile: object) -> np.ndarray:
    with _silence_stderr():
        image = getattr(tile, "image", tile)
        arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def extract_tile_xy(tile: object) -> Tuple[int, int]:
    if hasattr(tile, "x") and hasattr(tile, "y"):
        return int(getattr(tile, "x")), int(getattr(tile, "y"))
    meta = getattr(tile, "meta", None)
    if isinstance(meta, dict) and "x" in meta and "y" in meta:
        return int(meta["x"]), int(meta["y"])
    raise RuntimeError("Unable to extract tile coordinates from wsidata tile metadata.")


def resolve_mpp(wsi: "WSIData", level: int) -> float:
    default_mpp = 0.5
    props = getattr(wsi, "properties", None)
    mpp = getattr(props, "mpp", None) if props is not None else None
    if mpp is None:
        return default_mpp
    try:
        mpp = float(mpp)
        downsample = getattr(props, "level_downsample", None)
        if downsample is not None and level < len(downsample):
            mpp *= float(downsample[level])
        return max(mpp, default_mpp) # type: ignore
    except Exception:
        return default_mpp


def generate_tiles(
    wsi: "WSIData",
    cfg: TileConfig,
) -> Iterable[Tuple[np.ndarray, Tuple[int, int]]]:
    zs.pp.find_tissues(wsi)
    mpp = resolve_mpp(wsi, cfg.level)
    zs.pp.tile_tissues(
        wsi,
        key_added=cfg.tiles_key,
        tile_px=cfg.tile_size,
        mpp=mpp,
        ops_level=cfg.level,
        return_tiles=False,
    )

    count = 0
    for tile in wsi.iter.tile_images(cfg.tiles_key):
        if count >= cfg.max_tiles:
            return
        arr = tile_image_to_array(tile)
        x, y = extract_tile_xy(tile)
        yield arr, (x, y)
        count += 1


def normalize_state_dict(
    state: dict[str, torch.Tensor],
    strip_slide_encoder: bool = True,
) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "tile_encoder.",
    )
    if strip_slide_encoder:
        prefixes = prefixes + ("slide_encoder.",) # type: ignore
    normalized: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out_k = k
        for prefix in prefixes:
            if out_k.startswith(prefix):
                out_k = out_k[len(prefix) :]
        normalized[out_k] = v
    return normalized


def load_model_from_path(path: Path, device: str) -> Tuple[object, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model, "torchscript"
    except Exception:
        pass

    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj, "module"
    if isinstance(obj, dict):
        return obj, "state_dict"
    raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")


def get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos position embedding.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h, indexing="xy"), axis=0).astype(np.float32)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, pos_h)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, pos_w)
    emb = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token:
        cls = np.zeros([1, embed_dim], dtype=np.float32)
        emb = np.concatenate([cls, emb], axis=0)
    return emb.astype(np.float32)


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
    return stride


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


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_chans: int = 1536,
        embed_dim: int = 256,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


class LongNetViTLite(nn.Module):
    """
    LongNet encoder for inference using long_net package.

    Uses DilatedAttention from long_net package when available, falls back to TransformerEncoder otherwise.
    """
    def __init__(
        self,
        in_chans: int,
        embed_dim: int = 256,
        depth: int = 12,
        slide_ngrids: int = 1000,
        tile_size: int = 256,
        max_wsi_size: int = 262144,
        global_pool: bool = False,
        dropout: float = 0.25,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_chans, embed_dim, bias=True)
        self.tile_size = tile_size
        self.slide_ngrids = slide_ngrids

        num_patches = slide_ngrids**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, num_patches + 1, embed_dim),
            persistent=False,
        )

        # Use DilatedAttention from long_net package if available
        if HAS_LONGNET and DilatedAttention is not None:
            self.use_longnet = True
            # Calculate segment size based on max sequence length
            max_seq_len = (max_wsi_size // tile_size) ** 2
            segment_size = max(64, min(1024, max_seq_len // 4))  # Reasonable segment size

            # Build encoder layers using DilatedAttention
            self.encoder_layers = nn.ModuleList()
            heads = 8 if embed_dim % 8 == 0 else 4 if embed_dim % 4 == 0 else 1

            for i in range(depth):
                # Vary dilation rate across layers
                dilation_rate = 2 ** (i % 4)  # Cycles through 1, 2, 4, 8
                attn = DilatedAttention(
                    dim=embed_dim,
                    heads=heads,
                    dilation_rate=dilation_rate,
                    segment_size=segment_size,
                    qk_norm=True,
                )
                # Add feedforward and normalization
                ff = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                )
                self.encoder_layers.append(nn.ModuleDict({
                    'attn': attn,
                    'ff': ff,
                    'norm1': nn.LayerNorm(embed_dim),
                    'norm2': nn.LayerNorm(embed_dim),
                }))
        else:
            self.use_longnet = False
            nhead = 8 if embed_dim % 8 == 0 else 4 if embed_dim % 4 == 0 else 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        self.global_pool = global_pool
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)
        nn.init.normal_(self.cls_token, std=0.02)

    def coords_to_pos(self, coords: torch.Tensor, tile_size: int = 256) -> torch.Tensor:
        """
        Convert grid coordinates (col,row) to position indices for pos-embed lookup.

        Args:
            coords: Tile coordinates of shape (..., 2) as pixel-space coordinates (x,y).
                    These are converted to grid indices (col,row) internally.
            tile_size: Unused (kept for backwards compatibility).

        Returns:
            Position indices of shape (...) for looking up in pos_embed.
            IMPORTANT: This follows the *reference slide encoder* convention:
              pos = col * ngrids + row + 1
            (offset by 1 for CLS token at position 0).
        """
        # Convert pixel-space coords to grid coords (col, row)
        coords_grid, _, _ = _coords_pixel_to_grid(coords)
        coords_ = coords_grid.to(torch.long)

        # Bounds checking
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

        # Reference convention: pos = col * ngrids + row
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
        x: torch.Tensor,
        coords: torch.Tensor,
        all_layer_embed: bool = False,
    ) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        pos = self.coords_to_pos(coords, self.tile_size)
        pos_embed = self.pos_embed[0, pos, :]
        x = x + pos_embed

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.use_longnet:
            # Use DilatedAttention layers
            all_hiddens = [x] if all_layer_embed else []
            for layer in self.encoder_layers:
                # Self-attention with residual
                residual = x
                x = layer['norm1'](x)
                x = layer['attn'](x) + residual
                # Feedforward with residual
                residual = x
                x = layer['norm2'](x)
                x = layer['ff'](x) + residual
                if all_layer_embed:
                    all_hiddens.append(x)
            out = all_hiddens if all_layer_embed else [x]
        else:
            # Fallback to TransformerEncoder
            out = self.encoder(x)
            out = [out]

        outcomes: list[torch.Tensor] = []
        for o in out:
            if self.global_pool:
                pooled = o[:, 1:, :].mean(dim=1)
                outcomes.append(self.norm(pooled))
            else:
                outcomes.append(self.norm(o)[:, 0])
        return outcomes


class AttentionMILHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 256,
        depth: int = 12,
        slide_ngrids: int = 1000,
        tile_size: int = 256,
        max_wsi_size: int = 262144,
        dropout: float = 0.25,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.slide_encoder = LongNetViTLite(
            in_chans=in_dim,
            embed_dim=embed_dim,
            depth=depth,
            slide_ngrids=slide_ngrids,
            tile_size=tile_size,
            max_wsi_size=max_wsi_size,
            global_pool=False,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.classifier = nn.Linear(self.slide_encoder.embed_dim, 1)

    def forward(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords are required for LongNetMIL.")
        slide_out = self.slide_encoder(feats.unsqueeze(0), coords.unsqueeze(0))[-1]
        return self.classifier(slide_out)


class MeanPoolHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 256,
        depth: int = 12,
        slide_ngrids: int = 1000,
        tile_size: int = 256,
        max_wsi_size: int = 262144,
        dropout: float = 0.25,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.slide_encoder = LongNetViTLite(
            in_chans=in_dim,
            embed_dim=embed_dim,
            depth=depth,
            slide_ngrids=slide_ngrids,
            tile_size=tile_size,
            max_wsi_size=max_wsi_size,
            global_pool=True,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.classifier = nn.Linear(self.slide_encoder.embed_dim, 1)

    def forward(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords are required for LongNetMIL.")
        slide_out = self.slide_encoder(feats.unsqueeze(0), coords.unsqueeze(0))[-1]
        return self.classifier(slide_out)


def build_tile_encoder(
    payload: object, arch: Optional[str], device: str
) -> nn.Module:
    if isinstance(payload, nn.Module):
        model = payload
        model.to(device)
        model.eval()
        return model

    if isinstance(payload, dict):
        if arch is None:
            raise ValueError(
                "Tile encoder checkpoint is a state_dict. Provide --tile-encoder-arch."
            )
        if timm is None:
            raise RuntimeError("timm is required for --tile-encoder-arch.")
        model = timm.create_model(arch, pretrained=False, num_classes=0)
        state = normalize_state_dict(payload)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model

    raise RuntimeError(f"Unsupported tile encoder payload: {type(payload)}")


def build_slide_encoder(
    payload: object,
    arch: Optional[str],
    in_dim: int,
    device: str,
    tile_size: int,
    embed_dim: int = 768,
    depth: int = 12,
    slide_ngrids: int = 1000,
    max_wsi_size: int = 262144,
    dropout: float = 0.25,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    if isinstance(payload, nn.Module):
        model = payload
        model.to(device)
        model.eval()
        return model

    if isinstance(payload, dict):
        if arch is None:
            raise ValueError(
                "Slide encoder checkpoint is a state_dict. Provide --slide-encoder-arch."
            )
        if arch == "attention_mil":
            model = AttentionMILHead(
                in_dim=in_dim,
                embed_dim=embed_dim,
                depth=depth,
                slide_ngrids=slide_ngrids,
                tile_size=tile_size,
                max_wsi_size=max_wsi_size,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
            )
        else:
            model = MeanPoolHead(
                in_dim=in_dim,
                embed_dim=embed_dim,
                depth=depth,
                slide_ngrids=slide_ngrids,
                tile_size=tile_size,
                max_wsi_size=max_wsi_size,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
            )
        state = normalize_state_dict(payload, strip_slide_encoder=False)
        if not any(k.startswith("slide_encoder.") for k in state):
            state = {
                (f"slide_encoder.{k}" if not k.startswith("classifier.") else k): v
                for k, v in state.items()
            }
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model

    raise RuntimeError(f"Unsupported slide encoder payload: {type(payload)}")


def extract_features(
    tile_encoder: nn.Module,
    tile_tensor: torch.Tensor,
) -> torch.Tensor:
    forward_features = getattr(tile_encoder, "forward_features", None)
    if callable(forward_features):
        feats = forward_features(tile_tensor)
    else:
        feats = tile_encoder(tile_tensor)

    if isinstance(feats, (list, tuple)):
        feats = feats[-1]

    if feats.dim() == 4:
        feats = feats.mean(dim=(-2, -1))
    elif feats.dim() == 3:
        feats = feats[:, 0, :] if feats.shape[1] > 1 else feats.mean(dim=1)
    return feats


def run_slide_encoder(
    slide_encoder: nn.Module,
    feats: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    try:
        return slide_encoder(feats, coords)
    except Exception:
        try:
            return slide_encoder(feats)
        except Exception:
            return slide_encoder(feats.unsqueeze(0), coords.unsqueeze(0))


def logits_to_prob(logits: torch.Tensor) -> float:
    if logits.dim() == 0:
        return float(torch.sigmoid(logits))
    if logits.dim() == 1:
        if logits.numel() == 1:
            return float(torch.sigmoid(logits[0]))
        probs = torch.softmax(logits, dim=0)
        return float(probs[1] if probs.numel() > 1 else probs[0])
    if logits.dim() == 2:
        if logits.shape[1] == 1:
            return float(torch.sigmoid(logits[0, 0]))
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1])
    return float(torch.sigmoid(logits.view(-1)[0]))


def predict_probability(
    *,
    wsi_path: Path,
    tile_encoder: nn.Module,
    slide_encoder: nn.Module,
    tile_cfg: TileConfig,
    batch_size: int,
    input_size: int,
    device: str,
    longnet_dim: int,
    longnet_depth: int,
    longnet_slide_ngrids: int,
) -> float:
    """
    Predict P(class=1) for a single slide.

    Note: This script is meant for binary submission. For multi-class, extend
    logits_to_prob() and output per-class probabilities.
    """
    transform = default_transforms(tile_cfg.tile_size, input_size)

    tiles_features: List[torch.Tensor] = []
    tiles_coords: List[Tuple[int, int]] = []
    batch_imgs: List[torch.Tensor] = []
    batch_coords: List[Tuple[int, int]] = []

    wsi = open_wsidata(wsi_path)
    try:
        with torch.no_grad():
            for img, (x, y) in generate_tiles(wsi, tile_cfg):
                batch_imgs.append(transform(image=img)["image"])
                batch_coords.append((x, y))
                if len(batch_imgs) >= batch_size:
                    batch = torch.stack(batch_imgs, dim=0).to(device)
                    feats = extract_features(tile_encoder, batch)
                    tiles_features.append(feats.cpu())
                    tiles_coords.extend(batch_coords)
                    batch_imgs = []
                    batch_coords = []

            if batch_imgs:
                batch = torch.stack(batch_imgs, dim=0).to(device)
                feats = extract_features(tile_encoder, batch)
                tiles_features.append(feats.cpu())
                tiles_coords.extend(batch_coords)
    finally:
        if hasattr(wsi, "close"):
            wsi.close()

    if not tiles_features:
        return 0.0

    feats_all = torch.cat(tiles_features, dim=0)
    coords = torch.tensor(tiles_coords, dtype=torch.float32)
    feats_all = feats_all.to(device)
    coords = coords.to(device)
    with torch.no_grad():
        logits = run_slide_encoder(slide_encoder, feats_all, coords)
    return logits_to_prob(logits)


def _infer_csv_columns(df: pd.DataFrame, explicit: dict[str, Optional[str]]) -> tuple[Optional[str], str, Optional[str], str]:
    """
    Infer (wsi_col, slide_col, split_col, gt_col) with reasonable defaults.
    """
    gt_col = explicit.get("gt_col") or "GT"

    wsi_candidates = [explicit.get("wsi_col")] if explicit.get("wsi_col") else []
    wsi_candidates += ["wsi_path", "wsi", "image_path", "image"]
    wsi_col = next((c for c in wsi_candidates if c in df.columns), "")

    # Slide identifier: prefer SlideName (used by kidpro), otherwise ID.
    slide_candidates = [explicit.get("slide_col")] if explicit.get("slide_col") else []
    slide_candidates += ["SlideName", "ID", "id", "slide_id"]
    slide_col = next((c for c in slide_candidates if c in df.columns), "")

    # Split: prefer 'split'. If absent but ID looks like split labels, allow that.
    split_col = explicit.get("split_col")
    if split_col is None:
        if "split" in df.columns:
            split_col = "split"
        elif "ID" in df.columns:
            vals = set(map(str, df["ID"].dropna().unique().tolist()))
            if vals and vals.issubset({"train", "val", "test"}):
                split_col = "ID"

    if not slide_col:
        raise ValueError(
            "Could not infer slide id column. Provide --csv-slide-col "
            "or add a 'SlideName' or 'ID' column."
        )

    if gt_col and gt_col not in df.columns:
        # allow GT to be absent; metrics will be skipped.
        gt_col = gt_col

    return wsi_col, slide_col, split_col, gt_col


def run_inference_csv(
    *,
    csv_path: Path,
    output_path: Path,
    tile_encoder_path: Path,
    slide_encoder_path: Path,
    tile_cfg: TileConfig,
    batch_size: int,
    input_size: int,
    threshold: float,
    device: str,
    tile_encoder_arch: Optional[str],
    slide_encoder_arch: Optional[str],
    longnet_dim: int,
    longnet_depth: int,
    longnet_slide_ngrids: int,
    wsi_dir: Optional[Path],
    wsi_ext: str,
    csv_wsi_col: Optional[str],
    csv_slide_col: Optional[str],
    csv_gt_col: str,
    csv_split_col: Optional[str],
    csv_test_value: str,
) -> None:
    df = pd.read_csv(csv_path)

    wsi_col, slide_col, split_col, gt_col = _infer_csv_columns(
        df,
        explicit={
            "wsi_col": csv_wsi_col,
            "slide_col": csv_slide_col,
            "split_col": csv_split_col,
            "gt_col": csv_gt_col,
        },
    )

    # Load tile encoder once.
    tile_payload, _ = load_model_from_path(tile_encoder_path, device=device)
    tile_encoder = build_tile_encoder(tile_payload, tile_encoder_arch, device=device)

    # Load slide encoder payload once; build module if needed.
    slide_payload, _ = load_model_from_path(slide_encoder_path, device=device)
    if isinstance(slide_payload, nn.Module):
        slide_encoder = slide_payload
        slide_encoder.to(device)
        slide_encoder.eval()
    else:
        # For state_dict slide encoder, we must build it with the feature dim.
        # We infer in_dim from the tile encoder by running one tiny forward.
        dummy = torch.zeros(1, 3, tile_cfg.tile_size, tile_cfg.tile_size, device=device)
        with torch.no_grad():
            dummy_feats = extract_features(tile_encoder, dummy).detach()
        in_dim = int(dummy_feats.shape[-1])
        slide_encoder = build_slide_encoder(
            slide_payload,
            slide_encoder_arch,
            in_dim=in_dim,
            device=device,
            tile_size=tile_cfg.tile_size,
            embed_dim=longnet_dim,
            depth=longnet_depth,
            slide_ngrids=longnet_slide_ngrids,
        )

    preds: list[int] = []
    probs: list[float] = []
    slide_ids: list[str] = []
    gt_vals: list[Optional[int]] = []
    split_vals: list[Optional[str]] = []

    # Normalize extension
    if not wsi_ext.startswith("."):
        wsi_ext = f".{wsi_ext}"

    for _, row in df.iterrows():
        slide_id = str(row[slide_col])

        # Resolve WSI path
        if wsi_col:
            wsi_path = Path(str(row[wsi_col]))
        else:
            if wsi_dir is None:
                raise ValueError(
                    "CSV does not include a WSI path column (e.g. image/image_path/wsi_path). "
                    "Provide --wsi-dir/--wsi-ext to resolve paths from slide id."
                )
            wsi_path = Path(wsi_dir) / f"{slide_id}{wsi_ext}"

        prob = predict_probability(
            wsi_path=wsi_path,
            tile_encoder=tile_encoder,
            slide_encoder=slide_encoder,
            tile_cfg=tile_cfg,
            batch_size=batch_size,
            input_size=input_size,
            device=device,
            longnet_dim=longnet_dim,
            longnet_depth=longnet_depth,
            longnet_slide_ngrids=longnet_slide_ngrids,
        )
        pred = int(prob >= threshold)

        slide_ids.append(slide_id)
        probs.append(float(prob))
        preds.append(int(pred))

        if gt_col in df.columns and pd.notna(row.get(gt_col)):
            gt_vals.append(int(row[gt_col]))
        else:
            gt_vals.append(None)

        if split_col is not None and split_col in df.columns and pd.notna(row.get(split_col)):
            split_vals.append(str(row[split_col]))
        else:
            split_vals.append(None)

    # Metrics (rows with GT present)
    y_true = [g for g in gt_vals if g is not None]
    y_pred = [p for p, g in zip(preds, gt_vals) if g is not None]
    y_score = [s for s, g in zip(probs, gt_vals) if g is not None]

    if y_true:
        macro = f1_score(y_true, y_pred, average="macro")
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float("nan")
        print(f"macro_f1: {macro:.6f}")
        print(f"ROC_AUC: {auc:.6f}" if auc == auc else "ROC_AUC: nan")
    else:
        print("macro_f1: n/a (no GT rows)")
        print("ROC_AUC: n/a (no GT rows)")

    # Submission CSV for test rows
    out_df = pd.DataFrame(
        {"ID": slide_ids, "Predicted_Label": preds, "Predicted_Prob": probs}
    )

    # Determine which rows are "test"
    test_mask = None
    if split_col is not None and split_col in df.columns:
        test_mask = df[split_col].astype(str) == str(csv_test_value)
    else:
        # Fallback: if there is an 'ID' column that contains split labels and SlideName exists,
        # use that as split; otherwise write all rows.
        test_mask = pd.Series([True] * len(df))

    out_df = out_df.loc[test_mask.values].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)


def run_inference(
    wsi_path: Path,
    output_path: Path,
    tile_encoder_path: Path,
    slide_encoder_path: Path,
    tile_cfg: TileConfig,
    batch_size: int,
    input_size: int,
    threshold: float,
    device: str,
    tile_encoder_arch: Optional[str],
    slide_encoder_arch: Optional[str],
    longnet_dim: int = 768,
    longnet_depth: int = 12,
    longnet_slide_ngrids: int = 1000,
) -> None:
    transform = default_transforms(tile_cfg.tile_size, input_size)

    tile_payload, _ = load_model_from_path(tile_encoder_path, device=device)
    tile_encoder = build_tile_encoder(tile_payload, tile_encoder_arch, device=device)

    tiles_features: List[torch.Tensor] = []
    tiles_coords: List[Tuple[int, int]] = []
    batch_imgs: List[torch.Tensor] = []
    batch_coords: List[Tuple[int, int]] = []

    wsi = open_wsidata(wsi_path)
    try:
        with torch.no_grad():
            for img, (x, y) in generate_tiles(wsi, tile_cfg):
                batch_imgs.append(transform(image=img)["image"])
                batch_coords.append((x, y))
                if len(batch_imgs) >= batch_size:
                    batch = torch.stack(batch_imgs, dim=0).to(device)
                    feats = extract_features(tile_encoder, batch)
                    tiles_features.append(feats.cpu())
                    tiles_coords.extend(batch_coords)
                    batch_imgs = []
                    batch_coords = []

            if batch_imgs:
                batch = torch.stack(batch_imgs, dim=0).to(device)
                feats = extract_features(tile_encoder, batch)
                tiles_features.append(feats.cpu())
                tiles_coords.extend(batch_coords)
    finally:
        if hasattr(wsi, "close"):
            wsi.close()

    if not tiles_features:
        prob = 0.0
    else:
        feats_all = torch.cat(tiles_features, dim=0)
        coords = torch.tensor(tiles_coords, dtype=torch.float32)
        slide_payload, _ = load_model_from_path(slide_encoder_path, device=device)
        slide_encoder = build_slide_encoder(
            slide_payload,
            slide_encoder_arch,
            in_dim=int(feats_all.shape[1]),
            device=device,
            tile_size=tile_cfg.tile_size,
            embed_dim=longnet_dim,
            depth=longnet_depth,
            slide_ngrids=longnet_slide_ngrids,
        )
        feats_all = feats_all.to(device)
        coords = coords.to(device)
        with torch.no_grad():
            logits = run_slide_encoder(slide_encoder, feats_all, coords)
        prob = logits_to_prob(logits)

    pred = int(prob >= threshold)
    slide_id = wsi_path.stem
    df = pd.DataFrame(
        {
            "ID": [slide_id],
            "Predicted_Label": [pred],
            "Predicted_Prob": [prob],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        level=args.level,
        max_tiles=args.max_tiles,
    )
    if args.csv:
        run_inference_csv(
            csv_path=Path(args.csv),
            output_path=Path(args.output),
            tile_encoder_path=Path(args.tile_encoder_path),
            slide_encoder_path=Path(args.slide_encoder_path),
            tile_cfg=tile_cfg,
            batch_size=args.batch_size,
            input_size=args.input_size,
            threshold=args.threshold,
            device=args.device,
            tile_encoder_arch=args.tile_encoder_arch,
            slide_encoder_arch=args.slide_encoder_arch,
            longnet_dim=args.longnet_dim,
            longnet_depth=args.longnet_depth,
            longnet_slide_ngrids=args.longnet_slide_ngrids,
            wsi_dir=Path(args.wsi_dir) if args.wsi_dir else None,
            wsi_ext=args.wsi_ext,
            csv_wsi_col=args.csv_wsi_col,
            csv_slide_col=args.csv_slide_col,
            csv_gt_col=args.csv_gt_col,
            csv_split_col=args.csv_split_col,
            csv_test_value=args.csv_test_value,
        )
    else:
        run_inference(
            wsi_path=Path(args.wsi),
            output_path=Path(args.output),
            tile_encoder_path=Path(args.tile_encoder_path),
            slide_encoder_path=Path(args.slide_encoder_path),
            tile_cfg=tile_cfg,
            batch_size=args.batch_size,
            input_size=args.input_size,
            threshold=args.threshold,
            device=args.device,
            tile_encoder_arch=args.tile_encoder_arch,
            slide_encoder_arch=args.slide_encoder_arch,
            longnet_dim=args.longnet_dim,
            longnet_depth=args.longnet_depth,
            longnet_slide_ngrids=args.longnet_slide_ngrids,
        )


if __name__ == "__main__":
    main()
