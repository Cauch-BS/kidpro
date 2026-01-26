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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

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
    parser.add_argument("--wsi", required=True, help="Path to input .svs file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
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
    return open_wsi(slide_path)


def tile_image_to_array(tile: object) -> np.ndarray:
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
        pos_embed = self.pos_embed[0, pos, :]
        x = x + pos_embed

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        out = self.encoder(x)
        if self.global_pool:
            pooled = out[:, 1:, :].mean(dim=1)
            outcomes = [self.norm(pooled)]
        else:
            outcomes = [self.norm(out)[:, 0]]
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
            model = AttentionMILHead(in_dim=in_dim, tile_size=tile_size)
        else:
            model = MeanPoolHead(in_dim=in_dim, tile_size=tile_size)
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
    )


if __name__ == "__main__":
    main()
