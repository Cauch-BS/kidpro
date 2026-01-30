from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm import tqdm

from kidpro.config.load import CONFIG, RuntimeResolved
from kidpro.config.schema import AppCfg
from kidpro.data.dataset_mil import MILDataset
from kidpro.data.transform import get_transforms
from kidpro.modeling.factory_wsi import build_model_mil
from kidpro.utils.model_io import load_state_dict_generic

log = logging.getLogger(__name__)

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

parser = argparse.ArgumentParser(description="Run inference on a CSV file (Zarr cache aware)")
parser.add_argument("--csv_path", type=str, default="./query.csv", help="Path to the CSV file")
parser.add_argument("--wsi_dir", type=str, default="./wsi_dir", help="Path to the WSI directory")
parser.add_argument("--tile_encoder_weights_path", type=str, default="./tile_encoder.pth", help="Path to the tile encoder weights file")
parser.add_argument("--slide_encoder_weights_path", type=str, default="./slide_encoder.pt", help="Path to the slide encoder weights file")
parser.add_argument("--slide_col_name", type=str, default="SlideName", help="Name of the slide column")
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--output_csv", type=str, default="analysis.csv", help="Path to the output CSV file")
parser.add_argument("--threshold", type=float, default=0.389, help="Threshold for the prediction")
parser.add_argument("--amp", type=str, default="auto", help="Whether to use automatic mixed precision")

# Cache controls
parser.add_argument("--require_cached", action="store_true", help="Fail if cache miss (do not compute on-the-fly)")


def _resolve_path(p: str | Path, is_file: bool = False) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        if not pp.exists():
            raise FileNotFoundError(f"Path not found: {pp}")
        if is_file and not pp.is_file():
            raise ValueError(f"Path is not a file: {pp}")
        return pp
    pp2 = (Path.cwd() / pp)
    if not pp2.exists():
        raise FileNotFoundError(f"Path not found: {pp2}")
    if is_file and not pp2.is_file():
        raise ValueError(f"Path is not a file: {pp2}")
    return pp2.resolve()


def _resolve_weight_path(infer: Dict[str, Any]) -> tuple[Path, Path]:
    slide_encoder_weights_path = infer.get("slide_encoder_weights_path") or None
    tile_encoder_weights_path = infer.get("tile_encoder_weights_path") or None
    if (not slide_encoder_weights_path) or (not tile_encoder_weights_path):
        raise ValueError("Both slide_encoder_weights_path and tile_encoder_weights_path must be provided")
    return (
        _resolve_path(str(slide_encoder_weights_path), is_file=True),
        _resolve_path(str(tile_encoder_weights_path), is_file=True),
    )


@torch.no_grad()
def _get_tile_embeddings_from_stream(
    *,
    model: Any,
    tile_stream: Any,
    device: str,
    use_amp: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute tile embeddings on cache-miss, mirroring the training pipeline:
    - stream tiles in chunks
    - run tile encoder
    - cache embeddings back into the tile_stream (if supported)
    """
    if not hasattr(tile_stream, "iter_batches"):
        raise ValueError("Expected a tile stream with iter_batches().")

    encode_tiles = getattr(model, "encode_tiles", None)
    tile_encoder = getattr(model, "tile_encoder", None)
    if not callable(encode_tiles) and tile_encoder is None:
        raise TypeError("MIL model must expose encode_tiles() or tile_encoder to compute embeddings.")

    feats_list: list[torch.Tensor] = []
    coords_list: list[torch.Tensor] = []
    tile_count = 0

    for tiles, coords in tile_stream.iter_batches(int(chunk_size)):
        tiles = tiles.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        tile_count += int(tiles.size(0))

        if use_amp:
            with torch.autocast(device_type="cuda"):
                feats = encode_tiles(tiles) if callable(encode_tiles) else tile_encoder(tiles)  # type: ignore[misc]
        else:
            feats = encode_tiles(tiles) if callable(encode_tiles) else tile_encoder(tiles)  # type: ignore[misc]

        feats_list.append(feats)
        coords_list.append(coords)

    if tile_count == 0:
        raise RuntimeError("Empty tile stream for slide.")

    feats_all = torch.cat(feats_list, dim=0)
    coords_all = torch.cat(coords_list, dim=0)

    if hasattr(tile_stream, "set_cached_tile_embeddings"):
        tile_stream.set_cached_tile_embeddings(
            feats_all.detach().cpu().numpy(),
            coords_all.detach().cpu().numpy(),
        )

    return feats_all, coords_all


# -------------------------
# Inference
# -------------------------

def run_csv_ensemble_inference(cfg: AppCfg, rr: RuntimeResolved, infer: Dict[str, Any]) -> Dict[str, Any]:
    csv_path = _resolve_path(str(infer["csv_path"]))
    df = pd.read_csv(csv_path)

    slide_col = infer.get("slide_col", "slide")
    wsi_dir = Path(infer.get("wsi_dir", "."))

    if slide_col not in df.columns:
        raise ValueError(f"CSV missing slide_col={slide_col!r}. Columns={list(df.columns)}")

    output_dir = Path(infer.get("output_dir") or cfg.run_dir or Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / str(infer.get("output_csv", "predictions.csv"))

    threshold = float(infer.get("threshold", 0.389))

    amp_cfg = str(infer.get("amp", "auto")).lower()
    use_amp = (rr.device == "cuda") if amp_cfg == "auto" else (amp_cfg == "true")

    # Cache settings
    require_cached = bool(infer.get("require_cached", False))

    # Ensure cache is enabled in config
    if not (cfg.dataset.data.mil_cache.enabled and cfg.dataset.data.mil_cache.cache_tile_embeddings):
        if require_cached:
            raise RuntimeError(
                "infer_ensem requires cached MIL tile embeddings but "
                "dataset.data.mil_cache.enabled=false or cache_tile_embeddings=false. "
                "Enable them in dataset config (e.g. conf/dataset/wsi.yaml)."
            )
        log.warning(
            "[inference] MIL cache not enabled; will compute embeddings on-the-fly. "
            "Set dataset.data.mil_cache.enabled=true and cache_tile_embeddings=true to enable caching."
        )

    slide_weights_path, tile_weights_path = _resolve_weight_path(infer)

    log.info("[inference] slide_weights=%s", str(slide_weights_path))
    log.info("[inference] tile_encoder_weights=%s", str(tile_weights_path))

    # Build and load model
    model = build_model_mil(cfg)

    # Load tile encoder weights
    if not hasattr(model, "tile_encoder"):
        raise TypeError("MIL model missing tile_encoder; cannot load tile_encoder_weights_path.")
    log.info("[inference] loading tile_encoder weights from %s", str(tile_weights_path))
    model.tile_encoder = load_state_dict_generic(  # type: ignore[attr-defined]
        cast(torch.nn.Module, model.tile_encoder),
        tile_weights_path,
        drop_heads=True,
    )


    # Load slide encoder + classifier weights
    log.info("[inference] loading slide_encoder + classifier from %s", str(slide_weights_path))
    if not hasattr(model, "slide_encoder") or not hasattr(model, "classifier"):
        raise TypeError("MIL model missing slide_encoder/classifier; cannot load WSI checkpoint.")

    model.slide_encoder = load_state_dict_generic(  # type: ignore[attr-defined]
        cast(torch.nn.Module, model.slide_encoder),
        slide_weights_path,
        drop_heads=False,
        ckpt_prefix="slide_encoder.",
    )

    # Check if classifier keys exist in checkpoint
    from kidpro.utils.model_io import _load_checkpoint_state
    checkpoint_state = _load_checkpoint_state(slide_weights_path)
    has_classifier_keys = any(k.startswith("classifier.") for k in checkpoint_state.keys())

    if has_classifier_keys:
        model.classifier = load_state_dict_generic(  # type: ignore[attr-defined]
            cast(torch.nn.Module, model.classifier),
            slide_weights_path,
            drop_heads=False,
            ckpt_prefix="classifier.",
        )
    else:
        log.info("[inference] No classifier.* keys found in checkpoint; skipping classifier load (likely Identity module)")

    model = model.to(rr.device)
    model.eval()

    # Use the same deterministic preprocessing as MIL validation
    _, val_tf = get_transforms(cfg)

    # Create a minimal MIL dataframe so we can reuse MILDataset's embedding-cache reader
    df_mil = pd.DataFrame(
        {
            "SlideName": df[slide_col].astype(str),
            "GT": 0,  # Dummy GT for inference
            "split": "test",  # Dummy split
        }
    )
    # Ensure GT is non-null to satisfy MILDataset invariants
    df_mil["GT"] = pd.to_numeric(df_mil["GT"], errors="coerce").fillna(0).astype(int)
    ds = MILDataset(cfg, df_mil, transform=val_tf)

    out_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0

    pbar = tqdm(range(len(df)), desc="[inference]", unit="slide")
    for idx in pbar:
        row = df.iloc[idx]
        slide_id = str(row[slide_col])

        try:
            # Get tile stream from dataset
            tile_stream, _y_dummy, _slide_name = ds[idx]
            cached = tile_stream.get_cached_tile_embeddings()

            if cached is None:
                if require_cached:
                    raise RuntimeError(
                        f"Missing cached tile embeddings for slide={slide_id}. "
                        "Populate mil_embeds_cache by running MIL training once with "
                        "data.mil_cache.enabled=true and data.mil_cache.cache_tile_embeddings=true "
                        "(the training loop writes caches on cache-miss), or set "
                        "infer.require_cached=false."
                    )
                # Mimic training: compute embeddings on cache-miss (and cache them)
                feats_d, coords_d = _get_tile_embeddings_from_stream(
                    model=model,
                    tile_stream=tile_stream,
                    device=rr.device,
                    use_amp=use_amp,
                    chunk_size=int(cfg.dataset.data.mil_cache.chunk_size),
                )
                tile_feats = feats_d.detach().to("cpu")
                coords = coords_d.detach().to("cpu")
            else:
                emb_np, coords_np = cached
                tile_feats = torch.from_numpy(np.asarray(emb_np, dtype=np.float32))
                coords = torch.from_numpy(np.asarray(coords_np, dtype=np.float32))

            # Run slide encoder
            tile_feats_d = tile_feats.to(rr.device, non_blocking=True)
            coords_d = coords.to(rr.device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model.encode_slide(tile_feats_d, coords_d)
            else:
                logits = model.encode_slide(tile_feats_d, coords_d)

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()

            pred = int(torch.argmax(probs).item())
            pred_prob = float(probs[1].item()) if probs.numel() > 1 else float(probs[0].item())

            if probs.numel() == 2:
                pred = int(pred_prob >= threshold)

            out_rows.append({"ID": slide_id, "Predicted_Label": pred, "Predicted_Prob": pred_prob})
            processed += 1

        except Exception as exc:
            failed += 1
            log.error("[inference] slide=%s failed: %s", slide_id, exc, exc_info=True)

        pbar.set_postfix(
            processed=processed,
            skipped=skipped,
            failed=failed,
        )

    # Output CSV
    if not out_rows:
        log.warning("[inference] No rows processed. Creating empty CSV with expected columns.")
        out_df = pd.DataFrame(columns=["ID", "Predicted_Label", "Predicted_Prob"])
    else:
        out_df = pd.DataFrame(out_rows)[["ID", "Predicted_Label", "Predicted_Prob"]]

    out_df.to_csv(output_csv, index=False)
    log.info("[inference] wrote: %s (%d rows)", output_csv, len(out_df))

    result = {
        "csv_path": str(csv_path),
        "output_csv": str(output_csv),
        "num_rows": int(len(df)),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "require_cached": require_cached,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    args = parser.parse_args()

    # Find config directory
    config_dir = "./conf"
    absolute_config_dir = Path(config_dir).absolute()
    if not absolute_config_dir.exists():
        raise FileNotFoundError(f"Config directory not found. Expected: {config_dir}")

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra and compose config with defaults
    with initialize_config_dir(config_dir=str(absolute_config_dir), version_base=None):
        hcfg = compose(config_name="infer_ensem")

    # Temporarily disable struct mode to allow adding new keys
    OmegaConf.set_struct(hcfg.infer_ensem, False)

    # Override with argparse values
    hcfg.infer_ensem.csv_path = args.csv_path
    if args.wsi_dir:
        hcfg.infer_ensem.wsi_dir = args.wsi_dir
    hcfg.infer_ensem.slide_encoder_weights_path = args.slide_encoder_weights_path
    hcfg.infer_ensem.tile_encoder_weights_path = args.tile_encoder_weights_path
    hcfg.infer_ensem.output_dir = args.output_dir
    hcfg.infer_ensem.output_csv = args.output_csv
    hcfg.infer_ensem.threshold = args.threshold
    hcfg.infer_ensem.amp = args.amp
    if args.slide_col_name:
        hcfg.infer_ensem.slide_col = args.slide_col_name

    # cache overrides
    hcfg.infer_ensem.require_cached = bool(args.require_cached)

    # Re-enable struct mode
    OmegaConf.set_struct(hcfg.infer_ensem, True)

    run_dir = Path.cwd()
    cfg, rr = CONFIG(hcfg, run_dir=run_dir)

    # Your explicit overrides (kept)
    cfg.model.longnet_pretrained = False
    cfg.model.lora.enabled = False
    cfg.model.longnet_depth = 24
    cfg.model.longnet_dim = 1024

    infer = dict(hcfg.get("infer_ensem", {}))
    run_csv_ensemble_inference(cfg, rr, infer)


if __name__ == "__main__":
    main()
