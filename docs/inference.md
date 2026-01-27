# Inference

**Entrypoint**: `python -m kidpro.infer_wsi`  
**Config**: `kidpro/conf/infer_wsi.yaml`

## Basic Usage

```bash
python -m kidpro.infer_wsi inference.wsi_path=/path/to/slide.svs
```

## Process

1. WSIData cache is built or reused (unless you point to an existing `cache_dir`)
2. A WSI MIL model is loaded
3. Predictions are written to JSON

## Outputs

- `inference.output_dir/inference_output.json` (default: `prediction.json`)
- WSIData cache at `inference.cache_dir/<slide_id>.zarr`
- Optional patches at `inference.patch_dir/<slide_id>/images` (unless cleanup is enabled)

## Common Overrides

```bash
python -m kidpro.infer_wsi inference.output_dir=/path/to/output
python -m kidpro.infer_wsi inference.cache_dir=/path/to/wsidata_cache
python -m kidpro.infer_wsi inference.patch_dir=/path/to/tiles
python -m kidpro.infer_wsi inference.preprocess.level=1
python -m kidpro.infer_wsi inference.cleanup_tiles=true
```

## Model Weights

- If `mlflow.enabled=true`, the model is pulled from the MLflow registry
- Otherwise, fallback weights are loaded from `models/best_model.pt` or `inference.fallback_weights`

See also: [configuration.md](configuration.md), [troubleshooting.md](troubleshooting.md), [training.md](training.md).
