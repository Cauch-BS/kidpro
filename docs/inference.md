# Inference

Entrypoint: `python -m kidpro.infer_wsi`  
Config: `kidpro/conf/infer_wsi.yaml`

Basic run:

```bash
python -m kidpro.infer_wsi inference.wsi_path=/path/to/slide.svs
```

What happens:

1. Tiles are generated for the slide (unless you point to an existing `tiles_dir`).
2. A WSI MIL model is loaded.
3. Predictions are written to JSON.

Outputs:

- `inference.output_dir/inference_output.json` (default: `prediction.json`)
- Tiles at `inference.tiles_dir/<slide_id>/images` (unless cleanup is enabled)

Useful overrides:

```bash
python -m kidpro.infer_wsi inference.output_dir=/path/to/output
python -m kidpro.infer_wsi inference.tiles_dir=/path/to/tiles
python -m kidpro.infer_wsi inference.preprocess.level=1
python -m kidpro.infer_wsi inference.cleanup_tiles=true
```

Weights resolution:

- If `mlflow.enabled=true`, the model is pulled from the MLflow registry.
- Otherwise, fallback weights are loaded from `models/best_model.pt` or
  `inference.fallback_weights`.

See also: `configuration.md`, `troubleshooting.md`.
