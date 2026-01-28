# Inference

**Entrypoint**: `python -m kidpro.infer_ensem`  
**Config**: `kidpro/conf/infer_ensem.yaml`

## Basic Usage

```bash
python -m kidpro.infer_ensem infer_ensem.csv_path=/path/to/labels.csv
```

## Process

1. For each slide row in the CSV, a WSIData cache is reused (or created if enabled)
2. One or more WSI MIL checkpoints are loaded (optional ensembling)
3. Predictions are written to a CSV, with an accompanying `metrics.json`

## Outputs

- Submission CSV at `infer_ensem.output_dir/infer_ensem.output_csv` (default: `submission.csv`)
- Metrics JSON at `infer_ensem.output_dir/metrics.json`
- WSIData cache at `infer_ensem.cache_dir/<slide_id>.zarr` (or `dataset.paths.wsi_cache_dir` if configured)

## Common Overrides

```bash
python -m kidpro.infer_ensem infer_ensem.output_dir=/path/to/output
python -m kidpro.infer_ensem infer_ensem.cache_dir=/path/to/wsidata_cache
python -m kidpro.infer_ensem infer_ensem.ensure_cache=true
python -m kidpro.infer_ensem infer_ensem.weights_paths='[/path/to/model1.pt,/path/to/model2.pt]'
```

## Model Weights

- If `mlflow.enabled=true`, the model is pulled from the MLflow registry
- Otherwise, fallback weights are loaded from `models/best_model.pt` or `infer_ensem.fallback_weights`

See also: [configuration.md](configuration.md), [troubleshooting.md](troubleshooting.md), [training.md](training.md).
