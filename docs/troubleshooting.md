# Troubleshooting

## Missing OpenSlide Library

**Error**: `ImportError: libopenslide` or `openslide` fails to import.

**Fix**: Install the system dependency:
- macOS: `brew install openslide`
- Linux: `apt-get install libopenslide-dev` (or your distro equivalent)
- conda: `conda install -c conda-forge openslide`

## No Patches Found (MIL)

**Error**: `No patches found for slide ...`

**Fix**: Confirm `paths.root_dir` points to the MIL tiles folder and that `<root_dir>/<SlideName>/images/*.png` exists.

## No XML Files Found

**Error**: `No XML files found in: ...`

**Fix**: Set `patch.paths.xml_dir` to the annotation folder.

## Missing Fallback Weights

**Error**: `Fallback weights not found at ...`

**Fix**: Place `best_model.pt` in `models/` or set `inference.fallback_weights`.

## MLflow Resolution Failed

**Warning**: `MLflow resolution failed; falling back to local weights.`

**Fix**: Set `mlflow.tracking_uri` and ensure the model registry has the registered name in `mlflow.registry_model_name`.

## WSI Training Requires Frozen Backbone

**Error**: `train_wsi requires model.freeze_backbone=true.`

**Fix**: Run:

```bash
python -m kidpro.train_wsi model.freeze_backbone=true
```

See also: [setup.md](setup.md), [inference.md](inference.md), [training.md](training.md).
