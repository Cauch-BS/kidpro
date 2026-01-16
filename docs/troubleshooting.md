# Troubleshooting

## Missing OpenSlide library

Error: `ImportError: libopenslide` or `openslide` fails to import.
Fix: install the system dependency (`brew install openslide` or distro package).

## No patches found (MIL)

Error: `No patches found for slide ...`
Fix: confirm `paths.root_dir` points to the MIL tiles folder and that
`<root_dir>/<SlideName>/images/*.png` exists.

## No XML files found

Error: `No XML files found in: ...`
Fix: set `patch.paths.xml_dir` to the annotation folder.

## Missing fallback weights

Error: `Fallback weights not found at ...`
Fix: place `best_model.pt` in `models/` or set `inference.fallback_weights`.

## MLflow resolution failed

Warning: `MLflow resolution failed; falling back to local weights.`
Fix: set `mlflow.tracking_uri` and ensure the model registry has the
registered name in `mlflow.registry_model_name`.

## WSI training requires frozen backbone

Error: `train_wsi requires model.freeze_backbone=true.`
Fix: run:

```bash
python -m kidpro.train_wsi model.freeze_backbone=true
```

See also: `setup.md`, `inference.md`.
