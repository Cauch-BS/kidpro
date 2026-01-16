# Configuration

KidPro uses Hydra for configuration. The main configs live under `conf/` and are
selected by module entrypoints.

Key configs:

- `conf/config.yaml`: tile segmentation training
- `conf/config_wsi.yaml`: WSI MIL training
- `conf/infer_wsi.yaml`: inference
- `conf/preprocess.yaml`: MIL preprocessing
- `conf/patch/*.yaml`: patch generation
- `conf/dataset/*.yaml`: dataset presets
- `conf/model/*.yaml`: model presets

Hydra defaults specify the active presets. Example from `conf/config.yaml`:

```yaml
defaults:
  - hydra: default
  - model: prov_gigapath
  - train: default
  - dataset: glom
  - mlflow: default
  - core: default
  - _self_
```

Common override patterns:

```bash
python -m kidpro.train_tile dataset=glom
python -m kidpro.train_tile model=virchow2 train.batch_size=8
python -m kidpro.train_wsi dataset.paths.label_csv=/path/to/labels.csv
python -m kidpro.infer_wsi inference.wsi_path=/path/to/slide.svs
```

Run directories:

- Hydra changes the working directory to a unique run dir.
- The run dir is stored in `cfg.run_dir` and is used for outputs.

Config export:

- Resolved config: `config_resolved.yaml`
- Environment snapshot: `training_env.json`
- Best checkpoint: `best_model.pt`

See also: `training.md`, `inference.md`.
