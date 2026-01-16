# Training

## Tile segmentation training

Entrypoint: `python -m kidpro.train_tile`  
Config: `conf/config.yaml` (defaults to `dataset=glom`, `model=prov_gigapath`)

Typical run:

```bash
python -m kidpro.train_tile
```

Common overrides:

```bash
python -m kidpro.train_tile dataset=glom train.batch_size=8 train.lr=1e-4
python -m kidpro.train_tile model=uni2_h
```

Outputs:

- Best checkpoint saved to `<run_dir>/best_model.pt` (see `conf/core/default.yaml`)
- Resolved config at `<run_dir>/config_resolved.yaml`
- Environment snapshot at `<run_dir>/training_env.json`

## WSI MIL training

Entrypoint: `python -m kidpro.train_wsi`  
Config: `conf/config_wsi.yaml` (defaults to `dataset=wsi`)

Important: WSI training requires the tile encoder to be frozen:

```bash
python -m kidpro.train_wsi model.freeze_backbone=true
```

Common overrides:

```bash
python -m kidpro.train_wsi train.lr=1e-4 train.epochs=50
python -m kidpro.train_wsi dataset.paths.label_csv=/path/to/labels.csv
```

LoRA initialization (optional):

- If `model.lora.enabled=true`, the tile encoder is initialized from MLflow.
- Configure `mlflow.enabled=true` and `mlflow.registry_model_name`.

Outputs:

- Best checkpoint saved to `<run_dir>/best_model.pt`
- Summary JSON at `<run_dir>/best_summary.json`

See also: `configuration.md`, `data.md`.
