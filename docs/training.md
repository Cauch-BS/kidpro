# Training

## Tile Segmentation Training

**Entrypoint**: `python -m kidpro.train_tile`  
**Config**: `kidpro/conf/config.yaml` (defaults to `dataset=glom`, `model=prov_gigapath`)

### Basic Usage

```bash
python -m kidpro.train_tile
```

### Common Overrides

```bash
python -m kidpro.train_tile dataset=glom train.batch_size=8 train.lr=1e-4
python -m kidpro.train_tile model=uni2_h
```

### Outputs

- Best checkpoint: `<run_dir>/best_model.pt`
- Resolved config: `<run_dir>/config_resolved.yaml`
- Environment snapshot: `<run_dir>/training_env.json`

## WSI MIL Training

**Entrypoint**: `python -m kidpro.train_wsi`  
**Config**: `kidpro/conf/config_wsi.yaml` (defaults to `dataset=wsi`)

### Requirements

- WSI training reads tiles from wsidata caches created by `kidpro.preprocessing`
- Patch export is optional for inspection only
- **The tile encoder must be frozen**:

```bash
python -m kidpro.train_wsi model.freeze_backbone=true
```

### Common Overrides

```bash
python -m kidpro.train_wsi train.lr=1e-4 train.epochs=50
python -m kidpro.train_wsi dataset.paths.label_csv=/path/to/labels.csv
```

### Advanced Options

**LoRA initialization** (optional):
- Set `model.lora.enabled=true` to initialize tile encoder from MLflow
- Configure `mlflow.enabled=true` and `mlflow.registry_model_name`

**RankMix augmentation** (for class imbalance):
- See [rankmix.md](rankmix.md) for two-stage training with RankMix data augmentation
- Useful when dealing with severe class imbalance

### Outputs

- Best checkpoint: `<run_dir>/best_model.pt`
- Summary JSON: `<run_dir>/best_summary.json`

See also: [configuration.md](configuration.md), [data.md](data.md), [rankmix.md](rankmix.md).
