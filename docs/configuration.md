# Configuration

KidPro uses Hydra for configuration. The main configs live under `kidpro/conf/` and are
selected by module entrypoints.

Key configs:

- `kidpro/conf/config.yaml`: tile segmentation training
- `kidpro/conf/config_wsi.yaml`: WSI MIL training
- `kidpro/conf/infer_wsi.yaml`: inference
- `kidpro/conf/preprocess.yaml`: MIL preprocessing
- `kidpro/conf/patch/*.yaml`: patch generation
- `kidpro/conf/dataset/*.yaml`: dataset presets
- `kidpro/conf/model/*.yaml`: model presets

Hydra defaults specify the active presets. Example from `kidpro/conf/config.yaml`:

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
python -m kidpro.infer_wsi inference.cache_dir=/path/to/wsidata_cache
python -m kidpro.preprocessing preprocess.export_patches=false
```

Run directories:

- Hydra changes the working directory to a unique run dir.
- The run dir is stored in `cfg.run_dir` and is used for outputs.

Config export:

- Resolved config: `config_resolved.yaml`
- Environment snapshot: `training_env.json`
- Best checkpoint: `best_model.pt`

## LongNet MIL Configuration

### slide_ngrids Parameter

The `slide_ngrids` parameter defines the grid resolution for positional embeddings in the LongNet slide encoder. It determines the maximum spatial extent that the model can handle for whole-slide images.

**How it works:**

- Creates a 2D grid of size `slide_ngrids × slide_ngrids` for positional encoding
- Default value: `1000` (resulting in 1,000,000 possible patch positions)
- Tile coordinates are normalized to this grid during inference and training
- Each position in the grid gets a unique 2D sinusoidal positional embedding

**Selection criteria:**

The default value of 1000 is chosen to:

1. **Accommodate large WSIs**: A 1000×1000 grid with 256px tiles covers slides up to ~256,000×256,000 pixels at the tile extraction level
2. **Balance memory and coverage**: Larger grids increase memory for positional embeddings but provide finer spatial resolution
3. **Match pretrained weights**: When using `longnet_pretrained=true` with pretrained slide encoders (e.g., from prov_gigapath), the grid size must match the pretrained model's expectations

**When to adjust:**

You may need to change `slide_ngrids` if:

- Working with exceptionally large or small WSIs
- Using pretrained slide encoder weights with different grid expectations
- Memory constraints require a smaller positional embedding table
- You need finer spatial resolution for very dense patch sampling

**Configuration:**

Override in model configs or via command line:

```bash
# In model YAML (e.g., kidpro/conf/model/virchow2.yaml)
longnet_slide_ngrids: 1000

# Via command line
python -m kidpro.train_wsi model.longnet_slide_ngrids=1000
python -m kidpro.infer_wsi model.longnet_slide_ngrids=1000
```

**Related parameters:**

- `longnet_max_wsi_size`: Maximum pixel size for WSI (default: 262144)
- `tile_size`: Size of each tile/patch (default: 256)
- `longnet_dim`: Embedding dimension (affects positional embedding size)

See also: `training.md`, `inference.md`.
