# Data and Datasets

## Patch generation (segmentation)

Patch generation uses XML annotations and SVS slides to create an image+mask dataset.
Default config: `conf/patch/default.yaml`.

Command:

```bash
python -m kidpro.patch
```

Inputs:

- `patch.paths.svs_dir`: directory with `*.svs`
- `patch.paths.xml_dir`: directory with annotation `*.xml`
- `patch.segmentation_type`: one of `glomerulus`, `ifta`, `inflammation`

Outputs (by default):

```text
<out_home>/<output_map>/<SlideName>/
  images/<SlideName>_X0Y0_XXXXXX_YYYYYY.png
  masks/layer<id>/<SlideName>_X0Y0_XXXXXX_YYYYYY.png
```

The `layer_ids` list controls which mask layers are produced.

## Segmentation training CSV

`kidpro.train_tile` builds a patch-level CSV in the Hydra run directory. The CSV has:

- `name`: patch file name
- `path`: full path to patch
- `split`: train/val/test

Splits are generated using the ratios in `conf/dataset/*.yaml`.

## MIL preprocessing (WSI to tiles)

Preprocessing converts WSIs into a MIL tile layout expected by `MILDataset`.
Default config: `conf/preprocess.yaml`.

Command:

```bash
python -m kidpro.preprocessing
```

Required label CSV columns: `SlideName`, `GT`

WSI discovery:

- If the label CSV includes `image` or `image_path`, those are used.
- Otherwise, the path is resolved from `paths.wsi_dir` + `paths.wsi_ext`.

Output layout:

```text
<root_dir>/<SlideName>/images/<SlideName>_X0Y0_XXXXXX_YYYYYY.png
```

Tile filenames encode top-left coordinates in the WSI.

See also: `configuration.md`, `training.md`.
