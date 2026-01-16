# 🔬 Path Kids: AI for Renal Pathology

---

## 🎯 Project Goal

KidPro is a renal pathology pipeline that combines patch-level segmentation and
WSI MIL classification for renal biopsy analysis.

- **Problem definition:** (e.g., delayed diagnosis due to limited pathology experts, difficulty spotting subtle lesions)
- **Approach:** (e.g., CNN-based glomerulus segmentation and classification models)
- **Expected impact:** (e.g., 30% reduction in diagnosis time, higher agreement rates)

## Workflow (Default)

1) Run preprocessing to generate tiles for WSI training.
2) Run `train_tile` to finetune the tile foundation model on glomerulus segmentation (PAS).
3) Freeze the tile model weights and run `train_wsi` using `best_model.pt` from `train_tile`.

Default segmentation training targets glomerulus only.

## Patch Generation (Glomerulus First)

Before segmentation training, generate glomerulus patches from the XML/SVS
annotations. This produces the `Glom_patch` dataset used by `dataset=glom`.

Default run (uses `conf/patch/default.yaml`, which targets glomerulus):

- `python -m kidpro.patch`

Common overrides:

- `python -m kidpro.patch patch.paths.svs_dir=/path/to/Slide`
- `python -m kidpro.patch patch.paths.xml_dir=/path/to/Annotation/Glomerulus`
- `python -m kidpro.patch patch.paths.out_home=/path/to/output`

Optional: generate alternative patch targets when needed:

- IFTA: `python -m kidpro.patch patch=ifta`
- Inflammation: `python -m kidpro.patch patch=inflam`

## MIL Preprocessing (WSI → tiles)

Offline preprocessing writes tiles into the MIL dataset layout expected by `MILDataset`:
`<root_dir>/<SlideName>/images/<SlideName>_X0Y0_XXXXXX_YYYYYY.png`.

Default run (uses `conf/preprocess.yaml`):

- `python -m kidpro.preprocessing`

Inputs:

- `paths.label_csv` must include `SlideName`.
- If the CSV has `image_path` (or `image`) column, those paths are used.
- Otherwise, paths are resolved from `paths.wsi_dir` + `paths.wsi_ext`.

Common overrides:

- `python -m kidpro.preprocessing preprocess.level=1`
- `python -m kidpro.preprocessing preprocess.occupancy_threshold=0.2`
- `python -m kidpro.preprocessing data.patch_size=256`

## Training (Hydra)

Training configs live under `conf/` and are selected via the module entrypoint:

- Tile segmentation: `python -m kidpro.train_tile` (uses `conf/config.yaml`)
- WSI MIL: `python -m kidpro.train_wsi` (uses `conf/config_wsi.yaml`)

Override examples:

- Change dataset preset: `python -m kidpro.train_tile dataset=glom`
- Swap foundation model: `python -m kidpro.train_tile model=prov_gigapath`
- Adjust training params: `python -m kidpro.train_tile train.batch_size=8 train.lr=1e-4`
- Run WSI with frozen tile weights:
  `python -m kidpro.train_wsi`

## Model Artifacts (Reminder)

Inference falls back to local weights when MLflow is unavailable or empty.
Place the WSI checkpoint at:

- `models/best_model.pt`

The `models/` directory is versioned with a `.gitkeep` placeholder, but **the
weights file is not included in the repo**. Remember to add the checkpoint
locally before running inference.

## WSI Inference

Run WSI inference with the default config (`conf/infer_wsi.yaml`):

- `python -m kidpro.infer_wsi inference.wsi_path=/path/to/slide.svs`

Common overrides:

- `python -m kidpro.infer_wsi inference.output_dir=/path/to/output`
- `python -m kidpro.infer_wsi inference.tiles_dir=/path/to/tiles`
- `python -m kidpro.infer_wsi inference.preprocess.level=1`
- `python -m kidpro.infer_wsi mlflow.tracking_uri=http://localhost:5000`

Outputs:

- Prediction JSON at `inference.output_dir/inference.output_json`

## 🏗️ Architecture

Planned AI pipeline overview (to be filled in).
