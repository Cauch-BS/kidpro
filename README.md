# 🔬 Path Kids: AI for Renal Pathology

---

## 🎯 Project Goal

This project aims to analyze renal biopsy images and solve:

- **Problem definition:** (e.g., delayed diagnosis due to limited pathology experts, difficulty spotting subtle lesions)
- **Approach:** (e.g., CNN-based glomerulus segmentation and classification models)
- **Expected impact:** (e.g., 30% reduction in diagnosis time, higher agreement rates)

## Patch CLI (Hydra)

Hydra configs for patching live under `conf/patch`.

Example usage:

- Default run: `python -m kidpro.patch`
- Preset run: `python -m kidpro.patch --config-name patch/ifta`
- Override paths: `python -m kidpro.patch patch.paths.svs_dir=/data/Slide patch.paths.xml_dir=/data/Annotation/Glomerulus`
- Override params: `python -m kidpro.patch patch.params.overlap_th=0.3 patch.params.num_workers=8`

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

- Segmentation: `python -m kidpro.train` (uses `conf/config.yaml`)
- MIL: `python -m kidpro.train_mil` (uses `conf/config_mil.yaml`)

Override examples:

- Change dataset preset: `python -m kidpro.train dataset=ifta`
- Swap model: `python -m kidpro.train model=unet train=default`
- Adjust training params: `python -m kidpro.train train.batch_size=8 train.lr=1e-4`

## 🏗️ Architecture

Planned AI pipeline overview (to be filled in).
