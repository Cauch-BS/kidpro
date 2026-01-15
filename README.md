# 🔬 Path Kids: AI for Renal Pathology

Project Status: Planning
Domain: Medical AI
Python Version: 3.10

> **Path Kids** is a deep learning project to assist and automate renal pathology (Whole Slide Image, WSI) analysis.

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

## Training (Hydra)

Training configs live under `conf/` and are selected via the module entrypoint:

- Segmentation: `python -m kidpro.train` (uses `conf/config.yaml`)
- Classification: `python -m kidpro.train_cls` (uses `conf/config_cls.yaml`)
- MIL: `python -m kidpro.train_mil` (uses `conf/config_mil.yaml`)

Override examples:

- Change dataset preset: `python -m kidpro.train dataset=ifta`
- Swap model: `python -m kidpro.train model=unet train=default`
- Adjust training params: `python -m kidpro.train train.batch_size=8 train.lr=1e-4`

## 🏗️ Architecture

Planned AI pipeline overview (to be filled in).
