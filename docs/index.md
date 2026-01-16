# KidPro Documentation

This folder contains the onboarding and reference docs for the KidPro codebase.

Start here:

- `overview.md`: What the project does and the end-to-end workflow.
- `setup.md`: Environment setup and required local assets.
- `data.md`: Dataset formats and preprocessing inputs/outputs.
- `training.md`: Training workflows for tile segmentation and WSI MIL.
- `inference.md`: WSI inference steps and outputs.
- `configuration.md`: Hydra config structure and common overrides.
- `architecture.md`: Component map and data flow diagram.
- `development.md`: Repo layout, extension points, and dev tooling.
- `troubleshooting.md`: Common pitfalls and fixes.

Quickstart (happy path):

1. Create the environment:
   `conda env create -f conda/environment.yml && conda activate kidpro`
2. Generate patches (segmentation dataset):
   `python -m kidpro.patch`
3. Preprocess for MIL:
   `python -m kidpro.preprocessing`
4. Train tile segmentation:
   `python -m kidpro.train_tile`
5. Train WSI MIL (requires frozen tile encoder):
   `python -m kidpro.train_wsi model.freeze_backbone=true`
6. Run inference:
   `python -m kidpro.infer_wsi inference.wsi_path=/path/to/slide.svs`

> [!NOTE] Unless otherwise specified, all commands should be run from the root of the repository.
