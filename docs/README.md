# KidPro Documentation

This folder contains the onboarding and reference docs for the KidPro codebase.

## Getting Started

- **[Overview](overview.md)**: What the project does and the end-to-end workflow
- **[Setup](setup.md)**: Environment setup and required local assets
- **[Data](data.md)**: Dataset formats and preprocessing inputs/outputs

## Core Workflows

- **[Training](training.md)**: Training workflows for tile segmentation and WSI MIL
- **[Inference](inference.md)**: WSI inference steps and outputs
- **[RankMix](rankmix.md)**: Advanced data augmentation for class imbalance (optional)

## Reference

- **[Configuration](configuration.md)**: Hydra config structure and common overrides
- **[Architecture](architecture.md)**: Component map and data flow diagram
- **[Development](development.md)**: Repo layout, extension points, and dev tooling
- **[Troubleshooting](troubleshooting.md)**: Common pitfalls and fixes

## Quickstart

1. **Setup environment**:
   ```bash
   conda env create -f conda/environment.yml && conda activate kidpro
   ```

2. **Generate patches** (segmentation dataset):
   ```bash
   python -m kidpro.patch
   ```

3. **Preprocess for MIL** (build wsidata cache + optional patches):
   ```bash
   python -m kidpro.preprocessing
   ```

4. **Train tile segmentation**:
   ```bash
   python -m kidpro.train_tile
   ```

5. **Train WSI MIL** (requires frozen tile encoder):
   ```bash
   python -m kidpro.train_wsi model.freeze_backbone=true
   ```

6. **Run inference**:
   ```bash
   python -m kidpro.infer_wsi inference.wsi_path=/path/to/slide.svs
   ```

> **Note**: Unless otherwise specified, all commands should be run from the root of the repository.

For advanced training options including RankMix data augmentation for class imbalance, see [rankmix.md](rankmix.md).
