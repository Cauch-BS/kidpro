# Overview

KidPro is a renal pathology pipeline that combines patch-level segmentation and slide-level classification for renal biopsy analysis.

## Components

- **Patch generation**: Creates image+mask patches from XML/SVS annotations (glomerulus, IFTA, inflammation)
- **Tile segmentation**: Trains segmentation models on patch datasets
- **WSI MIL training**: Trains slide-level classifiers using tiles from wsidata caches
- **WSI inference**: Auto-tiles slides and runs classification

## End-to-End Workflow

1. Generate segmentation patches from annotations
2. Preprocess WSIs into wsidata caches (and optional patches)
3. Train tile segmentation
4. Train WSI MIL (with frozen tile encoder)
5. Run WSI inference

See [setup.md](setup.md) for installation, [training.md](training.md) for detailed training workflows, and [data.md](data.md) for dataset formats.
