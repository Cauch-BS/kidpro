# Overview

KidPro is a renal pathology pipeline that combines patch-level segmentation and
slide-level classification:

- Patch generation from XML/SVS annotations (glomerulus, IFTA, inflammation).
- Tile segmentation training on patch datasets.
- WSI MIL training on tiles generated from whole-slide images.
- WSI inference that can auto-tile a slide and run classification.

End-to-end flow (default):

1. Generate segmentation patches from annotations.
2. Preprocess WSIs into MIL tiles.
3. Train tile segmentation (`kidpro.train_tile`).
4. Train WSI MIL (`kidpro.train_wsi` with frozen tile encoder).
5. Run WSI inference (`kidpro.infer_wsi`).

Key entrypoints:

- Patch generation: `python -m kidpro.patch`
- Preprocessing: `python -m kidpro.preprocessing`
- Tile training: `python -m kidpro.train_tile`
- WSI training: `python -m kidpro.train_wsi`
- WSI inference: `python -m kidpro.infer_wsi inference.wsi_path = /path/to/slide.svs`

See also: `setup.md`, `data.md`, `training.md`.
