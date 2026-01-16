# Setup

## Prerequisites

- Python 3.10+
- OpenSlide system library (required by `openslide-python`)
  - macOS: `brew install openslide`
  - Linux: `apt-get install libopenslide-dev` (or your distro equivalent)
  - conda: `conda install -c conda-forge openslide`
- Optional GPU with CUDA for faster training/inference

## Environment

Use the provided conda environment:

```bash
conda env create -f conda/environment.yml && conda activate kidpro
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Model weights

WSI inference falls back to local weights if MLflow resolution fails or is disabled.
Place the checkpoint at: `models/best_model.pt`. One is provided in the repository, and you can replace it with your own model.

You can also override this with `inference.fallback_weights`.

## Sanity check

Confirm entry points are reachable:

```bash
python -m kidpro.train_tile --help
python -m kidpro.train_wsi --help
python -m kidpro.infer_wsi --help
```

See also: `configuration.md`, `data.md`.
