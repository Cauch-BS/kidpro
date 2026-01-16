# Development

## Repo layout

- `src/kidpro/`: main package
  - `patch/`, `preprocessing/`: data generation
  - `training/`: training loops
  - `modeling/`: model factories and backbones
  - `config/`: schema and config loading
- `conf/`: Hydra config tree
- `models/`: local checkpoint location (not versioned)

## Entry points

```bash
python -m kidpro.patch
python -m kidpro.preprocessing
python -m kidpro.train_tile
python -m kidpro.train_wsi
python -m kidpro.infer_wsi
```

## Linting and type checks

Configured tools:

- Ruff: `ruff.toml`
- Mypy: `mypy.ini`

Common commands:

```bash
ruff check .
mypy src/kidpro
```

## Extension points

- Add new datasets under `conf/dataset/*.yaml`.
- Add new backbones under `src/kidpro/modeling/sources/`.
- Extend model selection in `src/kidpro/modeling/factory_*`.

See also: `setup.md`, `configuration.md`.
