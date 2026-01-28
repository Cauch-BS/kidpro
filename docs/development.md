# Development

## Repo Layout

- `src/kidpro/`: main package
  - `patch/`, `preprocessing/`: data generation
  - `training/`: training loops
  - `modeling/`: model factories and backbones
  - `config/`: schema and config loading
- `kidpro/conf/`: Hydra config tree
- `models/`: local checkpoint location (not versioned)

## Entry Points

```bash
python -m kidpro.patch
python -m kidpro.preprocessing
python -m kidpro.train_tile
python -m kidpro.train_wsi
python -m kidpro.infer_ensem
```

## Linting and Type Checks

**Configured tools**:

- Ruff: `ruff.toml`
- Mypy: `mypy.ini`

**Common commands**:

```bash
ruff check .
mypy src/kidpro
```

## Extension Points

- Add new datasets under `kidpro/conf/dataset/*.yaml`
- Add new backbones under `src/kidpro/modeling/sources/`
- Extend model selection in `src/kidpro/modeling/factory_*`

See also: [setup.md](setup.md), [configuration.md](configuration.md), [architecture.md](architecture.md).
