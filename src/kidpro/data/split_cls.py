from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config.schema import AppCfg


def build_patch_index(root: Path, exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> dict[str, Path]:
  idx: dict[str, Path] = {}
  for ext in exts:
    for p in root.rglob(f"*{ext}"):
      idx[p.name] = p
  return idx


def get_slide_id_from_patch_name(patch_name: str) -> str:
  # Your notebook rule:
  return patch_name.split("_PAS")[0]


def stratified_split_slide(
  slide_df: pd.DataFrame,
  y_col: str = "slide_y",
  train_ratio: float = 0.8,
  val_ratio: float = 0.1,
  seed: int = 42,
) -> dict[str, str]:
  rng = np.random.RandomState(seed)
  slide_split: dict[str, str] = {}

  for cls, sub in slide_df.groupby(y_col):
    slide_ids = sub["slide_id"].values.copy()
    rng.shuffle(slide_ids)

    n = len(slide_ids)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)

    for sid in slide_ids[:n_tr]:
      slide_split[str(sid)] = "train"
    for sid in slide_ids[n_tr : n_tr + n_va]:
      slide_split[str(sid)] = "val"
    for sid in slide_ids[n_tr + n_va :]:
      slide_split[str(sid)] = "test"

  return slide_split


def build_cls_dataset_csv(cfg: AppCfg) -> pd.DataFrame:
  # Validate required config
  if cfg.dataset.paths.label_csv is None:
    raise ValueError("dataset.paths.label_csv is required for classification datasets.")

  patch_root = Path(cfg.dataset.paths.root_dir)
  label_csv = Path(cfg.dataset.paths.label_csv)

  patch_index = build_patch_index(patch_root)

  label_df = pd.read_csv(label_csv)

  rows = []
  missing = 0
  for _, r in label_df.iterrows():
    name = str(r["patch_name"])
    p = patch_index.get(name)
    if p is None:
      missing += 1
      continue

    # notebook rule: "m1" => 1 else 0
    target = str(r["target"]).lower()
    y = 1 if target == "m1" else 0

    rows.append({"name": name, "path": str(p), "y": y})

  df = pd.DataFrame(rows)
  if df.empty:
    raise RuntimeError("No rows built; check patch_root and label_csv naming alignment.")

  df["slide_id"] = df["name"].apply(get_slide_id_from_patch_name)

  slide_df = (
    df.groupby("slide_id")["y"].max().reset_index().rename(columns={"y": "slide_y"})
  )
  slide_split = stratified_split_slide(
    slide_df,
    train_ratio= cfg.dataset.data.train_ratio,
    val_ratio=cfg.dataset.data.val_ratio,
    seed=cfg.train.seed,
  )

  df["split"] = df["slide_id"].map(slide_split)
  if df["split"].isna().any():
    raise RuntimeError("Split assignment failed for some rows (NaN split).")

  # Save in run dir
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()
  csv_path = run_dir / cfg.dataset.paths.csv_name
  df.to_csv(csv_path, index=False)

  # Optional: print missing count for visibility
  print(f"[OK] CSV saved: {csv_path} (patches={len(df)} missing={missing})")
  return df
