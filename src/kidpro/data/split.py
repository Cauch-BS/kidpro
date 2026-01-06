from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..config.schema import AppCfg


def build_dataset_csv(cfg: AppCfg) -> pd.DataFrame:
  root = Path(cfg.dataset.paths.root_dir)
  slides = sorted([p for p in root.iterdir() if p.is_dir()])
  slide_ids = [s.name for s in slides]

  random.shuffle(slide_ids)

  n = len(slide_ids)
  n_test = int(n * cfg.dataset.data.test_ratio)
  n_val = int((n - n_test) * cfg.dataset.data.val_ratio)

  test_slides = set(slide_ids[:n_test])
  val_slides = set(slide_ids[n_test : n_test + n_val])
  train_slides = set(slide_ids[n_test + n_val :])

  def split_of(slide: str) -> str:
    if slide in train_slides:
      return "train"
    if slide in val_slides:
      return "val"
    return "test"

  rows = []
  for slide in slides:
    slide_id = slide.name
    split = split_of(slide_id)

    img_dir = slide / "images"
    mask_root = slide / "masks"

    for img_path in img_dir.glob("*.png"):
      valid = False
      for lid in cfg.dataset.task.layer_ids:
        mp = mask_root / f"layer{lid}" / img_path.name
        if mp.exists():
          valid = True
          break
      if not valid:
        continue

      rows.append({"name": img_path.name, "path": str(img_path), "split": split})

  df = pd.DataFrame(rows)

  # FIX: Handle Optional[Path]
  run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()

  csv_path = run_dir / cfg.dataset.paths.csv_name
  df.to_csv(csv_path, index=False)
  print(f"[OK] CSV saved: {csv_path} (patches={len(df)})")
  return df
