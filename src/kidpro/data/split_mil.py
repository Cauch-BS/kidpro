from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config.schema import AppCfg


def build_mil_split_csv(
    cfg: AppCfg,
    slide_col: str = "SlideName",
    gt_col: str = "GT",
    split_col: str = "split",
    restrict_to_existing_train: bool = False,
) -> pd.DataFrame:
    """
    Build a slide-level MIL CSV with columns: SlideName, GT, split.

    - Reads cfg.dataset.paths.label_csv (must exist).
    - Uses cfg.train.seed as random_state.
    - Writes CSV into run_dir / cfg.dataset.paths.csv_name.
    """

    train_ratio = cfg.dataset.data.train_ratio
    test_ratio = cfg.dataset.data.test_ratio
    val_ratio = cfg.dataset.data.val_ratio

    if cfg.dataset.paths.label_csv is None:
        raise ValueError("dataset.paths.label_csv is required for MIL split building.")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    label_csv = Path(cfg.dataset.paths.label_csv)
    df = pd.read_csv(label_csv)

    # basic column checks
    for c in (slide_col, gt_col):
        if c not in df.columns:
            raise ValueError(f"Label CSV missing required column: {c}")

    # optional: restrict to a pre-defined train pool if your label sheet has it
    if restrict_to_existing_train:
        if split_col not in df.columns:
            raise ValueError(
                f"restrict_to_existing_train=True requires column {split_col!r} in label CSV."
            )
        df = df[df[split_col] == "train"].copy()

    # drop missing GT
    df = df[df[gt_col].notna()].copy()
    if df.empty:
        raise ValueError("No usable rows found after filtering GT notna().")

    # ensure slide-level uniqueness (MIL expects one row per slide)
    # if your label CSV already has one row per slide, this is a no-op
    df = df.drop_duplicates(subset=[slide_col]).copy()

    random_state = cfg.train.seed

    # stratify only if there is more than one class
    stratify = df[gt_col] if df[gt_col].nunique() > 1 else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    # split remaining into val/test
    if (val_ratio + test_ratio) <= 0:
        raise ValueError("val_ratio + test_ratio must be > 0.")

    val_frac_of_temp = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df[gt_col] if temp_df[gt_col].nunique() > 1 else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_frac_of_temp),
        random_state=random_state,
        shuffle=True,
        stratify=stratify_temp,
    )

    # assign split column
    out = df[[slide_col, gt_col]].copy()
    out[split_col] = None
    out.loc[out[slide_col].isin(train_df[slide_col]), split_col] = "train"
    out.loc[out[slide_col].isin(val_df[slide_col]), split_col] = "val"
    out.loc[out[slide_col].isin(test_df[slide_col]), split_col] = "test"

    if out[split_col].isna().any():
        raise RuntimeError("Split assignment failed for some slides (NaN split).")

    # save in run dir, consistent with the rest of your codebase
    run_dir = Path(cfg.run_dir) if cfg.run_dir else Path.cwd()
    csv_path = run_dir / cfg.dataset.paths.csv_name
    out.to_csv(csv_path, index=False)

    print("[OK] MIL CSV saved:", csv_path)
    print("[MIL Split Summary]")
    print("Total:", len(out))
    print("Train:", (out[split_col] == "train").sum())
    print("Val  :", (out[split_col] == "val").sum())
    print("Test :", (out[split_col] == "test").sum())

    return out
