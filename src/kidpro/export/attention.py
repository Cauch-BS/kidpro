"""Export attention weights and visualizations."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.schema import AppCfg

log = logging.getLogger(__name__)


def export_attention_weights(
  cfg: AppCfg,
  model: nn.Module,
  loader: DataLoader,
  device: str,
  output_dir: Path,
  top_k: int = 10,
) -> None:
  """
  Export attention weights for all slides in the loader.

  Creates for each slide:
    - {slide_name}_attention.csv: patch names + attention weights
    - {slide_name}_topk.json: top-K patches by attention
    - (optional) attention_summary.json: statistics across all slides

  Args:
    cfg: Application config
    model: Trained MIL model with attention
    loader: DataLoader (typically validation set)
    device: "cuda" or "cpu"
    output_dir: Where to save exports
    top_k: Number of top patches to save
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  model.eval()

  all_stats = []

  log.info(f"[EXPORT] Exporting attention weights to: {output_dir}")

  with torch.no_grad():
    for x, y, slide_name in tqdm(loader, desc="Exporting attention"):
      slide_name = slide_name[0] if isinstance(slide_name, (list, tuple)) else slide_name

      x = x.squeeze(0).to(device)  # (N, C, H, W) or (N, D)
      y_true = int(y.item())

      # Forward with attention
      logits, attention_weights = model(x, return_attention=True)

      # Get prediction
      probs = torch.softmax(logits, dim=1)
      y_pred = int(torch.argmax(logits, dim=1).item())
      prob_pos = float(probs[0, 1].item())

      # Convert attention to numpy
      attn = attention_weights.cpu().numpy()  # (N,)

      # Get patch names if available
      # For now, we'll use indices as patch IDs
      # You can enhance this by passing patch names through the dataset
      patch_ids = [f"patch_{i:04d}" for i in range(len(attn))]

      # --- Export 1: Full attention CSV ---
      df_attn = pd.DataFrame({
        "patch_id": patch_ids,
        "attention_weight": attn,
      })
      df_attn = df_attn.sort_values("attention_weight", ascending=False)

      csv_path = output_dir / f"{slide_name}_attention.csv"
      df_attn.to_csv(csv_path, index=False)

      # --- Export 2: Top-K JSON ---
      top_k_actual = min(top_k, len(attn))
      top_indices = np.argsort(attn)[-top_k_actual:][::-1]

      top_patches = {
        "slide_name": slide_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_positive": prob_pos,
        "top_k_patches": [
          {
            "rank": i + 1,
            "patch_id": patch_ids[idx],
            "attention_weight": float(attn[idx]),
          }
          for i, idx in enumerate(top_indices)
        ]
      }

      json_path = output_dir / f"{slide_name}_topk.json"
      with open(json_path, "w") as f:
        json.dump(top_patches, f, indent=2)

      # --- Collect statistics ---
      stats = {
        "slide_name": slide_name,
        "n_patches": len(attn),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_positive": prob_pos,
        "attention_mean": float(np.mean(attn)),
        "attention_std": float(np.std(attn)),
        "attention_max": float(np.max(attn)),
        "attention_min": float(np.min(attn)),
        "attention_entropy": float(-np.sum(attn * np.log(attn + 1e-10))),
      }
      all_stats.append(stats)

  # --- Export 3: Summary statistics ---
  summary_path = output_dir / "attention_summary.json"
  with open(summary_path, "w") as f:
    json.dump(all_stats, f, indent=2)

  # Also save as CSV for easy viewing
  df_summary = pd.DataFrame(all_stats)
  df_summary.to_csv(output_dir / "attention_summary.csv", index=False)

  log.info(f"[EXPORT] Exported attention for {len(all_stats)} slides")
  log.info(f"[EXPORT] Files saved to: {output_dir}")

  # --- Sanity checks ---
  _run_sanity_checks(all_stats)


def _run_sanity_checks(stats: list[dict]) -> None:
  """Run sanity checks on exported attention weights."""
  log.info("[SANITY] Running attention sanity checks...")

  # Check 1: Attention weights sum to ~1.0
  errors = []
  for s in stats:
    # Approximate check (weights are normalized via softmax)
    attn_mean = s["attention_mean"]
    n_patches = s["n_patches"]
    expected_mean = 1.0 / n_patches

    # Mean should be close to 1/N
    if abs(attn_mean - expected_mean) > 0.01:
      errors.append(f"{s['slide_name']}: mean={attn_mean:.4f}, expected~{expected_mean:.4f}")

  if errors:
    log.warning(f"[SANITY] {len(errors)} slides with unusual attention means:")
    for e in errors[:5]:
      log.warning(f"  {e}")
  else:
    log.info("[SANITY] ✓ Attention weights normalized correctly")

  # Check 2: Entropy reasonable (not all weight on one patch)
  low_entropy = [s for s in stats if s["attention_entropy"] < 0.5]
  if low_entropy:
    log.warning(f"[SANITY] {len(low_entropy)} slides with very low entropy (concentrated attention):")
    for s in low_entropy[:5]:
      log.warning(f"  {s['slide_name']}: entropy={s['attention_entropy']:.4f}")
  else:
    log.info("[SANITY] ✓ Attention entropy in reasonable range")

  # Check 3: Max attention not too dominant
  high_max = [s for s in stats if s["attention_max"] > 0.5]
  if high_max:
    log.info(f"[SANITY] {len(high_max)} slides with dominant patch (attn_max > 0.5)")

  log.info("[SANITY] Sanity checks complete!")
