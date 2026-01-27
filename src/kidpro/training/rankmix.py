"""RankMix: Data Augmentation for Weakly Supervised Learning of WSI Classification.

Implementation based on:
  Chen & Lu, "RankMix: Data Augmentation for Weakly Supervised Learning of
  Classifying Whole Slide Images with Diverse Sizes and Imbalanced Categories",
  CVPR 2023.

RankMix mixes ranked tile embeddings from pairs of WSIs to create augmented
training samples. This is particularly effective for severe class imbalance.

Key components:
  - TileScorer: MLP that predicts tile importance from tile embeddings
  - rank_and_select: Selects top-k tiles by score while preserving spatial order
  - rankmix: Mixes tile embeddings and labels from two slides
  - RankMixSampler: Samples slide pairs with bias toward minority class
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


class TileScorer(nn.Module):
  """MLP that predicts tile importance from tile embeddings.

  The score function predicts tile-level pseudo-labels from slide-level labels,
  enabling identification of which tiles are most relevant for classification.

  Args:
    embed_dim: Dimension of tile embeddings (e.g., 1536 for GigaPath)
    hidden_dim: Hidden layer dimension (default: embed_dim // 2)
  """

  def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None) -> None:
    super().__init__()
    hidden_dim = hidden_dim or (embed_dim // 2)
    self.mlp = nn.Sequential(
      nn.Linear(embed_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1),
      nn.Sigmoid(),
    )

  def forward(self, tile_embeds: Tensor) -> Tensor:
    """Compute importance scores for each tile.

    Args:
      tile_embeds: Tensor of shape (N, D) where N is number of tiles

    Returns:
      Tensor of shape (N,) with importance scores in [0, 1]
    """
    output: Tensor = self.mlp(tile_embeds).squeeze(-1)
    return output


def rank_and_select(embeds: Tensor, scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
  """Select top-k embeddings by score, preserving original spatial order.

  Args:
    embeds: Tile embeddings of shape (N, D)
    scores: Importance scores of shape (N,)
    k: Number of tiles to select

  Returns:
    Tuple of (selected_embeds, selected_indices) where selected_embeds has
    shape (k, D) and indices are sorted to preserve spatial order.
  """
  # Get indices of top-k scoring tiles
  top_k_indices = torch.argsort(scores, descending=True)[:k]
  # Sort indices to preserve original spatial order
  sorted_indices = torch.sort(top_k_indices)[0]
  return embeds[sorted_indices], sorted_indices


def rankmix(
  emb_a: Tensor,
  emb_b: Tensor,
  score_a: Tensor,
  score_b: Tensor,
  y_a: int,
  y_b: int,
  alpha: float = 1.0,
) -> tuple[Tensor, float, float]:
  """Mix ranked tile embeddings from two slides.

  Implements the core RankMix algorithm:
    1. Select top-k tiles from each slide (k = min(len(a), len(b)))
    2. Sample mixing ratio λ from Beta(α, α)
    3. Mix embeddings: H_mix = λ*H_a + (1-λ)*H_b
    4. Mix labels: Y_mix = λ*Y_a + (1-λ)*Y_b

  Args:
    emb_a: Tile embeddings from slide A, shape (N_a, D)
    emb_b: Tile embeddings from slide B, shape (N_b, D)
    score_a: Importance scores for slide A, shape (N_a,)
    score_b: Importance scores for slide B, shape (N_b,)
    y_a: Label for slide A (0 or 1)
    y_b: Label for slide B (0 or 1)
    alpha: Beta distribution parameter (higher = more uniform λ distribution)

  Returns:
    Tuple of (mixed_embeddings, mixed_label, lambda_value)
    - mixed_embeddings: Shape (k, D) where k = min(N_a, N_b)
    - mixed_label: Float in [0, 1]
    - lambda_value: The sampled mixing ratio
  """
  k = min(len(emb_a), len(emb_b))

  # Select top-k embeddings from each slide
  h_a, _ = rank_and_select(emb_a, score_a, k)
  h_b, _ = rank_and_select(emb_b, score_b, k)

  # Sample mixing ratio
  lam = float(np.random.beta(alpha, alpha))

  # Mix embeddings and labels
  h_mxp = lam * h_a + (1 - lam) * h_b
  y_mxp = lam * float(y_a) + (1 - lam) * float(y_b)

  return h_mxp, y_mxp, lam


class RankMixSampler:
  """Samples slide pairs with bias toward minority class.

  For class imbalance, we want the minority class (GT=True) to participate
  in more mixed samples. This sampler ensures that with probability
  `minority_ratio`, at least one slide in the pair comes from the minority class.

  Args:
    df: DataFrame with columns ["SlideName", "GT"]
    minority_label: The label value for the minority class (default: 1 for GT=True)
    minority_ratio: Probability of including a minority slide in a pair (default: 0.7)
  """

  def __init__(
    self,
    df: pd.DataFrame,
    minority_label: int = 1,
    minority_ratio: float = 0.7,
  ) -> None:
    self.df = df.reset_index(drop=True)
    self.minority_ratio = minority_ratio

    # Separate indices by class
    self.minority_indices = df[df["GT"] == minority_label].index.tolist()
    self.majority_indices = df[df["GT"] != minority_label].index.tolist()

    if not self.minority_indices:
      log.warning("[RANKMIX] No minority samples found (GT=%d). RankMix may not be effective.", minority_label)
    if not self.majority_indices:
      log.warning("[RANKMIX] No majority samples found. RankMix may not be effective.")

    log.info(
      "[RANKMIX SAMPLER] minority=%d samples, majority=%d samples, minority_ratio=%.2f",
      len(self.minority_indices),
      len(self.majority_indices),
      minority_ratio,
    )

  def sample_pair(self) -> tuple[int, int]:
    """Sample a pair of slide indices for mixing.

    Returns:
      Tuple of (idx_a, idx_b) where idx_a is biased toward minority class.
    """
    # With probability minority_ratio, sample first slide from minority class
    if self.minority_indices and random.random() < self.minority_ratio:
      idx_a = random.choice(self.minority_indices)
    else:
      idx_a = random.choice(self.majority_indices) if self.majority_indices else random.randint(0, len(self.df) - 1)

    # Sample second slide uniformly
    idx_b = random.randint(0, len(self.df) - 1)

    return idx_a, idx_b

  def sample_partner_idx(self) -> int:
    """Sample a single slide index for mixing with current slide.

    This is used when the current slide is already determined by the dataloader.

    Returns:
      Index of a slide to mix with.
    """
    # Bias toward minority class
    if self.minority_indices and random.random() < self.minority_ratio:
      return int(random.choice(self.minority_indices))
    return random.randint(0, len(self.df) - 1)

  def get_slide_name(self, idx: int) -> str:
    """Get slide name for an index."""
    return str(self.df.iloc[idx]["SlideName"])

  def get_label(self, idx: int) -> int:
    """Get label for an index."""
    return int(self.df.iloc[idx]["GT"])


def compute_rankmix_loss(
  logits: Tensor,
  soft_target: float,
) -> Tensor:
  """Compute binary cross-entropy loss for soft (mixed) labels.

  Uses PyTorch's built-in BCE with logits, which natively supports soft labels.

  Args:
    logits: Model output logits of shape (1, 2) for binary classification
    soft_target: Soft label in [0, 1] representing mixed ground truth

  Returns:
    Scalar loss tensor
  """
  # Convert 2-class logits to binary logit (positive class - negative class)
  # This gives the log-odds of the positive class
  pos_logit = logits[:, 1] - logits[:, 0]

  # Create target tensor
  target = torch.tensor([soft_target], device=logits.device, dtype=torch.float32)

  # Use PyTorch's built-in BCE with logits
  return F.binary_cross_entropy_with_logits(pos_logit, target)
