"""Loss functions for training."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyWithBrierLoss(nn.Module):
  """Combined CrossEntropy and Brier score loss to prevent overconfidence.

  The Brier score component penalizes overconfident predictions by measuring
  the mean squared error between predicted probabilities and true labels.
  This helps calibrate the model's probability estimates.

  Loss = CrossEntropyLoss + brier_weight * BrierScore

  Args:
    weight: Optional class weights for CrossEntropyLoss (tensor of shape (num_classes,))
    brier_weight: Weight for the Brier score component (default: 0.1)
    reduction: Reduction method for CrossEntropyLoss ('mean', 'sum', or 'none')
  """

  def __init__(
    self,
    weight: Tensor | None = None,
    brier_weight: float = 0.1,
    reduction: str = "mean",
  ) -> None:
    super().__init__()
    self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    self.brier_weight = brier_weight
    self.reduction = reduction

  def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    """Compute combined loss.

    Args:
      logits: Model output logits of shape (batch_size, num_classes)
      target: Ground truth labels of shape (batch_size,) with class indices

    Returns:
      Combined loss tensor (scalar if reduction='mean')
    """
    # Cross-entropy loss
    ce = self.ce_loss(logits, target)
    # Ensure ce is a Tensor (mypy type narrowing)
    if not isinstance(ce, Tensor):
      raise TypeError(f"CrossEntropyLoss returned {type(ce)}, expected Tensor")

    # Brier score: mean squared error between predicted probabilities and one-hot targets
    probs = F.softmax(logits, dim=1)
    num_classes = logits.size(1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()

    # Brier score = mean((predicted_prob - true_prob)^2) over all classes
    brier = torch.mean((probs - target_one_hot) ** 2)

    # Combine losses
    total_loss = ce + self.brier_weight * brier

    return total_loss


class FeatMag(nn.Module):
  """Feature Magnitude Loss for FRMIL.

  Computes margin-based loss between feature representations of normal and anomaly samples.
  """

  def __init__(self, margin: float = 8.48) -> None:
    """
    Args:
      margin: Margin parameter for the loss
    """
    super().__init__()
    self.margin = margin

  def forward(
    self, feat_ano: Tensor, feat_norm: Tensor, w_scale: float = 1.0
  ) -> Tensor:
    """
    Compute feature magnitude loss.

    Args:
      feat_ano: Anomaly features of shape (batch, num_tiles, feat_dim)
      feat_norm: Normal features of shape (batch, num_tiles, feat_dim)
      w_scale: Scaling factor (typically number of tiles)

    Returns:
      Scalar loss tensor
    """
    # Compute mean feature magnitude for each sample
    mag_ano = torch.mean(torch.norm(feat_ano, dim=-1), dim=-1)  # (batch,)
    mag_norm = torch.mean(torch.norm(feat_norm, dim=-1), dim=-1)  # (batch,)

    # Margin-based loss
    loss = F.relu(self.margin - (mag_ano - mag_norm)) / w_scale
    return loss.mean()


def compute_dsmil_loss(
  instance_predictions: Tensor,
  bag_prediction: Tensor,
  target: Tensor,
  num_classes: int = 2,
) -> Tensor:
  """Compute DSMIL loss: combination of bag loss and max instance loss.

  Loss = 0.5 * bag_loss + 0.5 * max_loss

  Args:
    instance_predictions: Instance predictions of shape (num_tiles, num_classes)
    bag_prediction: Bag prediction of shape (1, num_classes)
    target: Target label of shape (1,) for binary or (1, num_classes) for multi-class
    num_classes: Number of classes

  Returns:
    Scalar loss tensor
  """
  criterion = nn.BCEWithLogitsLoss()

  # Prepare target for bag loss
  if num_classes == 1:
    target_bag = target.view(1, -1)
  else:
    # Multi-class case
    if target.ndim == 1:
      # Convert 1D target to one-hot: (1,) -> (1, num_classes)
      target_bag = torch.zeros_like(bag_prediction)
      target_class_idx = int(target.item())
      target_bag[0, target_class_idx] = 1.0
    else:
      # Already 2D: (1, num_classes) - use as-is (handles RankMix soft labels)
      target_bag = target if target.shape[0] == 1 else target.view(1, -1)

  # Prepare target for max instance loss
  if num_classes == 1:
    target_max = target.view(1, -1)
  else:
    # Multi-class case
    if target.ndim == 1:
      # Convert 1D target to one-hot: (1,) -> (num_classes,)
      target_max_onehot = torch.zeros(num_classes, device=target.device, dtype=bag_prediction.dtype)
      target_class_idx = int(target.item())
      target_max_onehot[target_class_idx] = 1.0
      target_max = target_max_onehot.view(1, -1)
    else:
      # Already 2D: (1, num_classes) - extract first row for max loss
      # For RankMix, this preserves soft label values
      target_max = target[0:1, :] if target.shape[0] == 1 else target.view(1, -1)

  # Bag loss
  bag_loss = criterion(bag_prediction.view(1, -1), target_bag.view(1, -1))

  # Max instance loss
  max_prediction, _ = torch.max(instance_predictions, dim=0)  # (num_classes,)
  max_loss = criterion(max_prediction.view(1, -1), target_max.view(1, -1))

  loss = 0.5 * bag_loss + 0.5 * max_loss
  return cast(Tensor, loss)


def compute_frmil_loss(
  bag_prediction: Tensor,
  instance_predictions: Tensor,
  query_features: Tensor,
  target: Tensor,
  num_classes: int = 2,
  class_weights: Tensor | None = None,
  pos_weight: Tensor | None = None,
  margin: float = 8.48,
  norm_idx: int | None = None,
  ano_idx: int | None = None,
) -> Tensor:
  """Compute FRMIL loss: combination of bag loss, max instance loss, and feature magnitude loss.

  Loss = (bag_loss + max_loss + loss_ft) / 3

  Args:
    bag_prediction: Bag prediction of shape (1, num_classes)
    instance_predictions: Instance predictions (attention scores) of shape (num_tiles,)
    query_features: Query features (shifted features) of shape (num_tiles, feat_dim) or (batch, num_tiles, feat_dim)
    target: Target label of shape (1,) for binary or (1, num_classes) for multi-class
    num_classes: Number of classes
    class_weights: Class weights for cross-entropy loss (for multi-class)
    pos_weight: Positive class weight for binary cross-entropy
    margin: Margin for feature magnitude loss
    norm_idx: Index of normal sample (for feature magnitude loss)
    ano_idx: Index of anomaly sample (for feature magnitude loss)

  Returns:
    Scalar loss tensor
  """
  mag_loss_fn = FeatMag(margin=margin)

  # Bag loss
  if num_classes == 1:
    bag_loss = F.binary_cross_entropy_with_logits(
      bag_prediction.view(-1), target.view(-1), weight=pos_weight
    )
  else:
    # Multi-class: use cross-entropy
    if target.ndim == 1:
      target_class = target.long()
    else:
      target_class = target.argmax(dim=-1).long()
    if class_weights is not None:
      bag_loss = F.cross_entropy(bag_prediction, target_class, weight=class_weights)
    else:
      bag_loss = F.cross_entropy(bag_prediction, target_class)

  # Max instance loss
  max_prediction, _ = torch.max(instance_predictions, dim=0)  # Scalar or (num_classes,)
  if num_classes == 1:
    max_loss = F.binary_cross_entropy_with_logits(
      max_prediction.view(-1), target.view(-1), weight=pos_weight
    )
  else:
    # For multi-class, instance_predictions are attention scores, not class logits
    # Use bag prediction's max instead
    max_prediction = torch.max(bag_prediction, dim=-1)[0]
    if class_weights is not None:
      max_loss = F.cross_entropy(
        bag_prediction, target_class, weight=class_weights
      )  # Use bag prediction
    else:
      max_loss = F.cross_entropy(bag_prediction, target_class)

  # Feature magnitude loss (only if we have both normal and anomaly samples)
  loss_ft = torch.tensor(0.0, device=bag_prediction.device)
  if norm_idx is not None and ano_idx is not None and query_features.ndim == 3:
    # query_features shape: (batch, num_tiles, feat_dim)
    # We need to extract normal and anomaly features
    norm_idx_int = int(norm_idx)
    ano_idx_int = int(ano_idx)
    feat_norm = query_features[norm_idx_int : norm_idx_int + 1, :, :]  # (1, num_tiles, feat_dim)
    feat_ano = query_features[ano_idx_int : ano_idx_int + 1, :, :]  # (1, num_tiles, feat_dim)
    w_scale = float(query_features.shape[1])  # Number of tiles
    loss_ft = mag_loss_fn(feat_ano, feat_norm, w_scale=w_scale)

  loss = (bag_loss + max_loss + loss_ft) / 3.0
  return loss


def build_mil_criterion(
  cfg: Any,
  class_counts: dict[int, int] | None = None,
  device: str = "cuda",
) -> nn.Module:
  """Build the appropriate loss criterion for MIL training.

  This centralizes loss creation logic that was previously split between
  train_wsi.py and loop_mil.py.

  Args:
    cfg: Application configuration (AppCfg)
    class_counts: Optional dictionary mapping class index to count (for class weights)
    device: Device to place weights on

  Returns:
    Loss criterion module (CrossEntropyWithBrierLoss for standard MIL, or specific losses for DSMIL/FRMIL)
  """
  from ..config.schema import AppCfg

  cfg_typed = cast(AppCfg, cfg)

  # For DSMIL and FRMIL, we don't use the standard criterion - they have their own loss functions
  # But we still need a criterion for validation and fallback cases
  aggregator_type = cfg_typed.model.aggregator_type
  if aggregator_type in ("dsmil", "frmil"):
    # For DSMIL/FRMIL, use standard CrossEntropyLoss for validation/fallback
    # The actual training loss is computed by compute_dsmil_loss/compute_frmil_loss
    return nn.CrossEntropyLoss()

  # Standard MIL models (LongNet, GatedAttention, etc.) use CrossEntropyWithBrierLoss
  use_class_weights = getattr(cfg_typed.train, "use_class_weights", False)
  brier_weight = getattr(cfg_typed.train, "brier_weight", 0.1)

  if use_class_weights and class_counts is not None:
    num_classes = cfg_typed.dataset.task.num_classes # type: ignore [union-attr]
    total = sum(class_counts.values())
    weights = []
    for i in range(num_classes):
      count = class_counts.get(i, 0)
      if count == 0:
        count = 1  # Avoid division by zero
      weights.append(total / (num_classes * count))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    return CrossEntropyWithBrierLoss(weight=weight_tensor, brier_weight=brier_weight)
  else:
    return CrossEntropyWithBrierLoss(brier_weight=brier_weight)
