"""Attention-based MIL models."""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
  """
  Gated Attention MIL (Ilse et al. 2018)

  Architecture:
    1. Feature extraction (backbone) -> embeddings (N, D)
    2. Attention mechanism -> attention weights (N,)
    3. Weighted aggregation -> bag representation (D,)
    4. Classification head -> logits (num_classes,)

  This model supports:
    - Interpretability: exports attention weights per patch
    - End-to-end training: backbone can be frozen or fine-tuned
  """

  def __init__(
    self,
    backbone: nn.Module,
    feat_dim: int,
    num_classes: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.25,
  ):
    """
    Args:
      backbone: Feature extractor (timm model with num_classes=0)
      feat_dim: Dimension of backbone features (e.g., 2048 for ResNet50)
      num_classes: Number of output classes
      hidden_dim: Hidden dimension for attention network
      dropout: Dropout rate
    """
    super().__init__()

    self.backbone = backbone
    self.feat_dim = feat_dim
    self.num_classes = num_classes

    # Gated attention mechanism
    # V: value network (what information to extract)
    self.attention_V = nn.Sequential(
      nn.Linear(feat_dim, hidden_dim),
      nn.Tanh()
    )

    # U: gate network (what to pay attention to)
    self.attention_U = nn.Sequential(
      nn.Linear(feat_dim, hidden_dim),
      nn.Sigmoid()
    )

    # w: attention weights projection
    self.attention_w = nn.Linear(hidden_dim, 1)

    # Classifier head
    self.classifier = nn.Sequential(
      nn.Linear(feat_dim, num_classes)
    )

    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(
    self,
    x: torch.Tensor,
    return_attention: bool = False
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass.

    Args:
      x: Input tensor, shape (N, C, H, W) for images or (N, D) for features
      return_attention: If True, also return attention weights

    Returns:
      If return_attention=False:
        logits: (1, num_classes) - slide-level predictions
      If return_attention=True:
        (logits, attention_weights):
          - logits: (1, num_classes)
          - attention_weights: (N,) - normalized attention per patch
    """
    # Extract features if input is images
    if x.dim() == 4:  # (N, C, H, W)
      H = self.backbone(x)  # (N, feat_dim)
    else:  # Already features (N, feat_dim)
      H = x

    # Apply dropout to features
    H = self.dropout(H)  # (N, feat_dim)

    # Gated attention
    A_V = self.attention_V(H)  # (N, hidden_dim)
    A_U = self.attention_U(H)  # (N, hidden_dim)
    A = self.attention_w(A_V * A_U)  # (N, 1) - element-wise gate

    # Softmax over instances to get attention weights
    A = torch.softmax(A, dim=0)  # (N, 1)
    attention_weights = A.squeeze(1)  # (N,)

    # Weighted sum of features
    M = torch.sum(A * H, dim=0, keepdim=True)  # (1, feat_dim)

    # Classification
    logits = self.classifier(M)  # (1, num_classes)

    if return_attention:
      return logits, attention_weights
    return logits


class AttentionMIL(nn.Module):
  """
  Simple (non-gated) Attention MIL.

  Simpler variant without gating mechanism.
  Generally performs slightly worse than gated version.
  """

  def __init__(
    self,
    backbone: nn.Module,
    feat_dim: int,
    num_classes: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.25,
  ):
    super().__init__()

    self.backbone = backbone
    self.feat_dim = feat_dim
    self.num_classes = num_classes

    # Simple attention
    self.attention = nn.Sequential(
      nn.Linear(feat_dim, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, 1)
    )

    self.classifier = nn.Sequential(
      nn.Linear(feat_dim, num_classes)
    )

    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(
    self,
    x: torch.Tensor,
    return_attention: bool = False
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward pass (same interface as GatedAttentionMIL)."""
    # Extract features if input is images
    if x.dim() == 4:  # (N, C, H, W)
      H = self.backbone(x)  # (N, feat_dim)
    else:  # Already features
      H = x

    H = self.dropout(H)

    # Attention weights
    A = self.attention(H)  # (N, 1)
    A = torch.softmax(A, dim=0)
    attention_weights = A.squeeze(1)  # (N,)

    # Weighted aggregation
    M = torch.sum(A * H, dim=0, keepdim=True)  # (1, feat_dim)

    # Classification
    logits = self.classifier(M)  # (1, num_classes)

    if return_attention:
      return logits, attention_weights
    return logits
