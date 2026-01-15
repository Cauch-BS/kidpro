"""Attention-based MIL models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadFlashAttentionMIL(nn.Module):
  """
  Multi-head attention pooling using scaled_dot_product_attention (FlashAttention).

  Uses a learned query per head to attend over instance features and aggregates
  into a single bag representation.
  """

  def __init__(
    self,
    backbone: nn.Module,
    feat_dim: int,
    num_classes: int = 2,
    num_heads: int = 4,
    head_dim: int | None = None,
    dropout: float = 0.25,
  ):
    super().__init__()

    if num_heads <= 0:
      raise ValueError("num_heads must be > 0.")
    if head_dim is None:
      if feat_dim % num_heads != 0:
        raise ValueError("feat_dim must be divisible by num_heads when head_dim is None.")
      head_dim = feat_dim // num_heads
    if head_dim <= 0:
      raise ValueError("head_dim must be > 0.")

    self.backbone = backbone
    self.feat_dim = feat_dim
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.head_dim = head_dim

    self.key_proj = nn.Linear(feat_dim, num_heads * head_dim)
    self.value_proj = nn.Linear(feat_dim, num_heads * head_dim)
    self.query = nn.Parameter(torch.randn(num_heads, head_dim))
    self.out_proj = nn.Linear(num_heads * head_dim, feat_dim)

    self.classifier = nn.Sequential(
      nn.Linear(feat_dim, num_classes)
    )

    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def _compute_attention(
    self,
    keys: torch.Tensor,
    values: torch.Tensor,
    return_attention: bool,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Args:
      keys: (N, H, D)
      values: (N, H, D)
    """
    keys = keys.transpose(0, 1)  # (H, N, D)
    values = values.transpose(0, 1)  # (H, N, D)
    queries = self.query.unsqueeze(1)  # (H, 1, D)

    if return_attention:
      scale = float(self.head_dim) ** -0.5
      scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # (H, 1, N)
      attn = torch.softmax(scores, dim=-1)  # (H, 1, N)
      attended = torch.matmul(attn, values)  # (H, 1, D)
      attn_weights = attn.squeeze(1).mean(dim=0)  # (N,)
      return attended, attn_weights

    # FlashAttention path
    q = queries.unsqueeze(0)  # (1, H, 1, D)
    k = keys.unsqueeze(0)     # (1, H, N, D)
    v = values.unsqueeze(0)   # (1, H, N, D)
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    attended = attended.squeeze(0)  # (H, 1, D)
    return attended, None

  def forward(
    self,
    x: torch.Tensor,
    return_attention: bool = False
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # Extract features if input is images
    if x.dim() == 4:  # (N, C, H, W)
      H = self.backbone(x)  # (N, feat_dim)
    else:  # Already features (N, feat_dim)
      H = x

    H = self.dropout(H)  # (N, feat_dim)
    n_instances = H.size(0)

    keys = self.key_proj(H).view(n_instances, self.num_heads, self.head_dim)
    values = self.value_proj(H).view(n_instances, self.num_heads, self.head_dim)

    attended, attn_weights = self._compute_attention(
      keys,
      values,
      return_attention=return_attention,
    )

    pooled = attended.transpose(0, 1).contiguous().view(1, -1)  # (1, H*D)
    pooled = self.out_proj(pooled)  # (1, feat_dim)

    logits = self.classifier(pooled)

    if return_attention:
      return logits, attn_weights if attn_weights is not None else torch.empty(0)
    return logits  # type: ignore
