from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...config.schema import AppCfg

from ._types import MILTemplate

log = logging.getLogger(__name__)

AGGREGATOR_NAME = "gated_attention"


class GatedAttentionSlideEncoder(nn.Module):
    """
    Wrapper module for gated attention mechanism.
    Exposes attention components as a single module for compatibility with training code.
    """
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Apply gated attention to tile features.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)

        Returns:
            Slide embedding of shape (1, feat_dim)
        """
        # Apply dropout to features
        H = self.dropout(feats)  # (num_tiles, feat_dim)

        # Gated attention
        A_V = self.attention_V(H)  # (num_tiles, hidden_dim)
        A_U = self.attention_U(H)  # (num_tiles, hidden_dim)
        A = self.attention_w(A_V * A_U)  # (num_tiles, 1) - element-wise gate

        # Softmax over instances to get attention weights
        A = torch.softmax(A, dim=0)  # (num_tiles, 1)

        # Weighted sum of features
        M = torch.sum(A * H, dim=0, keepdim=True)  # (1, feat_dim)

        return M


class GatedAttentionMIL(MILTemplate):
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
        tile_encoder: nn.Module,
        feat_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        """
        Args:
          tile_encoder: Feature extractor (timm model with num_classes=0)
          feat_dim: Dimension of backbone features (e.g., 2048 for ResNet50)
          num_classes: Number of output classes
          hidden_dim: Hidden dimension for attention network
          dropout: Dropout rate
        """
        super().__init__()

        self.tile_encoder = tile_encoder
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Create slide encoder module (for compatibility with training code)
        self.slide_encoder = GatedAttentionSlideEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Expose attention components for direct access (used in forward methods)
        self.attention_V = self.slide_encoder.attention_V
        self.attention_U = self.slide_encoder.attention_U
        self.attention_w = self.slide_encoder.attention_w
        self.dropout = self.slide_encoder.dropout

        # Classifier head
        self.classifier = nn.Linear(feat_dim, num_classes)

    def encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tiles into feature embeddings.

        Args:
            x: Input tiles of shape (num_tiles, channels, height, width)

        Returns:
            Tile feature embeddings of shape (num_tiles, feat_dim)
        """
        return cast(torch.Tensor, self.tile_encoder(x))

    def encode_slide_embedding(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode tile features into a slide-level embedding using gated attention.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates (accepted but ignored for gated attention)

        Returns:
            Slide embedding of shape (1, feat_dim)
        """
        return self.slide_encoder(feats) # type: ignore[no-any-return]

    def classify_slide_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify slide embedding into logits.

        Args:
            embedding: Slide embedding of shape (1, feat_dim)

        Returns:
            Logits of shape (1, num_classes)
        """
        return cast(torch.Tensor, self.classifier(embedding))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass: encode tiles, aggregate, and classify.

        Args:
          x: Input tiles of shape (num_tiles, channels, height, width)
          coords: Tile coordinates (accepted but ignored)

        Returns:
          logits: (1, num_classes) - slide-level predictions
        """
        # Encode tiles
        feats = self.encode_tiles(x)  # (num_tiles, feat_dim)

        # Encode slide embedding using gated attention
        slide_embedding = self.encode_slide_embedding(feats, coords)  # (1, feat_dim)

        # Classification
        logits = self.classifier(slide_embedding)  # (1, num_classes)

        return logits # type: ignore[no-any-return]

    def forward_with_attention(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights for interpretability.

        Args:
          x: Input tiles of shape (num_tiles, channels, height, width)
          coords: Tile coordinates (accepted but ignored)

        Returns:
          (logits, attention_weights):
            - logits: (1, num_classes) - slide-level predictions
            - attention_weights: (num_tiles,) - normalized attention per patch
        """
        # Encode tiles
        feats = self.encode_tiles(x)  # (num_tiles, feat_dim)

        # Apply dropout to features
        H = self.dropout(feats)  # (num_tiles, feat_dim)

        # Gated attention
        A_V = self.attention_V(H)  # (num_tiles, hidden_dim)
        A_U = self.attention_U(H)  # (num_tiles, hidden_dim)
        A = self.attention_w(A_V * A_U)  # (num_tiles, 1) - element-wise gate

        # Softmax over instances to get attention weights
        A = torch.softmax(A, dim=0)  # (num_tiles, 1)
        attention_weights = A.squeeze(1)  # (num_tiles,)

        # Weighted sum of features
        M = torch.sum(A * H, dim=0, keepdim=True)  # (1, feat_dim)

        # Classification
        logits = self.classifier(M)  # (1, num_classes)

        return logits, attention_weights


def build_mil(cfg: "AppCfg", tile_encoder: nn.Module, num_classes: int) -> MILTemplate:
    """
    Build a complete GatedAttentionMIL model from config.

    This returns a complete MILTemplate, not just a slide encoder.
    """
    feat_dim = int(getattr(cfg.model, "foundation_dim", 1536))
    hidden_dim = getattr(cfg.model, "gated_attention_hidden_dim", 128)
    dropout = getattr(cfg.model, "gated_attention_dropout", 0.25)

    model = GatedAttentionMIL(
        tile_encoder=tile_encoder,
        feat_dim=feat_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    return model
