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


def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    """
    Normalize coordinates to [0, 1] range for positional embedding.

    Args:
        coords: Tile coordinates of shape (num_tiles, 2) as (x, y)

    Returns:
        Normalized coordinates of shape (num_tiles, 2) in [0, 1]
    """
    if coords.numel() == 0:
        return coords

    coords_min = coords.min(dim=0, keepdim=True)[0]
    coords_max = coords.max(dim=0, keepdim=True)[0]
    coords_range = coords_max - coords_min
    coords_range = torch.clamp(coords_range, min=1e-8)  # Avoid division by zero

    coords_norm = (coords - coords_min) / coords_range
    return coords_norm


class GatedAttentionSlideEncoder(nn.Module):
    """
    Multi-head gated attention mechanism with positional embeddings.
    Exposes attention components as a single module for compatibility with training code.
    """
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        num_heads: int = 16,
        pos_embed_dim: int = 64,
        use_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.use_pos_embed = use_pos_embed
        self.pos_embed_dim = pos_embed_dim if use_pos_embed else 0

        # Positional embedding: learnable embedding for normalized coordinates
        if use_pos_embed:
            # Use a small MLP to embed normalized coordinates
            self.pos_embed_mlp = nn.Sequential(
                nn.Linear(2, pos_embed_dim),  # 2D coordinates -> pos_embed_dim
                nn.ReLU(),
                nn.Linear(pos_embed_dim, pos_embed_dim),
            )
            # Project positional embeddings to match feat_dim for addition
            self.pos_proj = nn.Linear(pos_embed_dim, feat_dim)
        else:
            self.pos_embed_mlp = None # type: ignore[assignment]
            self.pos_proj = None # type: ignore[assignment]

        # Multi-head attention: V, U, and w projections for each head
        # V: value network (what information to extract)
        self.attention_V = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, self.head_dim),
                nn.Tanh()
            ) for _ in range(num_heads)
        ])
        # U: gate network (what to pay attention to)
        self.attention_U = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, self.head_dim),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        # w: attention weights projection
        self.attention_w = nn.ModuleList([
            nn.Linear(self.head_dim, 1) for _ in range(num_heads)
        ])

        # Output projection to combine multi-head outputs
        self.output_proj = nn.Linear(hidden_dim, feat_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply multi-head gated attention to tile features with positional embeddings.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates of shape (num_tiles, 2) as (x, y). Optional.

        Returns:
            Slide embedding of shape (1, feat_dim)
        """

        # Add positional embeddings if enabled and coords provided
        if self.use_pos_embed and coords is not None:
            coords_norm = _normalize_coords(coords)  # (num_tiles, 2) in [0, 1]
            pos_embed = self.pos_embed_mlp(coords_norm)  # (num_tiles, pos_embed_dim)
            pos_embed_proj = self.pos_proj(pos_embed)  # (num_tiles, feat_dim)
            feats = feats + pos_embed_proj  # Add positional information
        elif self.use_pos_embed and coords is None:
            log.warning("[GatedAttention] use_pos_embed=True but coords=None. Positional embeddings not applied.")

        # Apply dropout to features
        H = self.dropout(feats)  # (num_tiles, feat_dim)

        # Multi-head gated attention
        head_outputs = []
        for head_idx in range(self.num_heads):
            # Gated attention for this head
            A_V = self.attention_V[head_idx](H)  # (num_tiles, head_dim)
            A_U = self.attention_U[head_idx](H)  # (num_tiles, head_dim)
            A = self.attention_w[head_idx](A_V * A_U)  # (num_tiles, 1) - element-wise gate

            # Softmax over instances to get attention weights
            A = torch.softmax(A, dim=0)  # (num_tiles, 1)

            # Weighted sum of features for this head
            M_head = torch.sum(A * H, dim=0, keepdim=True)  # (1, feat_dim)
            head_outputs.append(M_head)

        # Concatenate multi-head outputs
        M_concat = torch.cat(head_outputs, dim=-1)  # (1, hidden_dim)

        # Project back to feat_dim
        M = self.output_proj(M_concat)  # (1, feat_dim)

        return M # type: ignore[no-any-return]


class GatedAttentionMIL(MILTemplate):
    """
    Multi-head Gated Attention MIL with positional embeddings (Ilse et al. 2018, enhanced)

    Architecture:
      1. Feature extraction (backbone) -> embeddings (N, D)
      2. Positional embedding (optional) -> add to features
      3. Multi-head attention mechanism -> attention weights (N, num_heads)
      4. Weighted aggregation -> bag representation (D,)
      5. Classification head -> logits (num_classes,)

    This model supports:
      - Multi-head attention (default: 16 heads)
      - Positional embeddings from tile coordinates
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
        num_heads: int = 16,
        pos_embed_dim: int = 64,
        use_pos_embed: bool = True,
    ) -> None:
        """
        Args:
          tile_encoder: Feature extractor (timm model with num_classes=0)
          feat_dim: Dimension of backbone features (e.g., 2048 for ResNet50)
          num_classes: Number of output classes
          hidden_dim: Hidden dimension for attention network (must be divisible by num_heads)
          dropout: Dropout rate
          num_heads: Number of attention heads (default: 16)
          pos_embed_dim: Dimension of positional embeddings (default: 64)
          use_pos_embed: Whether to use positional embeddings (default: True)
        """
        super().__init__()

        self.tile_encoder = tile_encoder
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_pos_embed = use_pos_embed

        # Create slide encoder module (for compatibility with training code)
        self.slide_encoder = GatedAttentionSlideEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            pos_embed_dim=pos_embed_dim,
            use_pos_embed=use_pos_embed,
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
        Encode tile features into a slide-level embedding using multi-head gated attention.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates of shape (num_tiles, 2) as (x, y). Required if use_pos_embed=True.

        Returns:
            Slide embedding of shape (1, feat_dim)
        """
        return self.slide_encoder(feats, coords) # type: ignore[no-any-return]

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
          coords: Tile coordinates of shape (num_tiles, 2) as (x, y). Required if use_pos_embed=True.

        Returns:
          logits: (1, num_classes) - slide-level predictions
        """
        # Encode tiles
        feats = self.encode_tiles(x)  # (num_tiles, feat_dim)

        # Encode slide embedding using multi-head gated attention with positional embeddings
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
          coords: Tile coordinates of shape (num_tiles, 2) as (x, y). Required if use_pos_embed=True.

        Returns:
          (logits, attention_weights):
            - logits: (1, num_classes) - slide-level predictions
            - attention_weights: (num_tiles, num_heads) - normalized attention per patch per head
        """
        # Encode tiles
        feats = self.encode_tiles(x)  # (num_tiles, feat_dim)
        num_tiles = feats.shape[0]

        # Add positional embeddings if enabled and coords provided
        if self.use_pos_embed and coords is not None:
            coords_norm = _normalize_coords(coords)  # (num_tiles, 2) in [0, 1]
            pos_embed = self.slide_encoder.pos_embed_mlp(coords_norm)  # (num_tiles, pos_embed_dim)
            pos_embed_proj = self.slide_encoder.pos_proj(pos_embed)  # (num_tiles, feat_dim)
            feats = feats + pos_embed_proj  # Add positional information

        # Apply dropout to features
        H = self.dropout(feats)  # (num_tiles, feat_dim)

        # Multi-head gated attention
        head_attention_weights = []
        head_outputs = []

        for head_idx in range(self.num_heads):
            # Gated attention for this head
            A_V = self.attention_V[head_idx](H)  # (num_tiles, head_dim)
            A_U = self.attention_U[head_idx](H)  # (num_tiles, head_dim)
            A = self.attention_w[head_idx](A_V * A_U)  # (num_tiles, 1) - element-wise gate

            # Softmax over instances to get attention weights
            A = torch.softmax(A, dim=0)  # (num_tiles, 1)
            head_attention_weights.append(A.squeeze(1))  # (num_tiles,)

            # Weighted sum of features for this head
            M_head = torch.sum(A * H, dim=0, keepdim=True)  # (1, feat_dim)
            head_outputs.append(M_head)

        # Concatenate multi-head outputs
        M_concat = torch.cat(head_outputs, dim=-1)  # (1, hidden_dim)

        # Project back to feat_dim
        M = self.slide_encoder.output_proj(M_concat)  # (1, feat_dim)

        # Stack attention weights from all heads
        attention_weights = torch.stack(head_attention_weights, dim=1)  # (num_tiles, num_heads)

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
    num_heads = getattr(cfg.model, "gated_attention_num_heads", 16)
    pos_embed_dim = getattr(cfg.model, "gated_attention_pos_embed_dim", 64)
    use_pos_embed = getattr(cfg.model, "gated_attention_use_pos_embed", True)

    model = GatedAttentionMIL(
        tile_encoder=tile_encoder,
        feat_dim=feat_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_heads=num_heads,
        pos_embed_dim=pos_embed_dim,
        use_pos_embed=use_pos_embed,
    )
    return model
