"""DSMIL: Dual-Stream Multiple Instance Learning.

Implementation based on:
  Li et al., "Dual-stream Multiple Instance Learning Network for Whole Slide Image
  Classification with Self-supervised Contrastive Learning", CVPR 2021.

DSMIL uses two streams:
  1. Instance classifier: Predicts class scores for each tile
  2. Bag classifier: Aggregates instance predictions using attention mechanism
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ...config.schema import AppCfg

from ._types import MILTemplate, SlideEncoderBackbone

log = logging.getLogger(__name__)

AGGREGATOR_NAME = "dsmil"


class FCLayer(nn.Module):
    """Fully connected layer for instance classification."""

    def __init__(self, in_size: int, out_size: int = 1) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: Input features of shape (N, in_size)

        Returns:
            Tuple of (feats, class_scores) where class_scores has shape (N, out_size)
        """
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    """Instance classifier that processes tile embeddings."""

    def __init__(self, feature_extractor: nn.Module, feature_size: int, output_class: int) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tiles of shape (N, channels, height, width) or embeddings (N, feature_size)

        Returns:
            Tuple of (feats, classes) where:
            - feats: Features of shape (N, feature_size)
            - classes: Class scores of shape (N, output_class)
        """
        device = x.device
        feats = self.feature_extractor(x)  # N x feature_size
        if feats.ndim > 2:
            feats = feats.view(feats.shape[0], -1)
        c = self.fc(feats)  # N x output_class
        return feats, c


class BClassifier(nn.Module):
    """Bag classifier using attention mechanism with critical instance selection."""

    def __init__(
        self,
        input_size: int,
        output_class: int,
        dropout_v: float = 0.0,
    ) -> None:
        """
        Args:
            input_size: Dimension of input features
            output_class: Number of output classes
            dropout_v: Dropout rate for value network
        """
        super().__init__()
        self.input_size = input_size
        self.output_class = output_class

        # Always use nonlinear transformation
        self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
        self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())

        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size),
        )

        # 1D convolutional layer that can handle multiple classes (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        self.distilled_bag_head: nn.Module | None = None

    def add_distilled_bag_head(self) -> None:
        """Add a distilled bag head for knowledge distillation."""
        self.distilled_bag_head = nn.Conv1d(self.output_class, self.output_class, kernel_size=self.input_size)

    def forward(
        self, feats: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: Instance features of shape (N, input_size)
            c: Instance class scores of shape (N, output_class)

        Returns:
            Tuple of (bag_prediction, attention_weights, bag_representation):
            - bag_prediction: Shape (1, output_class)
            - attention_weights: Shape (N, output_class)
            - bag_representation: Shape (output_class, input_size)
        """
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)  # N x input_size, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x 128, unsorted

        # Handle multiple classes without for loop
        # Sort class scores along the instance dimension, m_indices in shape N x output_class
        _, m_indices = torch.sort(c, dim=0, descending=True)
        # Select critical instances, m_feats in shape output_class x input_size
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # Compute queries of critical instances, q_max in shape output_class x 128
        q_max = self.q(m_feats)
        # Compute inner product of Q to each entry of q_max
        # A in shape N x output_class, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))

        # Normalize attention scores, A in shape N x output_class
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        # Compute bag representation, B in shape output_class x input_size
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x output_class x input_size
        C = self.fcc(B)  # 1 x output_class x 1
        C = C.view(1, -1)  # 1 x output_class

        return C, A, B


class DSMILSlideEncoder(nn.Module):
    """Wrapper module containing DSMIL's instance and bag classifiers.

    This module wraps both the instance classifier and bag classifier since they
    work together to aggregate tile embeddings into slide-level representations.
    Used for compatibility with training code that accesses model.slide_encoder
    for LoRA application and parameter grouping.
    """

    def __init__(self, i_classifier: nn.Module, b_classifier: nn.Module):
        super().__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier


class DSMILAggregator(nn.Module):
    """DSMIL aggregator module (for compatibility with SlideEncoderBackbone)."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        dropout_node: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Instance classifier
        self.i_classifier = IClassifier(
            feature_extractor=nn.Identity(), feature_size=in_dim, output_class=num_classes
        )

        # Bag classifier (always uses nonlinear transformation)
        self.b_classifier = BClassifier(
            input_size=in_dim, output_class=num_classes, dropout_v=dropout_node
        )

    def forward(
        self,
        x: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        all_layer_embed: bool = False,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """
        Args:
            x: Tile embeddings of shape (batch, num_tiles, in_dim) or (num_tiles, in_dim)
            coords: Tile coordinates (accepted but ignored)
            all_layer_embed: Whether to return all layer embeddings (ignored)
            **kwargs: Extra keyword arguments (ignored)

        Returns:
            List containing single bag embedding of shape (batch, embed_dim) or (1, embed_dim)
        """
        if x is None:
            x = kwargs.get("x", None)
        if x is None:
            raise TypeError("DSMILAggregator.forward requires x (either as arg or kwarg).")

        # Handle batch dimension
        if x.ndim == 2:
            # (num_tiles, in_dim) -> add batch dimension
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        results = []

        for b in range(batch_size):
            feats_b = x[b]  # (num_tiles, in_dim)
            feats, classes = self.i_classifier(feats_b)  # (num_tiles, in_dim), (num_tiles, num_classes)
            bag_pred, _, _ = self.b_classifier(feats, classes)  # (1, num_classes), ...

            # Use bag prediction as embedding (project to embed_dim if needed)
            if self.embed_dim != self.num_classes:
                # Simple projection if dimensions don't match
                if not hasattr(self, "embed_proj"):
                    self.embed_proj = nn.Linear(self.num_classes, self.embed_dim).to(bag_pred.device)
                bag_embed = self.embed_proj(bag_pred)  # (1, embed_dim)
            else:
                bag_embed = bag_pred  # (1, embed_dim)

            results.append(bag_embed)

        if squeeze_output:
            return [results[0]]
        return results


class DSMILMIL(MILTemplate):
    """DSMIL model implementing MILTemplate interface."""

    def __init__(
        self,
        tile_encoder: nn.Module,
        in_dim: int,
        num_classes: int,
        dropout_node: float = 0.25,
    ) -> None:
        """
        Args:
            tile_encoder: Tile encoder (foundation model)
            in_dim: Dimension of tile embeddings
            num_classes: Number of output classes
            dropout_node: Dropout rate for bag classifier
        """
        super().__init__()
        self.tile_encoder = tile_encoder
        self.in_dim = in_dim
        self.num_classes = num_classes

        # Instance classifier
        self.i_classifier = IClassifier(
            feature_extractor=nn.Identity(), feature_size=in_dim, output_class=num_classes
        )

        # Bag classifier (always uses nonlinear transformation)
        self.b_classifier = BClassifier(
            input_size=in_dim, output_class=num_classes, dropout_v=dropout_node
        )

        # Slide encoder wrapper (for compatibility with training code that accesses model.slide_encoder)
        # This wraps both i_classifier and b_classifier since they work together to aggregate tiles
        # into slide-level representations. The training code uses this for LoRA and parameter grouping.
        self.slide_encoder = DSMILSlideEncoder(self.i_classifier, self.b_classifier)

        # Final classifier (maps bag representation to logits)
        # Note: BClassifier already outputs class logits, but we keep this for consistency
        self.classifier = nn.Identity()  # BClassifier already outputs logits

    def encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tiles into feature embeddings."""
        return cast(torch.Tensor, self.tile_encoder(x))

    def encode_slide_embedding(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode tile features into a slide-level embedding.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates (ignored for DSMIL)

        Returns:
            Slide embedding of shape (1, num_classes)
        """
        # Get instance predictions
        feats_processed, classes = self.i_classifier(feats)  # (N, D), (N, C)

        # Get bag prediction
        bag_pred, _, _ = self.b_classifier(feats_processed, classes)  # (1, C), ...

        return bag_pred # type: ignore[no-any-return]

    def classify_slide_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify slide embedding into logits (already logits from BClassifier)."""
        return embedding

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass: encode tiles, aggregate, and classify."""
        feats = self.encode_tiles(x)
        return self.encode_slide(feats, coords)

    def tile_logits(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute per-tile logits using the instance classifier.

        This provides "instance predictions" for RankMix compatibility.

        Args:
            feats: Tile embeddings of shape (N, in_dim)
            coords: Tile coordinates (ignored)

        Returns:
            Logits of shape (N, num_classes)
        """
        _, classes = self.i_classifier(feats)
        return classes # type: ignore[no-any-return]

    def forward_with_attention(
        self, x: torch.Tensor, coords: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns intermediate outputs for loss computation.

        Returns:
            Tuple of (instance_predictions, bag_prediction, attention_weights, bag_representation)
        """
        feats = self.encode_tiles(x)
        feats_processed, classes = self.i_classifier(feats)
        bag_pred, attention, bag_repr = self.b_classifier(feats_processed, classes)
        return classes, bag_pred, attention, bag_repr


def build(cfg: "AppCfg") -> SlideEncoderBackbone:
    """Build a DSMILAggregator from config."""
    in_chans = int(getattr(cfg.model, "foundation_dim", 1536))
    dim = cfg.model.longnet_dim
    # Type narrowing for MIL task
    if cfg.dataset.task.type != "mil":
        raise ValueError("DSMIL requires MIL task type")
    num_classes = cfg.dataset.task.num_classes
    dropout_node = getattr(cfg.model, "dsmil_dropout_node", 0.0)

    encoder = DSMILAggregator(
        in_dim=in_chans,
        embed_dim=dim,
        num_classes=num_classes,
        dropout_node=dropout_node,
    )
    return SlideEncoderBackbone(encoder=encoder, embed_dim=dim)


def build_mil(cfg: "AppCfg", tile_encoder: nn.Module, num_classes: int) -> MILTemplate:
    """
    Build a complete DSMILMIL model from config.

    This returns a complete MILTemplate, not just a slide encoder.
    """
    in_chans = int(getattr(cfg.model, "foundation_dim", 1536))
    dropout_node = getattr(cfg.model, "dsmil_dropout_node", 0.0)

    model = DSMILMIL(
        tile_encoder=tile_encoder,
        in_dim=in_chans,
        num_classes=num_classes,
        dropout_node=dropout_node,
    )

    return model
