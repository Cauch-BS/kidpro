from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SlideEncoderBackbone:
    """Public return type for slide encoder builders."""
    encoder: nn.Module
    embed_dim: int


class MILTemplate(nn.Module, ABC):
    """
    Abstract base class for MIL (Multiple Instance Learning) models.

    All MIL architectures should inherit from this class and implement
    the required methods for encoding tiles, aggregating slide features,
    and classifying slide-level predictions.
    """

    @abstractmethod
    def encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tiles into feature embeddings.

        Args:
            x: Input tiles of shape (num_tiles, channels, height, width)

        Returns:
            Tile feature embeddings of shape (num_tiles, feat_dim)
        """
        pass

    @abstractmethod
    def encode_slide_embedding(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode tile features into a slide-level embedding.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates of shape (num_tiles, 2). Optional, depends on architecture.

        Returns:
            Slide embedding of shape (1, embed_dim)
        """
        pass

    @abstractmethod
    def classify_slide_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify slide embedding into logits.

        Args:
            embedding: Slide embedding of shape (1, embed_dim)

        Returns:
            Logits of shape (1, num_classes)
        """
        pass

    def encode_slide(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Encode tile features and classify in one step.

        Args:
            feats: Tile features of shape (num_tiles, feat_dim)
            coords: Tile coordinates of shape (num_tiles, 2). Optional, depends on architecture.

        Returns:
            Logits of shape (1, num_classes)
        """
        slide_out = self.encode_slide_embedding(feats, coords)
        return self.classify_slide_embedding(slide_out)

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass: encode tiles, aggregate, and classify.

        Args:
            x: Input tiles of shape (num_tiles, channels, height, width)
            coords: Tile coordinates of shape (num_tiles, 2). Optional, depends on architecture.

        Returns:
            Logits of shape (1, num_classes)
        """
        feats = self.encode_tiles(x)
        return self.encode_slide(feats, coords)
