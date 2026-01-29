"""FRMIL: Feature Recalibration Multiple Instance Learning.

Implementation based on:
  Reference implementation from reference-frmil.txt

FRMIL uses feature recalibration and multi-head attention for bag-level classification.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ...config.schema import AppCfg

from ._types import MILTemplate, SlideEncoderBackbone

log = logging.getLogger(__name__)

AGGREGATOR_NAME = "frmil"


class MAB(nn.Module):
    """Multi-head Attention Block."""

    def __init__(self, dim_Q: int, dim_V: int, num_heads: int, ln: bool = False) -> None:
        """
        Args:
            dim_Q: Dimension of query
            dim_V: Dimension of value (should equal dim_Q for self-attention)
            num_heads: Number of attention heads
            ln: Whether to use layer normalization
        """
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)

        if ln:
            self.ln0: nn.Module | None = nn.LayerNorm(dim_V)
            self.ln1: nn.Module | None = nn.LayerNorm(dim_V)
        else:
            self.ln0 = None
            self.ln1 = None

        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, inst_mode: bool = False) -> torch.Tensor:
        """
        Args:
            Q: Query tensor of shape (batch, num_queries, dim_Q) or (batch, dim_Q)
            K: Key tensor of shape (batch, num_keys, dim_Q)
            inst_mode: If True, return all tokens. If False, return pooled (bag mode).

        Returns:
            If inst_mode: Output of shape (batch, num_queries, dim_V)
            If not inst_mode: Output of shape (batch, dim_V)
        """
        # Ensure inputs are 3D
        if Q.ndim == 1:
            Q = Q.unsqueeze(0).unsqueeze(1)  # (1, 1, dim_Q)
        elif Q.ndim == 2:
            Q = Q.unsqueeze(1)  # (batch, 1, dim_Q)
        elif Q.ndim == 4:
            if Q.shape[0] == 1:
                Q = Q.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D Q shape: {Q.shape}")

        if K.ndim == 1:
            K = K.unsqueeze(0).unsqueeze(1)  # (1, 1, dim_Q)
        elif K.ndim == 2:
            K = K.unsqueeze(1)  # (batch, 1, dim_Q)
        elif K.ndim == 4:
            if K.shape[0] == 1:
                K = K.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D K shape: {K.shape}")

        # Both must be 3D now
        if Q.ndim != 3 or K.ndim != 3:
            raise RuntimeError(f"Q and K must be 3D after shape handling. Q: {Q.shape}, K: {K.shape}")

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # Verify Q, K, V are still 3D after linear layers
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise RuntimeError(f"After linear layers, tensors must be 3D. Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        dim_split = self.dim_V // self.num_heads
        if dim_split == 0:
            raise RuntimeError(f"dim_split is 0: dim_V={self.dim_V}, num_heads={self.num_heads}")
        if Q.shape[2] % dim_split != 0:
            raise RuntimeError(f"Q.shape[2] ({Q.shape[2]}) must be divisible by dim_split ({dim_split})")

        # Split along dimension 2 and concatenate along dimension 0 for multi-head
        Q_split = Q.split(dim_split, 2)
        K_split = K.split(dim_split, 2)
        V_split = V.split(dim_split, 2)

        # Verify splits are 3D
        if any(t.ndim != 3 for t in Q_split) or any(t.ndim != 3 for t in K_split) or any(t.ndim != 3 for t in V_split):
            raise RuntimeError(f"Split tensors must be 3D. Q_split[0]: {Q_split[0].shape if Q_split else 'empty'}, K_split[0]: {K_split[0].shape if K_split else 'empty'}")

        Q_ = torch.cat(Q_split, 0)
        K_ = torch.cat(K_split, 0)
        V_ = torch.cat(V_split, 0)

        # Verify Q_, K_, V_ are 3D
        if Q_.ndim != 3 or K_.ndim != 3 or V_.ndim != 3:
            raise RuntimeError(f"After concatenation, tensors must be 3D. Q_: {Q_.shape}, K_: {K_.shape}, V_: {V_.shape}")

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.ln0 is not None:
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if self.ln1 is not None:
            O = self.ln1(O)

        if inst_mode:
            # [batch, num_queries, dim_V] --> [batch, num_queries, dim_V]
            return O
        else:
            # bag mode [batch, 1, dim_V] --> [batch, dim_V]
            return O.squeeze(1)


class FRMILSlideEncoder(nn.Module):
    """Wrapper module containing FRMIL's slide aggregation components.

    This module wraps the components that aggregate tile embeddings into slide-level
    representations: encoder for recalibration, CNN position learning, and self-attention
    block. The CLS token is kept as a direct parameter of FRMILMIL (not wrapped) since
    it's a nn.Parameter and is already registered with the parent module.

    Used for compatibility with training code that accesses model.slide_encoder for
    LoRA application and parameter grouping.
    """

    def __init__(
        self,
        enc: nn.Module,
        conv_head: nn.Module,
        selt_att: nn.Module,
    ):
        super().__init__()
        self.enc = enc
        self.conv_head = conv_head
        self.selt_att = selt_att


class FRMILAggregator(nn.Module):
    """FRMIL aggregator module (for compatibility with SlideEncoderBackbone)."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 1,
        k: int = 1,
    ) -> None:
        """
        Args:
            in_dim: Dimension of input features
            embed_dim: Dimension of output embedding
            num_classes: Number of output classes
            num_heads: Number of attention heads
            k: Number of top instances to select for recalibration
        """
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.k = k

        # Encoder for recalibration
        self.enc = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

        # CLS token
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, in_dim))
        nn.init.xavier_uniform_(self.cls_token)

        # CNN position learning
        self.conv_head = nn.Conv2d(in_dim, in_dim, 3, 1, 3 // 2, groups=in_dim)
        nn.init.xavier_uniform_(self.conv_head.weight)

        # Self-attention
        self.selt_att = MAB(in_dim, in_dim, num_heads)

        # Final classifier
        self.fc = nn.Sequential(nn.Linear(in_dim, num_classes))

        # Distilled heads (optional)
        self.distilled_score_head: nn.Module | None = None
        self.distilled_bag_head: nn.Module | None = None

        self.mode = 0

    def add_distilled_bag_head(self) -> None:
        """Add a distilled bag head for knowledge distillation."""
        self.distilled_bag_head = nn.Sequential(nn.Linear(self.in_dim, self.num_classes))

    def add_distilled_score_head(self) -> None:
        """Add a distilled score head for knowledge distillation."""
        self.distilled_score_head = nn.Sequential(nn.Linear(self.in_dim, 1), nn.Sigmoid())

    def recalib(self, inputs: torch.Tensor, option: str = "max") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recalibrate features by selecting top-k instances.

        Args:
            inputs: Input features of shape (batch, num_instances, in_dim)
            option: "max" to select top-k, "mean" to use mean

        Returns:
            Tuple of (attention_scores, query_features):
            - attention_scores: Shape (batch, num_instances)
            - query_features: Shape (batch, in_dim)
        """
        A1_list: list[torch.Tensor] = []
        Q_list: list[torch.Tensor] = []
        bs = inputs.shape[0]

        if option == "mean":
            Q_mean = torch.mean(inputs, dim=1, keepdim=True)
            A1_mean = self.enc(Q_mean.squeeze(1))
            return A1_mean, Q_mean.squeeze(1)

        for i in range(bs):
            a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)  # (num_instances, 1)
            _, m_indices = torch.sort(a1, 0, descending=True)

            feat_q = []
            num_instances = m_indices.shape[0]
            len_i = num_instances - 1
            # Ensure k doesn't exceed number of instances
            k_actual = min(self.k, num_instances)
            for i_q in range(k_actual):
                if option == "max":
                    # Extract scalar index and ensure it's a 1D tensor
                    idx = m_indices[i_q, 0] if m_indices.ndim == 2 else m_indices[i_q]
                    idx_tensor = idx.unsqueeze(0) if idx.ndim == 0 else idx
                    feats = torch.index_select(inputs[i], dim=0, index=idx_tensor)
                else:
                    idx = m_indices[len_i - i_q, 0] if m_indices.ndim == 2 else m_indices[len_i - i_q]
                    idx_tensor = idx.unsqueeze(0) if idx.ndim == 0 else idx
                    feats = torch.index_select(inputs[i], dim=0, index=idx_tensor)
                feat_q.append(feats)

            feats = torch.stack(feat_q)  # (k, in_dim)
            A1_list.append(a1.squeeze(1))  # (num_instances,)
            Q_list.append(feats.mean(0))  # (in_dim,)

        A1 = torch.stack(A1_list)  # (batch, num_instances)
        Q = torch.stack(Q_list)  # (batch, in_dim)
        return A1, Q

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
            List containing single bag embedding
        """
        if x is None:
            x = kwargs.get("x", None)
        if x is None:
            raise TypeError("FRMILAggregator.forward requires x (either as arg or kwarg).")

        # Handle batch dimension
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if self.mode == 1:
            # Used in feature magnitude analysis
            result = self.selt_att(x, x, True)
            if squeeze_output:
                return [result[0]]
            return [result]

        A1, Q = self.recalib(x, "max")

        # Shift features (always enabled)
        Q_expanded = Q.unsqueeze(1)  # (batch, 1, in_dim)
        x = F.relu(x - Q_expanded)
        i_shift = x

        # Pad inputs to square grid
        H = x.shape[1]  # Number of instances
        _H = int(np.ceil(np.sqrt(H)))
        _W = _H
        add_length = _H * _W - H
        if add_length > 0:
            x = torch.cat([x, x[:, :add_length, :]], dim=1)  # (batch, _H*_W, in_dim)

        # CLS token
        B = x.shape[0]
        # Ensure x is 3D before concatenation
        if x.ndim == 2:
            x = x.unsqueeze(0)
            B = 1
        elif x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D x shape: {x.shape}")

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 1+_H*_W, in_dim)

        # CNN Position Learning
        cls_token, feat_token = x[:, 0], x[:, 1:]  # (batch, in_dim), (batch, _H*_W, in_dim)
        # Get actual sequence length and recompute _H, _W to match exactly
        actual_seq_len = feat_token.shape[1]
        _H = int(np.ceil(np.sqrt(actual_seq_len)))
        _W = _H
        target_len = _H * _W

        # Pad or truncate to match target length exactly
        if actual_seq_len < target_len:
            pad_needed = target_len - actual_seq_len
            padding = feat_token[:, :pad_needed, :]
            feat_token = torch.cat([feat_token, padding], dim=1)
        elif actual_seq_len > target_len:
            feat_token = feat_token[:, :target_len, :]

        cnn_feat = feat_token.transpose(1, 2).view(B, self.in_dim, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x = cnn_feat.flatten(2).transpose(1, 2)  # (batch, _H*_W, in_dim)

        # Ensure cls_token is 2D and x is 3D before concatenation
        if cls_token.ndim == 1:
            cls_token = cls_token.unsqueeze(0)  # (1, in_dim)
        elif cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1) if cls_token.shape[1] == 1 else cls_token.view(cls_token.shape[0], -1)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D x shape: {x.shape}")

        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  # (batch, 1+_H*_W, in_dim)

        # Bag pooling with critical feature
        Q_expanded = Q.unsqueeze(1)  # (batch, 1, in_dim)
        bag = self.selt_att(Q_expanded, x)  # (batch, in_dim)
        out = self.fc(bag)  # (batch, num_classes)

        if squeeze_output:
            return [out[0]]
        return [out]


class FRMILMIL(MILTemplate):
    """FRMIL model implementing MILTemplate interface."""

    def __init__(
        self,
        tile_encoder: nn.Module,
        in_dim: int,
        num_classes: int,
        num_heads: int = 1,
        k: int = 1,
    ) -> None:
        """
        Args:
            tile_encoder: Tile encoder (foundation model)
            in_dim: Dimension of tile embeddings
            num_classes: Number of output classes
            num_heads: Number of attention heads
            k: Number of top instances for recalibration
        """
        super().__init__()
        self.tile_encoder = tile_encoder
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        # Encoder for recalibration
        self.enc = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

        # CLS token
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, in_dim))
        nn.init.xavier_uniform_(self.cls_token)

        # CNN position learning
        self.conv_head = nn.Conv2d(in_dim, in_dim, 3, 1, 3 // 2, groups=in_dim)
        nn.init.xavier_uniform_(self.conv_head.weight)

        # Self-attention
        self.selt_att = MAB(in_dim, in_dim, num_heads)

        # Slide encoder wrapper (for compatibility with training code that accesses model.slide_encoder)
        # This wraps the components that aggregate tiles into slide-level representations.
        # Note: cls_token is kept as a direct parameter (not wrapped) since it's a nn.Parameter.
        # The training code uses this for LoRA and parameter grouping.
        self.slide_encoder = FRMILSlideEncoder(
            enc=self.enc,
            conv_head=self.conv_head,
            selt_att=self.selt_att,
        )

        # Final classifier
        self.classifier = nn.Sequential(nn.Linear(in_dim, num_classes))

        # Distilled heads (optional)
        self.distilled_score_head: nn.Module | None = None
        self.distilled_bag_head: nn.Module | None = None

        self.mode = 0

    def add_distilled_bag_head(self) -> None:
        """Add a distilled bag head for knowledge distillation."""
        self.distilled_bag_head = nn.Sequential(nn.Linear(self.in_dim, self.num_classes))

    def add_distilled_score_head(self) -> None:
        """Add a distilled score head for knowledge distillation."""
        self.distilled_score_head = nn.Sequential(nn.Linear(self.in_dim, 1), nn.Sigmoid())

    def recalib(self, inputs: torch.Tensor, option: str = "max") -> tuple[torch.Tensor, torch.Tensor]:
        """Recalibrate features by selecting top-k instances."""
        A1_list: list[torch.Tensor] = []
        Q_list: list[torch.Tensor] = []
        bs = inputs.shape[0]

        if option == "mean":
            Q_mean = torch.mean(inputs, dim=1, keepdim=True)
            A1_mean = self.enc(Q_mean.squeeze(1))
            return A1_mean, Q_mean.squeeze(1)

        for i in range(bs):
            a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
            _, m_indices = torch.sort(a1, 0, descending=True)

            feat_q = []
            num_instances = m_indices.shape[0]
            len_i = num_instances - 1
            # Ensure k doesn't exceed number of instances
            k_actual = min(self.k, num_instances)
            for i_q in range(k_actual):
                if option == "max":
                    # Extract scalar index and ensure it's a 1D tensor
                    idx = m_indices[i_q, 0] if m_indices.ndim == 2 else m_indices[i_q]
                    idx_tensor = idx.unsqueeze(0) if idx.ndim == 0 else idx
                    feats = torch.index_select(inputs[i], dim=0, index=idx_tensor)
                else:
                    idx = m_indices[len_i - i_q, 0] if m_indices.ndim == 2 else m_indices[len_i - i_q]
                    idx_tensor = idx.unsqueeze(0) if idx.ndim == 0 else idx
                    feats = torch.index_select(inputs[i], dim=0, index=idx_tensor)
                feat_q.append(feats)

            feats = torch.stack(feat_q)
            A1_list.append(a1.squeeze(1))
            Q_list.append(feats.mean(0))

        A1 = torch.stack(A1_list)
        Q = torch.stack(Q_list)
        return A1, Q

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
            coords: Tile coordinates (ignored for FRMIL)

        Returns:
            Slide embedding of shape (1, num_classes)
        """
        # Add batch dimension
        feats_batch = feats.unsqueeze(0)  # (1, num_tiles, feat_dim)

        _, Q = self.recalib(feats_batch, "max")

        # Shift features (always enabled)
        Q_expanded = Q.unsqueeze(1)  # (1, 1, in_dim)
        feats_batch = F.relu(feats_batch - Q_expanded)
        i_shift = feats_batch

        # Pad inputs to square grid
        H = feats_batch.shape[1]
        _H = int(np.ceil(np.sqrt(H)))
        _W = _H
        add_length = _H * _W - H
        if add_length > 0:
            feats_batch = torch.cat([feats_batch, feats_batch[:, :add_length, :]], dim=1)

        # CLS token
        B = feats_batch.shape[0]
        # Ensure feats_batch is 3D before concatenation
        if feats_batch.ndim == 2:
            feats_batch = feats_batch.unsqueeze(0)
            B = 1
        elif feats_batch.ndim == 4:
            if feats_batch.shape[0] == 1:
                feats_batch = feats_batch.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D feats_batch shape: {feats_batch.shape}")

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feats_batch = torch.cat((cls_tokens, feats_batch), dim=1)

        # CNN Position Learning
        cls_token, feat_token = feats_batch[:, 0], feats_batch[:, 1:]
        # Get actual sequence length and recompute _H, _W to match exactly (same fix as forward_with_aux)
        actual_seq_len = feat_token.shape[1]
        _H = int(np.ceil(np.sqrt(actual_seq_len)))
        _W = _H
        target_len = _H * _W

        # Pad or truncate to match target length exactly
        if actual_seq_len < target_len:
            pad_needed = target_len - actual_seq_len
            padding = feat_token[:, :pad_needed, :]
            feat_token = torch.cat([feat_token, padding], dim=1)
        elif actual_seq_len > target_len:
            feat_token = feat_token[:, :target_len, :]

        cnn_feat = feat_token.transpose(1, 2).view(B, self.in_dim, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x = cnn_feat.flatten(2).transpose(1, 2)  # (batch, _H*_W, in_dim)

        # Ensure cls_token is 2D and x is 3D before concatenation
        if cls_token.ndim == 1:
            cls_token = cls_token.unsqueeze(0)  # (1, in_dim)
        elif cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1) if cls_token.shape[1] == 1 else cls_token.view(cls_token.shape[0], -1)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D x shape: {x.shape}")

        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  # (batch, 1+_H*_W, in_dim)

        # Bag pooling with critical feature
        Q_expanded = Q.unsqueeze(1)
        bag = self.selt_att(Q_expanded, x)  # (1, in_dim)
        out = self.classifier(bag)  # (1, num_classes)

        return out # type: ignore[no-any-return]

    def classify_slide_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify slide embedding into logits (already logits from classifier)."""
        return embedding

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass: encode tiles, aggregate, and classify."""
        feats = self.encode_tiles(x)
        return self.encode_slide(feats, coords)

    def tile_logits(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute per-tile logits using the encoder network.

        This provides "instance predictions" for RankMix compatibility.

        Args:
            feats: Tile embeddings of shape (N, in_dim)
            coords: Tile coordinates (ignored)

        Returns:
            Logits of shape (N, 1) - single score per tile
        """
        # Use encoder to get instance scores
        scores = self.enc(feats)  # (N, 1)
        return scores # type: ignore[no-any-return]

    def forward_with_aux(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns auxiliary outputs for loss computation.

        Args:
            feats: Tile embeddings of shape (num_tiles, in_dim) - already encoded features
            coords: Tile coordinates (ignored for FRMIL)

        Returns:
            Tuple of (bag_prediction, query_features, instance_predictions):
            - bag_prediction: Shape (1, num_classes)
            - query_features: Shape (num_tiles, in_dim) - shifted features
            - instance_predictions: Shape (num_tiles,) - attention scores
        """
        # Follow the exact same pattern as encode_slide_embedding
        # Add batch dimension
        feats_batch = feats.unsqueeze(0)  # (1, num_tiles, in_dim)

        A1, Q = self.recalib(feats_batch, "max")

        # Shift features (always enabled) - store original for return
        Q_expanded = Q.unsqueeze(1)  # (1, 1, in_dim)
        i_shift = F.relu(feats_batch - Q_expanded)  # (1, num_tiles, in_dim)

        # Pad inputs to square grid (modify feats_batch in place like encode_slide_embedding)
        feats_batch = i_shift  # Work with shifted features
        H = feats_batch.shape[1]
        _H = int(np.ceil(np.sqrt(H)))
        _W = _H
        add_length = _H * _W - H
        if add_length > 0:
            feats_batch = torch.cat([feats_batch, feats_batch[:, :add_length, :]], dim=1)

        # CLS token
        B = feats_batch.shape[0]
        if feats_batch.ndim == 2:
            feats_batch = feats_batch.unsqueeze(0)
        elif feats_batch.ndim == 4:
            if B == 1:
                feats_batch = feats_batch.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D feats_batch shape: {feats_batch.shape}")

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feats_batch = torch.cat((cls_tokens, feats_batch), dim=1)

        # CNN Position Learning
        cls_token, feat_token = feats_batch[:, 0], feats_batch[:, 1:]
        # Get actual sequence length and recompute _H, _W to match exactly
        actual_seq_len = feat_token.shape[1]
        _H = int(np.ceil(np.sqrt(actual_seq_len)))
        _W = _H
        target_len = _H * _W

        # Pad or truncate to match target length exactly
        if actual_seq_len < target_len:
            pad_needed = target_len - actual_seq_len
            padding = feat_token[:, :pad_needed, :]
            feat_token = torch.cat([feat_token, padding], dim=1)
        elif actual_seq_len > target_len:
            feat_token = feat_token[:, :target_len, :]

        cnn_feat = feat_token.transpose(1, 2).view(B, self.in_dim, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x = cnn_feat.flatten(2).transpose(1, 2)  # (batch, _H*_W, in_dim)

        # Ensure cls_token is 2D and x is 3D before concatenation
        if cls_token.ndim == 1:
            cls_token = cls_token.unsqueeze(0)  # (1, in_dim)
        elif cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1) if cls_token.shape[1] == 1 else cls_token.view(cls_token.shape[0], -1)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D x shape: {x.shape}")

        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  # (batch, 1+_H*_W, in_dim)

        # Bag pooling with critical feature
        # Q should be (batch, in_dim) from recalib, ensure it's 2D
        if Q.ndim == 1:
            Q = Q.unsqueeze(0)  # (1, in_dim)
        elif Q.ndim == 3:
            # If somehow 3D, squeeze middle dimension
            Q = Q.squeeze(1) if Q.shape[1] == 1 else Q.view(Q.shape[0], -1)
        # x should be (batch, seq_len, in_dim) - ensure it's 3D
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, seq_len, in_dim)
        elif x.ndim == 4:
            if x.shape[0] == 1:
                x = x.squeeze(0)  # Remove first dimension if batch=1
            else:
                raise RuntimeError(f"Unexpected 4D x shape: {x.shape}")

        # Now create Q_expanded: (batch, 1, in_dim)
        Q_expanded = Q.unsqueeze(1)

        # Final verification: both must be 3D
        if Q_expanded.ndim != 3:
            raise RuntimeError(f"Q_expanded must be 3D, got shape {Q_expanded.shape}, Q shape was {Q.shape}")
        if x.ndim != 3:
            raise RuntimeError(f"x must be 3D, got shape {x.shape}")

        bag = self.selt_att(Q_expanded, x)  # (1, in_dim)
        bag_pred = self.classifier(bag)  # (1, num_classes)

        # Return shifted features (original, before padding) for loss computation
        return bag_pred, i_shift.squeeze(0), A1.squeeze(0)


def build(cfg: "AppCfg") -> SlideEncoderBackbone:
    """Build a FRMILAggregator from config."""
    in_chans = int(getattr(cfg.model, "foundation_dim", 1536))
    dim = cfg.model.longnet_dim
    # Type narrowing for MIL task
    if cfg.dataset.task.type != "mil":
        raise ValueError("FRMIL requires MIL task type")
    num_classes = cfg.dataset.task.num_classes
    num_heads = getattr(cfg.model, "frmil_n_heads", 1)
    k = getattr(cfg.model, "frmil_k", 1)

    encoder = FRMILAggregator(
        in_dim=in_chans,
        embed_dim=dim,
        num_classes=num_classes,
        num_heads=num_heads,
        k=k,
    )
    return SlideEncoderBackbone(encoder=encoder, embed_dim=dim)


def build_mil(cfg: "AppCfg", tile_encoder: nn.Module, num_classes: int) -> MILTemplate:
    """
    Build a complete FRMILMIL model from config.

    This returns a complete MILTemplate, not just a slide encoder.
    """
    in_chans = int(getattr(cfg.model, "foundation_dim", 1536))
    num_heads = getattr(cfg.model, "frmil_n_heads", 1)
    k = getattr(cfg.model, "frmil_k", 1)

    model = FRMILMIL(
        tile_encoder=tile_encoder,
        in_dim=in_chans,
        num_classes=num_classes,
        num_heads=num_heads,
        k=k,
    )

    return model
