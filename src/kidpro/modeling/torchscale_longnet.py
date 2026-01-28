from __future__ import annotations

import re
from typing import Iterable

from torchscale.architecture.config import EncoderConfig
from torchscale.model.LongNet import LongNetEncoder

__all__ = ["LongNetEncoder", "make_longnet_from_name"]

_LONGNET_NAME_RE = re.compile(
  r"^LongNet_(?P<layers>\d+)_layers_(?P<dim>\d+)_dim(?:_mlp(?P<mlp>\d+(?:\.\d+)?))?$"
)


def _pick_num_heads(embed_dim: int) -> int:
  """
  Pick a reasonable number of heads for a given embedding dimension.

  Prefers 16/8 heads (historical TorchScale LongNet configs), but ensures divisibility.
  """
  preferred: Iterable[int] = (16, 8, 12, 6, 4, 2, 1)
  for h in preferred:
    if embed_dim % h == 0:
      head_dim = embed_dim // h
      if 32 <= head_dim <= 128:
        return h
  # Fallback: any divisor.
  for h in range(min(embed_dim, 32), 0, -1):
    if embed_dim % h == 0:
      return h
  return 1


def make_longnet_from_name(
  config_name: str,
  *,
  dilated_ratio: str = "[1, 2, 4, 8, 16]",
  segment_length: str = "[1024, 2048, 4096, 8192, 16384]",
  drop_path_rate: float = 0.1,
  dropout: float = 0.1,
) -> LongNetEncoder:
  """
  Construct a TorchScale LongNet encoder from the naming convention used in this repo.

  We depend on the TorchScale git repository (not PyPI), where `torchscale.model.LongNet`
  and `torchscale.component.dilated_attention` exist.
  """
  m = _LONGNET_NAME_RE.match(config_name)
  if not m:
    raise ValueError(
      f"Unrecognized LongNet config name: {config_name!r}. "
      "Expected e.g. 'LongNet_12_layers_1536_dim' or 'LongNet_12_layers_256_dim_mlp2'."
    )

  layers = int(m.group("layers"))
  dim = int(m.group("dim"))
  mlp_ratio = float(m.group("mlp") or 4.0)

  heads = _pick_num_heads(dim)
  ffn_dim = int(dim * mlp_ratio)

  cfg = EncoderConfig(
    encoder_layers=layers,
    encoder_embed_dim=dim,
    encoder_ffn_embed_dim=ffn_dim,
    encoder_attention_heads=heads,
    dropout=dropout,
    drop_path_rate=drop_path_rate,
    # LongNet / dilated attention knobs
    flash_attention=True,
    block_shift=True,
    segment_length=segment_length,
    dilated_ratio=dilated_ratio,
    # Keep MoE off for slide encoder use
    use_xmoe=False,
    moe_top1_expert=False,
    moe_freq=0,
    moe_expert_count=0,
    # This repo always passes token_embeddings; vocab_size is irrelevant but required by config.
    vocab_size=-1,
  )
  return LongNetEncoder(cfg)
