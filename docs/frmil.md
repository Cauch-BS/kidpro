# FRMIL: Feature Recalibration Multiple Instance Learning

FRMIL is a MIL aggregation method that uses feature recalibration, CNN position learning, and multi-head attention for bag-level classification.

## Architecture

FRMIL consists of several key components:

1. **Feature Recalibration**: Selects top-k instances based on encoder scores and shifts features
2. **CLS Token**: Learnable classification token for bag representation
3. **CNN Position Learning**: Uses depthwise convolution to learn positional relationships
4. **Multi-head Attention Block (MAB)**: Self-attention mechanism for bag pooling

### Key Features

- **Feature recalibration**: Identifies and emphasizes critical instances
- **Feature shifting**: Recalibrates features by subtracting query features: `F.relu(inputs - Q)`
- **Position learning**: CNN-based positional encoding for spatial relationships
- **Multi-head attention**: Flexible attention mechanism with configurable heads

## Configuration

FRMIL can be configured via the model config:

```yaml
model:
  aggregator_type: "frmil"
  foundation_dim: 1536  # Dimension of tile embeddings
  frmil_n_heads: 4  # Number of attention heads
  frmil_margin: 8.48  # Margin for feature magnitude loss
  frmil_k: 16  # Number of top instances for recalibration
```

### Parameters

- `frmil_n_heads` (default: 4): Number of attention heads in the MAB
- `frmil_margin` (default: 8.48): Margin parameter for feature magnitude loss
- `frmil_k` (default: 16): Number of top instances to select for recalibration query

**Note**: Feature shifting is always enabled. All features are shifted relative to the critical instance reference: `F.relu(inputs - Q)` where Q is the mean of top-k critical instances.

## Loss Function

FRMIL uses a combination of three loss components:

```python
loss = (bag_loss + max_loss + loss_ft) / 3
```

Where:

- `bag_loss`: Cross-entropy loss on bag-level prediction
- `max_loss`: Cross-entropy loss on maximum instance prediction
- `loss_ft`: Feature magnitude loss between normal and anomaly features

The feature magnitude loss encourages separation between normal and anomaly feature representations:

```python
loss_ft = ReLU(margin - (mag_ano - mag_norm)) / num_tiles
```

Where `mag_ano` and `mag_norm` are the mean feature magnitudes for anomaly and normal samples.

## Usage

### Basic Usage

```python
from kidpro.modeling.agg import build_mil
from kidpro.config.load import load_config

cfg = load_config("path/to/config.yaml")
cfg.model.aggregator_type = "frmil"

model = build_mil(cfg, tile_encoder, num_classes=2)
```

### With RankMix

When RankMix is enabled, the training loop automatically uses `compute_frmil_loss` which handles soft labels from mixed samples.

## Implementation Details

### Forward Pass

The model forward pass returns:

- `bag_prediction`: Shape `(1, num_classes)` - Bag-level prediction
- `query_features`: Shape `(num_tiles, feat_dim)` - Shifted/recalibrated features
- `instance_predictions`: Shape `(num_tiles,)` - Attention scores from encoder

### Feature Recalibration

1. Encode all instances to get attention scores: `A1 = enc(inputs)`
2. Sort instances by attention scores (descending)
3. Select top-k instances and compute query: `Q = mean(top_k_instances)`
4. Shift features: `inputs_shifted = F.relu(inputs - Q)` (for "cm16")

### Position Learning

1. Pad inputs to square grid: `H = ceil(sqrt(num_tiles))`
2. Add CLS token: `inputs = [CLS, padded_inputs]`
3. Reshape to 2D grid: `(batch, H*W, feat_dim) -> (batch, feat_dim, H, W)`
4. Apply depthwise convolution: `cnn_feat = conv_head(cnn_feat) + cnn_feat`
5. Flatten back: `(batch, feat_dim, H, W) -> (batch, H*W, feat_dim)`

### Bag Pooling

1. Use query Q as query for attention
2. Use processed features (with CLS token) as keys/values
3. Apply MAB to get bag representation
4. Classify bag representation to get final prediction

## Feature Magnitude Loss

The feature magnitude loss encourages the model to learn discriminative features:

- Normal samples should have lower feature magnitude
- Anomaly samples should have higher feature magnitude
- Margin enforces minimum separation: `mag_ano - mag_norm >= margin`

This loss is only computed when both normal and anomaly samples are present in the batch (not applicable for RankMix mixed samples).

## References

- Implementation based on reference-frmil.txt from the codebase
