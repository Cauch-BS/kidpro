# DSMIL: Dual-Stream Multiple Instance Learning

DSMIL is a MIL aggregation method that uses two streams: an instance classifier and a bag classifier with attention mechanism.

## Architecture

DSMIL consists of three main components:

1. **Instance Classifier (IClassifier)**: Processes tile embeddings and predicts class scores for each tile
2. **Bag Classifier (BClassifier)**: Aggregates instance predictions using attention mechanism with critical instance selection
3. **MILNet**: Combines both classifiers for end-to-end training

### Key Features

- **Dual-stream design**: Separate instance and bag-level predictions
- **Critical instance selection**: Selects top instances per class for attention computation
- **Attention mechanism**: Uses learned attention weights to aggregate instance features
- **Multi-class support**: Handles both binary and multi-class classification

## Configuration

DSMIL can be configured via the model config:

```yaml
model:
  aggregator_type: "dsmil"
  foundation_dim: 1536  # Dimension of tile embeddings
  dsmil_dropout_node: 0.0  # Dropout rate for bag classifier
```

### Parameters

- `dsmil_dropout_node` (default: 0.0): Dropout rate applied to the value network in the bag classifier

**Note**: The bag classifier always uses nonlinear transformation (ReLU + Tanh) for feature processing.

## Loss Function

DSMIL uses a combination of bag loss and max instance loss:

```python
loss = 0.5 * bag_loss + 0.5 * max_loss
```

Where:

- `bag_loss`: Cross-entropy loss on bag-level prediction
- `max_loss`: Cross-entropy loss on maximum instance prediction

This dual-loss approach ensures both bag-level and instance-level supervision.

## Usage

### Basic Usage

```python
from kidpro.modeling.agg import build_mil
from kidpro.config.load import load_config

cfg = load_config("path/to/config.yaml")
cfg.model.aggregator_type = "dsmil"

model = build_mil(cfg, tile_encoder, num_classes=2)
```

### With RankMix

When RankMix is enabled, the training loop automatically uses `compute_dsmil_loss` which handles soft labels from mixed samples.

## Implementation Details

### Forward Pass

The model forward pass returns:

- `instance_predictions`: Shape `(num_tiles, num_classes)` - Class scores for each tile
- `bag_prediction`: Shape `(1, num_classes)` - Bag-level prediction
- `attention_weights`: Shape `(num_tiles, num_classes)` - Attention weights per tile per class
- `bag_representation`: Shape `(num_classes, feat_dim)` - Bag representation per class

### Instance Selection

The bag classifier selects critical instances by:

1. Sorting instance predictions per class (descending)
2. Selecting the top instance for each class
3. Using these critical instances as queries for attention computation

### Attention Mechanism

Attention is computed as:

1. Compute queries Q from all instances
2. Compute queries q_max from critical instances (one per class)
3. Compute attention scores: `A = softmax(Q @ q_max^T / sqrt(dim))`
4. Aggregate features: `B = A^T @ V`
5. Apply 1D convolution to get bag prediction

## References

- Li, B., Li, Y., & Eliceiri, K. W. (2021). Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning. CVPR 2021.
