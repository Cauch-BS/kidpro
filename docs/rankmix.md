# RankMix Data Augmentation for Class Imbalance

RankMix is an optional data augmentation technique for MIL training that addresses
class imbalance by mixing ranked tile embeddings from pairs of WSIs.

**Reference**: Chen & Lu, "RankMix: Data Augmentation for Weakly Supervised Learning of
Classifying Whole Slide Images with Diverse Sizes and Imbalanced Categories", CVPR 2023.

## When to Use RankMix

Use RankMix when:

- You have severe class imbalance (e.g., GT=True is a rare minority class)
- Balanced sampling alone is insufficient to address the imbalance
- You want to augment minority class representation through synthetic samples

## Quick Start: Two-Stage Training

RankMix requires two explicit training runs:

### Stage 1: Train Base MIL Model

```bash
# Run standard MIL training (no RankMix)
python -m kidpro.train_wsi train.rankmix.enabled=false

# This saves best_model.pt to the run directory, e.g.:
# outputs/2024-01-15/10-30-00/best_model.pt
```

### Stage 2: Train with RankMix Augmentation

```bash
# Run RankMix training, loading Stage 1 checkpoint
python -m kidpro.train_wsi \
  train.rankmix.enabled=true \
  train.rankmix.stage1_checkpoint=/path/to/stage1/best_model.pt
```

Or in a YAML config override:

```yaml
train:
  rankmix:
    enabled: true
    alpha: 1.0
    minority_sampling_ratio: 0.7
    stage1_checkpoint: /path/to/stage1/best_model.pt
```

## How It Works

### Architecture Overview

RankMix operates on **tile embeddings** (the output of the frozen tile encoder) before
they are passed to the LongNet aggregator:

```text
Without RankMix (Stage 1 / default):
  Slide A tiles → Tile Encoder → Embeddings A → LongNet → Prediction

With RankMix (Stage 2):
  Slide A tiles → Tile Encoder → Embeddings A ─┐
                                                ├→ RankMix → Mixed Embeddings → LongNet → Prediction
  Slide B tiles → Tile Encoder → Embeddings B ─┘
```

**Key insight**: LongNet is always the model being trained. RankMix just changes
what data (embeddings) LongNet sees during training.

### Two-Stage Training (Explicit Runs)

**Stage 1** (separate run with `rankmix.enabled=false`):

- Standard MIL training with LongNet
- Model learns slide classification
- Saves `best_model.pt` checkpoint

**Stage 2** (separate run with `rankmix.enabled=true`):

- Loads Stage 1 checkpoint
- Initializes TileScorer for ranking tiles
- For each training slide, samples a partner slide (biased toward minority class)
- Scores tiles by predicted importance using TileScorer
- Selects top-k tiles from each slide (k = min(tiles_A, tiles_B))
- Mixes embeddings: `H_mix = λ*H_A + (1-λ)*H_B` where λ ~ Beta(α, α)
- Mixes labels: `Y_mix = λ*Y_A + (1-λ)*Y_B` (soft label)
- Trains with soft cross-entropy loss

### Minority Oversampling

The key to addressing class imbalance is the `minority_sampling_ratio` parameter:

- With `minority_sampling_ratio=0.7`, approximately 70% of mixed pairs include at least one minority class slide
- This dramatically increases minority class representation without literal oversampling
- Each minority slide participates in many mixed samples with different partners and λ values

### Example: Effective Augmentation

Consider 100 slides with 10 minority (GT=True) and 90 majority (GT=False):

| Metric | Without RankMix | With RankMix |
| --- | --- | --- |
| Iterations per epoch | 100 | 100 |
| Minority in training samples | 10% (natural) | ~70% (via pairing) |
| Unique samples possible | 100 | Combinatorially large |

## Configuration Options

| Parameter | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Enable RankMix augmentation (Stage 2 training) |
| `alpha` | `1.0` | Beta distribution parameter for λ; higher = more uniform mixing ratios |
| `minority_sampling_ratio` | `0.7` | Probability of sampling minority class in pairs |
| `stage1_checkpoint` | `null` | **Required when enabled=true**. Path to Stage 1 `best_model.pt` |

### Parameter Tuning Tips

- **alpha**:
  - `alpha=1.0` gives uniform distribution over λ ∈ [0, 1]
  - `alpha < 1.0` biases toward extreme values (0 or 1)
  - `alpha > 1.0` biases toward 0.5 (equal mixing)
  
- **minority_sampling_ratio**:
  - Higher values (0.7-0.9) for severe imbalance
  - Lower values (0.5-0.6) for moderate imbalance

- **Stage 1 training**:
  - Train until convergence (early stopping will handle this)
  - The Stage 1 model provides the foundation for Stage 2

## Default Behavior (No RankMix)

When `rankmix.enabled: false` (the default), training uses standard LongNet MIL
without any mixing augmentation. This preserves the original training behavior
and is the recommended starting point.

## Logging and Monitoring

When RankMix is enabled (Stage 2), you'll see logging like:

```text
[RANKMIX] Loading Stage 1 checkpoint: /path/to/stage1/best_model.pt
[RANKMIX] Stage 1 model loaded successfully
[RANKMIX] TileScorer initialized with 1180417 params @ lr=3.00e-05
[RANKMIX] Stage 2 Training: alpha=1.00, minority_ratio=0.70
Epoch 1/100 | [RANKMIX STATS] samples=87 avg_lambda=0.512
```

The `avg_lambda` indicates the average mixing ratio across samples in the epoch.
Values around 0.5 indicate balanced mixing.

## Implementation Details

### Files Modified/Created

- `src/kidpro/training/rankmix.py` - Core RankMix implementation
  - `TileScorer`: MLP for predicting tile importance
  - `rank_and_select`: Top-k selection with order preservation
  - `rankmix`: Embedding and label mixing
  - `RankMixSampler`: Minority-biased partner selection
  - `compute_rankmix_loss`: Soft cross-entropy via PyTorch BCE

- `src/kidpro/training/loop_mil.py` - Training loop integration
  - `_get_tile_embeddings`: Extracts embeddings without running aggregator
  - `fit_mil`: Extended with RankMix parameters and two-stage logic

- `src/kidpro/train_wsi.py` - Entry point initialization
  - Creates TileScorer and RankMixSampler when enabled
  - Adds scorer parameters to optimizer

- `src/kidpro/config/schema.py` - Configuration schema
  - `RankMixCfg`: Pydantic model for RankMix settings

- `src/kidpro/conf/train/default_wsi.yaml` - Default configuration
  - RankMix config block (disabled by default)

### Soft Labels

RankMix produces soft labels (e.g., `Y_mix=0.7` instead of hard 0/1). These are
handled using PyTorch's `binary_cross_entropy_with_logits`, which natively supports
continuous target values in [0, 1].

## Troubleshooting

### RankMix not starting

- Check that `rankmix.enabled: true` is set
- Verify Stage 1 has completed (check `stage1_epochs`)
- Look for "[RANKMIX] Starting Stage 2" in logs

### Performance not improving

- Ensure sufficient Stage 1 training for TileScorer to learn
- Try adjusting `minority_sampling_ratio` (higher for severe imbalance)
- Consider increasing `alpha` for more balanced mixing

### Memory issues

- RankMix requires loading two slides per iteration
- Reduce `batch_size` if needed
- Ensure embedding cache is enabled for efficiency

## References

- Chen, Y.-C., & Lu, C.-S. (2023). RankMix: Data Augmentation for Weakly Supervised
  Learning of Classifying Whole Slide Images with Diverse Sizes and Imbalanced
  Categories. CVPR 2023.
