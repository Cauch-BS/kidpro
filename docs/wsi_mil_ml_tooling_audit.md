# WSI/MIL ML Tooling Audit (kidpro)

Scope: **WSI/MIL (slide-level)** training + inference.

Primary entrypoints:

- Training: `src/kidpro/train_wsi.py`
- Inference: `src/kidpro/infer_ensem.py`

Key config bundles:

- `src/kidpro/conf/config_wsi.yaml`
- `src/kidpro/conf/dataset/wsi.yaml`
- `src/kidpro/conf/train/default_wsi.yaml`
- `src/kidpro/conf/model/prov_gigapath.yaml`

## Summary (high-signal)

- **Imbalance handling**: you had both balanced sampling and class-weighted loss enabled by default; this is “double compensation” and can bias gradients toward the minority class. The code now enforces **use exactly one**.
- **Backbone preprocessing**: Prov-GigaPath’s published preprocessing uses **ImageNet mean/std** normalization (and 256→224 resize/crop). Your MIL transforms previously used `albumentations.Normalize()` defaults ([-1, 1]) which can be a mismatch.
- **Embedding caching vs augmentation**: caching tile-encoder embeddings across epochs/runs is only theoretically consistent if the tile images (and their augmentations) are deterministic; stochastic flips + embedding cache effectively “lock in” one random augmentation per slide.
- **Model selection**: MIL checkpointing should select “best” using the same metric you early-stop on (AUC vs macro-F1 vs loss), otherwise you can stop on one objective but save weights optimized for another.
- **LoRA on backbone in MIL**: applying LoRA and then freezing everything yields no trainable LoRA parameters (confusing at best). In MIL, the intended PEFT path is LoRA on the **slide encoder**, not the frozen tile encoder.

## Techniques used, assumptions, and justification

### 1) MIL problem formulation (slide-level classification)

- **Where**: `src/kidpro/training/loop_mil.py`, `src/kidpro/train_wsi.py`
- **What**: Treat each slide as one training example; stream all tiles through a frozen tile encoder; train a slide encoder + classifier with `CrossEntropyLoss`.
- **Assumptions**:
  - Slide label is a valid supervisory signal for the distribution of tiles.
  - Tile encoder embeddings are sufficiently informative and transferable.
- **Justification**: This is standard MIL: you learn an aggregation function over instances (tiles) to predict a bag label (slide label).

### 2) Frozen tile encoder + trainable slide encoder (LongNet) + classifier

- **Where**:
  - Model factory: `src/kidpro/modeling/factory_wsi.py`
  - LongNet MIL: `src/kidpro/modeling/longnet.py`
- **What**: Frozen foundation backbone produces per-tile embeddings; LongNet processes long sequences of tiles with positional information; classifier outputs logits.
- **Assumptions**:
  - Spatial coordinates carry signal; 2D positional embeddings are meaningful for WSI tile grids.
  - Long-range dependencies (or at least non-local aggregation) improve over mean/max pooling.
- **Justification**:
  - Freezing reduces variance/overfitting, especially with limited slide labels.
  - LongNet/dilated attention is designed to scale attention to long sequences.
  - Baselines (`mean_pool`, `max_pool`) exist to sanity-check whether the heavy aggregator is warranted.

### 3) Imbalance handling: balanced sampling vs class weights

- **Where**: `src/kidpro/train_wsi.py`
- **What**:
  - Balanced sampling uses `WeightedRandomSampler` to change the training distribution.
  - Class weights in `CrossEntropyLoss(weight=...)` scale per-example gradients.
- **Assumption**: Class imbalance is large enough to harm optimization/metrics.
- **Justification**: Either method can help when minority class is underrepresented.
- **Risk**: Using both simultaneously can overweight the minority class and skew gradients; in extreme cases it can cause degenerate solutions (e.g. predicting the minority class too often).
- **Recommendation**: Use **one** mechanism; prefer balanced sampling if you want stable batch composition; prefer class weights if you want to keep the empirical distribution but reweight gradients.

### 4) Optimization: AdamW + warmup + cosine + gradient clipping + AMP

- **Where**: `src/kidpro/train_wsi.py`, `src/kidpro/training/loop_mil.py`
- **What**:
  - `AdamW` with weight decay, optional LR multiplier for classifier head.
  - Linear warmup then cosine annealing (stepped per batch).
  - Gradient norm clipping.
  - CUDA AMP via autocast + GradScaler.
- **Justification**:
  - AdamW is a common default for transformer-like encoders.
  - Warmup reduces early instability; cosine annealing is a strong general-purpose schedule.
  - Gradient clipping mitigates rare large updates with long sequences / mixed precision.
  - AMP improves throughput and can be stable with GradScaler.
- **Risk**: None “theoretical” beyond standard hyperparameter sensitivity; log/monitor loss scale and gradient norms if instability persists.

### 5) Data transforms for MIL (flips/resize/crop/normalize)

- **Where**: `src/kidpro/data/transform.py`
- **What**:
  - Random flips (train only), resize to `data.patch_size`, optional center-crop to `model.input_size`, normalize, to tensor.
- **Justification**:
  - Flips are label-preserving for many histology tasks (unless orientation carries label).
  - Resize/crop matches backbone input requirements (e.g. 256→224 for ViT-style encoders).
  - Normalization should match the backbone’s pretraining distribution.
- **Important**: Prov-GigaPath’s published example uses **ImageNet mean/std** normalization.

### 6) Tile embedding caching

- **Where**:
  - Caching mechanism: `src/kidpro/data/dataset_mil.py`
  - Usage: `src/kidpro/training/loop_mil.py` (`_stream_slide_logits`)
- **What**: Cache tile-encoder embeddings per slide so tile encoder runs once; train only slide encoder + classifier each epoch.
- **Justification**: If the tile encoder is frozen, caching is theoretically equivalent to recomputing embeddings (up to numerical noise) and drastically speeds training.
- **Risk**: If your training transforms are stochastic (e.g., random flips), caching makes the effective augmentation **fixed** (whatever was seen when cached), which is *not* the same as fresh augmentation each epoch.
- **Recommendation**: If embedding cache is enabled, keep MIL tile transforms deterministic (or disable embedding cache when stochastic aug is enabled).

### 7) Evaluation metrics + thresholding

- **Where**: `src/kidpro/training/loop_mil.py`
- **What**:
  - Reports accuracy, macro-F1, AUC (if both classes present), confusion matrix, precision/recall.
  - Sweeps thresholds to maximize macro-F1.
- **Justification**:
  - **AUC** is threshold-agnostic and robust under imbalance.
  - **Macro-F1** treats both classes symmetrically (good when minority class matters).
- **Risk**:
  - “Best threshold” is tuned on the validation set; treat it as model selection, not a free extra metric.
  - No probability calibration is applied; softmax probabilities may be miscalibrated.

## Recommended defaults (WSI/MIL)

- **Imbalance**: choose *one* of:
  - `train.use_balanced_sampling: true`, `train.use_class_weights: false` (simple and stable)
  - OR `train.use_balanced_sampling: false`, `train.use_class_weights: true` (keeps empirical distribution)
- **Transforms**: for Prov-GigaPath/timm backbones, use ImageNet mean/std normalization; keep 256→224 resize/crop behavior.
- **Embedding cache**: if `dataset.data.mil_cache.cache_tile_embeddings: true`, avoid stochastic flips (deterministic transforms).
- **Selection metric**: checkpoint “best” using the same metric as early stopping.
- **LoRA**: in MIL, apply LoRA to the **slide encoder**; keep the tile encoder frozen.
