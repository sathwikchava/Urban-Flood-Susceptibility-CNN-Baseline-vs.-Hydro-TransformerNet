# Urban Flood Susceptibility — CNN Baseline vs. a Simplified Hydro-TransformerNet

A 12-hour AI/ML technical assessment submission: a small-scale, honestly-evaluated implementation of the
methodology behind **Hydro-TransformerNet** (Chakrabortty et al., 2026, *Earth Systems and Environment*),
built to directly answer one question — on a controlled synthetic task, does adding a transformer bottleneck
and a hydro-inspired attention gate actually improve on a plain CNN encoder–decoder?

## Overview

The reference paper predicts urban flood susceptibility for Sharjah, UAE, using a ResNet-based CNN encoder,
a transformer that models the *temporal* persistence of rainfall/surface conditions, and a hydrologically
guided attention gate, trained against an expert-designed synthetic flood mask (no historical flood inventory
exists for the study area). This project is **not** an attempt to reproduce that paper's architecture, data
scale, or dataset — the assessment brief is explicit that this isn't required. Instead, it implements the same
overall *methodology* (CNN → attention bottleneck → hydro-attention gate → decoder, trained with a BCE-style
loss against a synthetic mask) at toy scale, and puts it head-to-head against a matched CNN-only baseline so
the effect of the added block can actually be measured rather than assumed.

## Dataset Used

Fully synthetic, generated inside the notebook — no licensed real Sharjah/UAE raster data was available for
this assessment. Each sample is a 9-channel × 32×32 patch:

| Channel | Represents |
|---|---|
| 0–8 | DEM, Slope, TWI, NDVI, NDWI, LULC, Impervious Surface, Soil Type, Rainfall |

Every channel is generated as a **low-frequency, Gaussian-smoothed random field**, not independent per-pixel
noise — this mirrors the spatial autocorrelation a real raster has (a real DEM or rainfall surface doesn't
jump randomly from one pixel to the next). Slope is derived from the DEM's gradient rather than sampled on
its own, the way it would be from a real elevation model. The binary flood label combines low elevation, high
TWI/rainfall/imperviousness with a genuine **neighbourhood-dependent depression term** (`local_mean(DEM, 7×7)
− DEM`), so the label cannot be recovered from a single pixel's values in isolation — a model needs some
spatial context to do well. The result is then quantile-thresholded per patch to hold the positive-pixel rate
close to a stable **~40%** in both splits: 800 training patches, 200 validation patches.

## Approach

Two models sharing an identical CNN encoder/decoder backbone, so the comparison isolates the effect of the
added block:

- **Model A — CNN Baseline:** a U-Net-style encoder–decoder, residual `ConvBlock`s, 4 depth levels
  (32→64→128→256 channels), skip connections. No attention or temporal component.
- **Model B — Simplified HydroTransformerNet:** same backbone, plus at the bottleneck a
  `TransformerBottleneck` (4-head self-attention **over the 16 spatial positions of the 4×4 bottleneck
  feature map**) and a `HydroAttentionGate` (`α = σ(Conv3×3(F))`, `F_out = F ⊙ α`).

  > **Framing note:** the paper's transformer attends across *simulated time steps* to model how rainfall and
  > surface conditions persist over time. What's built here has **no time axis at all** — every sample is one
  > static patch, so this is spatial self-attention across regions of that patch, used as a simplified
  > stand-in for the paper's temporal block because a synthetic single-timestamp dataset has no real
  > multi-day sequence to feed an actual temporal encoder. Likewise, the hydro-attention gate learns its mask
  > directly from the CNN/Transformer features rather than from a separate hydro-morphological mask input as
  > in the paper's eq. 8–9 — it's hydro-*inspired*, not a reproduction of that mechanism.

Both models are trained with `0.5 × BCE + 0.5 × Dice` loss, the AdamW optimizer with a cosine learning-rate
schedule, and early stopping on best validation AUC (patience 6, max 20 epochs), from the same random seed.

## Technology / Tools Used

- Python, PyTorch (`torch.nn.MultiheadAttention` for the transformer bottleneck)
- `scikit-learn` (AUC, F1, precision/recall, confusion matrix, ROC, precision–recall curve)
- `numpy`, `scipy.ndimage` (Gaussian/uniform filters for the synthetic rasters), `matplotlib`
- Runs on Google Colab or a local Jupyter environment; CUDA if available, CPU otherwise

## Setup Instructions

```bash
pip install torch scikit-learn scipy numpy matplotlib
```
On Colab, `scipy` and `scikit-learn` are already installed; the notebook's first cell installs them
defensively regardless.

## How to Run

1. Open `flood_susceptibility_cnn_vs_hydrotransformer.ipynb` in Jupyter or Colab.
2. Run all cells top to bottom. Roughly, in order:
   - Sec. 1 — synthetic dataset generation and a look at the generated channels/masks
   - Sec. 2–3 — model definitions (CNN Baseline, Simplified HydroTransformerNet)
   - Sec. 4–6 — loss/metrics/training loop, then training Model A and Model B
   - Sec. 7 — head-to-head metrics table, training curves, ROC curves, confusion matrices
   - Sec. 7a — Specificity and IoU, computed directly from the confusion matrices (no re-training)
   - Sec. 7b — a decision-threshold sweep (0.1–0.9) over the already-computed validation predictions
   - Sec. 8 — sample prediction maps and a look at the learned hydro-attention map
   - Sec. 9 — permutation feature importance
   - Sec. 10–11 — what the run actually shows, and where a production version would need to go further
3. Trained weights (`model_a.pt`, `model_b.pt`) and a results JSON are written to `outputs/`.

Training takes a few minutes on CPU at this patch size/dataset size, faster with a GPU.

## Results

These are the real numbers produced by running the notebook end-to-end, taken from each model's
best-validation-AUC checkpoint (200 validation patches, 32×32 pixels each = 204,800 total pixels):

| Metric | CNN Baseline | HydroTransformerNet | Higher |
|---|---:|---:|---|
| AUC | 0.9704 | 0.9706 | HydroTransformerNet |
| Accuracy | 0.9047 | 0.9050 | HydroTransformerNet |
| F1 | 0.8811 | 0.8811 | tie |
| Precision | 0.8798 | 0.8830 | HydroTransformerNet |
| **Recall** | **0.8825** | 0.8793 | **CNN Baseline** |
| Kappa | 0.8016 | 0.8020 | HydroTransformerNet |
| Specificity | 0.9195 | 0.9222 | HydroTransformerNet |
| IoU | 0.7875 | 0.7875 | tie |

Best checkpoint: epoch 19/20 for the CNN Baseline; epoch 12/20 for HydroTransformerNet (training continued to
epoch 18 before early stopping triggered on no further AUC improvement).

Confusion matrices — **CNN Baseline:** TP=72,367 · TN=112,911 · FP=9,889 · FN=9,633. **HydroTransformerNet:**
TP=72,102 · TN=113,243 · FP=9,557 · FN=9,898.

**Threshold sweep (Sec. 7b)** — Recall / Precision / Specificity / IoU at nine thresholds from 0.1 to 0.9:

| Threshold | Recall (CNN) | Recall (Hydro) | Specificity (CNN) | Specificity (Hydro) | IoU (CNN) | IoU (Hydro) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.9681 | 0.9687 | 0.7860 | 0.7854 | 0.7331 | 0.7331 |
| 0.2 | 0.9454 | 0.9451 | 0.8442 | 0.8444 | 0.7666 | 0.7665 |
| 0.3 | 0.9244 | 0.9241 | 0.8772 | 0.8777 | 0.7808 | 0.7810 |
| 0.4 | 0.9039 | 0.9022 | 0.9006 | 0.9020 | 0.7867 | 0.7868 |
| 0.5 | 0.8825 | 0.8793 | 0.9195 | 0.9222 | 0.7875 | 0.7875 |
| 0.6 | 0.8583 | 0.8537 | 0.9363 | 0.9390 | 0.7835 | 0.7823 |
| 0.7 | 0.8276 | 0.8215 | 0.9517 | 0.9544 | 0.7718 | 0.7690 |
| 0.8 | 0.7873 | 0.7786 | 0.9663 | 0.9692 | 0.7495 | 0.7442 |
| 0.9 | 0.7177 | 0.7069 | 0.9826 | 0.9849 | 0.6995 | 0.6912 |

Except at the lowest threshold tested (0.1), the CNN Baseline holds a small but consistent recall edge over
HydroTransformerNet across the entire sweep — this direction of trade-off (recall up, specificity down as the
threshold drops) is guaranteed by construction; the two models mainly differ in how they land on that curve.

**Permutation feature importance** (AUC drop when a channel is shuffled across the validation set):

| Channel | CNN Baseline | HydroTransformerNet |
|---|---:|---:|
| DEM | 0.1368 | 0.1330 |
| TWI | 0.1306 | 0.1346 |
| Rainfall | 0.0571 | 0.0563 |
| Impervious | 0.0028 | 0.0028 |
| Slope | 0.0010 | 0.0008 |
| NDVI | 0.0002 | 0.0003 |
| NDWI | 0.0002 | 0.0002 |
| SoilType | 0.0000 | 0.0000 |
| LULC | -0.0001 | 0.0000 |

DEM and TWI dominate for both models, Rainfall is a clear secondary driver, and NDVI/NDWI/LULC/SoilType/Slope
contribute almost nothing — **qualitatively similar to the paper's SHAP-based ranking** (elevation, slope and
rainfall as the dominant predictors), obtained here with a different attribution method (permutation
importance rather than SHAP) on entirely different data. This should be read as a qualitative echo, not a
validation of the paper's specific SHAP values.

## Key Observations

1. **The two models are extremely close overall.** F1 and IoU tie at four decimal places; AUC, accuracy,
   precision, kappa and specificity all favor HydroTransformerNet, but only narrowly.
2. **The CNN Baseline has the higher recall — at the default threshold and almost everywhere else in the
   sweep.** Since a missed flood-prone pixel is the more expensive error for disaster management, this is a
   genuine trade-off rather than a clean win for the more complex architecture.
3. **A U-Net encoder already has a fairly large effective receptive field by the bottleneck** (three stages
   of 2× pooling), so much of the "longer-range context" the transformer could add is already partly captured
   by the plain CNN. Attention layered on top of an already-deep CNN tends to produce modest rather than
   dramatic gains — especially at this dataset's scale. A more decisive win would likely need more data
   diversity, a task with dependencies genuinely beyond the CNN's receptive field, and/or real (not
   synthetic) labels.
4. **Lowering the threshold trades specificity for recall exactly as expected**, and IoU actually dips at the
   most permissive thresholds tested once the extra false positives start to outweigh the extra true
   positives.
5. **The learned hydro-attention map is visually plausible but not dramatically different from a generic
   saliency map** on this synthetic dataset — worth stating as a limitation rather than oversold as a
   strength.
6. **Both models plateau in the mid-to-late epochs**, which is exactly why early stopping on best validation
   AUC is used rather than reporting whichever epoch training happens to finish on.
7. **A production deployment should not default to a 0.5 threshold.** For disaster-management use, the
   operating point should be chosen from the Sec. 7b sweep to favor recall within an acceptable false-alarm
   budget, not left at the arbitrary default.

## Limitations

See `technical_documentation.md`, Section 11, for the full breakdown (synthetic data and mask, spatial rather
than temporal attention, small patch size, permutation importance in place of SHAP, no GIS export in this
proof-of-concept).
