# Urban Flood Susceptibility — CNN vs. CNN+Transformer+Hydro-Attention

Simplified proof-of-concept based on **Hydro-TransformerNet** (Chakrabortty et al., 2026, *Earth Systems and
Environment*), built for the 12-hour AI/ML Technical Assessment.

## Project Overview

The reference paper proposes Hydro-TransformerNet: a ResUNet CNN encoder + a transformer module (modelling
*temporal* persistence of rainfall/surface conditions) + a hydro-attention gate, trained against a synthetic
flood mask because no real flood inventory exists for the paper's Sharjah, UAE study area. This project
implements a simplified version of that methodology at toy scale, and directly compares it against a
plain CNN baseline to see what the transformer + hydro-attention block actually buys you.

**This is not a reproduction of the paper's full architecture, dataset, or scale** — the assessment brief
explicitly does not require that. It is a working, honestly-evaluated implementation of the same
*methodology* (CNN → attention bottleneck → hydro-attention gate → decoder, BCE-based training on a
synthetic mask).

## Dataset Used

**Fully synthetic**, generated in-notebook — no real Sharjah/UAE data was available for this assessment.
Each sample is a 9-channel × 32×32 patch:

| Channel | Stands in for |
|---|---|
| 0–8 | DEM, Slope, TWI, NDVI, NDWI, LULC, Impervious Surface, Soil Type, Rainfall |

Channels are generated as **spatially-smoothed, low-frequency random fields** (Gaussian-filtered noise,
min-max normalized), not per-pixel i.i.d. noise — this mimics the spatial autocorrelation of a real raster
(a real DEM doesn't jump randomly pixel-to-pixel) and was a deliberate fix over an earlier draft that used
independent per-pixel noise. The binary flood label is derived from a rule combining low elevation, high
TWI/NDWI, high imperviousness, and a genuine **neighborhood-dependent local-depression term** (not just a
per-pixel threshold), then quantile-thresholded per patch to keep class balance roughly stable. 800 training
patches, 200 validation patches.

## Approach

Two models, identical CNN encoder/decoder backbone, to isolate the effect of the added block:

- **Model A — CNN Baseline:** U-Net-style encoder–decoder, residual ConvBlocks, 4 depth levels
  (32→64→128→256 channels), skip connections.
- **Model B — HydroTransformerNet (simplified):** same backbone + a `TransformerBottleneck` (multi-head
  self-attention **over spatial positions of the bottleneck feature map**) + a `HydroAttentionGate`
  (`α = σ(Conv3×3(F))`, `F_out = F ⊙ α`).

  > **Important framing correction:** the paper's transformer runs over *simulated time steps* to model
  > rainfall/surface-condition persistence over time. What's implemented here has **no time axis at all** —
  > it's spatial self-attention across regions of a single static patch, used as a simplified stand-in for
  > the paper's temporal component (because this POC has no real multi-day rainfall time series to feed a
  > true temporal encoder). It should be described that way, not as "the temporal component."

Both trained with 0.5×BCE + 0.5×Dice loss, AdamW + cosine LR schedule, early stopping on best validation AUC
(patience 6, max 20 epochs), identical seed.

## Technology / Tools Used

- Python, PyTorch (`torch.nn.MultiheadAttention` for the transformer bottleneck)
- `scikit-learn` (AUC, F1, confusion matrix, ROC)
- `numpy`, `matplotlib`
- Google Colab / local Jupyter (CUDA if available, CPU fallback)

## Setup Instructions

```bash
pip install torch scikit-learn scipy numpy matplotlib
```
(On Colab, `scipy`/`scikit-learn` are already present; the notebook installs them defensively.)

## How to Run the Code

1. Open `flood_susceptibility_cnn_vs_hydrotransformer.ipynb` in Jupyter or Colab.
2. Run all cells top to bottom. Sections in order:
   - Sec. 1: synthetic dataset generation + visualization
   - Sec. 2–3: model definitions (CNN Baseline, HydroTransformerNet)
   - Sec. 4–6: training loop, Model A training, Model B training
   - Sec. 7: head-to-head metrics table + training curves + ROC + confusion matrices
   - **Sec. 7a (new): Specificity + IoU** — computed directly from the confusion matrices, no re-training
   - **Sec. 7b (new): threshold sweep** — Recall/Precision/Specificity/IoU at thresholds 0.1–0.9, using the
     already-computed validation predictions, no re-training
   - Sec. 8: sample prediction maps + hydro-attention visualization
   - Sec. 9: permutation feature importance
   - Sec. 10–11: key observations, limitations
3. Trained weights and a results JSON are saved to `outputs/`.

Training takes a few minutes on CPU for this patch size/dataset size; faster with a GPU.

## Results

Real, reproducible numbers from the best-validation-AUC checkpoint of each model (200 validation patches,
32×32 pixels = 204,800 total pixels):

| Metric | CNN Baseline | HydroTransformerNet | Winner |
|---|---:|---:|---|
| AUC | 0.9705 | 0.9707 | HydroTransNet |
| Accuracy | 0.9046 | 0.9056 | HydroTransNet |
| F1 | 0.8810 | 0.8817 | HydroTransNet |
| Precision | 0.8804 | 0.8852 | HydroTransNet |
| **Recall** | **0.8815** | 0.8781 | **CNN Baseline** |
| Kappa | 0.8014 | 0.8032 | HydroTransNet |
| Specificity | 0.9201 | 0.9240 | HydroTransNet |
| IoU | 0.7873 | 0.7884 | HydroTransNet |

Confusion matrices — CNN Baseline: TP=72,287 TN=112,984 FP=9,816 FN=9,713. HydroTransformerNet: TP=72,005
TN=113,464 FP=9,336 FN=9,995.

**Threshold sweep (Sec. 7b):** decreasing the threshold below 0.5 will monotonically raise Recall and lower
Specificity for both models — that direction is guaranteed by construction. The exact magnitude for these two
trained models is printed/plotted by Sec. 7b when the notebook is run (it re-thresholds the already-computed
validation predictions, no retraining required); it isn't restated here as a fixed table because this
document was written without a live PyTorch run available, and restating invented numbers would repeat
exactly the mistake this revision was meant to fix. Re-run Sec. 7b and copy its printed table in if you want
it inline here.

Permutation feature importance (AUC drop when a channel is shuffled): DEM and TWI dominate (~0.13 each) for
both models, Rainfall is a clear secondary driver (~0.057), NDVI/NDWI/LULC/SoilType/Slope contribute almost
nothing (~0.0–0.003) — **qualitatively similar to the paper's SHAP-based ranking, obtained using a different
attribution method (permutation importance, not SHAP)**. This is a qualitative echo, not a validation of the
paper's specific SHAP values.

## Key Observations

1. **HydroTransformerNet wins narrowly on 7 of 8 metrics** — but the **CNN baseline has the higher Recall**
   (0.8815 vs 0.8781). Since a missed flood pixel is the costlier error for disaster management, the simpler
   CNN is not obviously the worse choice here; this is a genuine trade-off, not a clean win either way.
2. **The gap between the two models is small everywhere.** A U-Net encoder already has a large receptive
   field by the bottleneck (three 2× poolings), so much of the "long-range context" the transformer adds is
   already partly captured by the CNN alone. Attention on top of an already-deep CNN tends to give modest,
   not dramatic, gains — especially at this dataset's scale. A bigger, cleaner win would likely need more
   data diversity, a task with dependencies genuinely beyond the CNN's receptive field, and/or real (not
   synthetic) labels.
3. **Specificity/IoU tell the same story** as the core metrics: HydroTransformerNet is marginally better at
   rejecting non-flood pixels and has marginally better mask overlap, consistent with its precision/accuracy
   edge — not a dramatic difference.
4. **Both models plateau/slightly overfit around epoch 13–16**, hence early stopping on best validation AUC.
5. **The hydro-attention map is visually plausible but not dramatically distinct from a plain saliency map**
   on this synthetic dataset — a fair limitation to flag rather than a strength to oversell.
6. **Deployment threshold should not default to 0.5.** For disaster-management use, the operating threshold
   should be chosen from the Sec. 7b sweep to favor Recall within an acceptable false-alarm budget.

**Note on a previous draft:** an earlier version of this analysis reported AUC 0.9886 vs 0.9586 and Recall
0.998 vs 0.935, with false negatives reduced from 26 to 1. Those numbers were **not produced by training a
model** — they came from a hand-written NumPy script that injected noise directly into the label-generating
formula and added a hard-coded "hydro boost" to Model B's output. They have been discarded. Every number
above comes from the two networks' real, stored training/evaluation logs.

## Limitations

See `technical_documentation.md` Sec. 11 for the full table (synthetic data/mask, spatial-not-temporal
transformer, small patch size, permutation importance instead of SHAP, no GIS export in this POC).
