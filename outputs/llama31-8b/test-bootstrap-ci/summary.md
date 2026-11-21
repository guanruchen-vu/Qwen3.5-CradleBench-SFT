# CRADLE test-set summary with bootstrap 95% CI

Paired percentile bootstrap: 10,000 resamples of 600 test posts, seed 42. Each model uses the same resampled question IDs. CIs are percentile intervals; they are not significance tests.

All model values below are percentages: point estimate [95% CI].

| Metric | Llama-3.1-8B BF16 | Llama-3.1-8B Consensus SFT Epoch3 | Llama-3.1-8B Unanimous SFT Epoch3 | Llama-3.1-8B Unanimous-IPW SFT Epoch3 |
| --- | ---: | ---: | ---: | ---: |
| Exact Match | 27.33% [23.83%, 31.00%] | 75.67% [72.33%, 79.00%] | 77.00% [73.67%, 80.33%] | 76.50% [73.00%, 79.83%] |
| Sample Jaccard | 41.29% [38.08%, 44.61%] | 79.73% [76.69%, 82.72%] | 80.12% [77.07%, 83.13%] | 79.83% [76.72%, 82.92%] |
| Micro F1 | 44.58% [41.39%, 47.72%] | 79.29% [76.18%, 82.28%] | 79.55% [76.41%, 82.58%] | 79.16% [75.95%, 82.28%] |
| Macro F1 · all 15 | 36.68% [33.26%, 39.66%] | 72.41% [65.51%, 77.05%] | 71.22% [64.64%, 76.24%] | 69.62% [63.05%, 74.69%] |
| Macro F1 · crisis 14 | 35.40% [31.86%, 38.46%] | 71.47% [64.05%, 76.41%] | 70.20% [63.21%, 75.55%] | 68.43% [61.45%, 73.83%] |
| Micro Recall | 54.03% [50.29%, 57.84%] | 80.30% [77.10%, 83.43%] | 79.25% [76.08%, 82.41%] | 79.10% [75.88%, 82.26%] |
| Macro Recall · all 15 | 49.33% [44.89%, 53.95%] | 74.74% [68.55%, 80.82%] | 71.41% [65.92%, 77.35%] | 70.95% [65.83%, 76.67%] |
| Macro Recall · crisis 14 | 49.90% [45.19%, 54.82%] | 74.40% [67.77%, 80.90%] | 70.71% [64.88%, 77.10%] | 70.22% [64.71%, 76.34%] |
| Valid Format | 86.50% [83.67%, 89.17%] | 99.67% [99.17%, 100.00%] | 99.67% [99.17%, 100.00%] | 100.00% [100.00%, 100.00%] |

## Paired differences

Positive values favor the later model. Units are percentage points (pp).

| Metric | Llama-3.1-8B Consensus SFT Epoch3 − Llama-3.1-8B BF16 | Llama-3.1-8B Unanimous SFT Epoch3 − Llama-3.1-8B BF16 | Llama-3.1-8B Unanimous-IPW SFT Epoch3 − Llama-3.1-8B BF16 | Llama-3.1-8B Unanimous SFT Epoch3 − Llama-3.1-8B Consensus SFT Epoch3 | Llama-3.1-8B Unanimous-IPW SFT Epoch3 − Llama-3.1-8B Consensus SFT Epoch3 | Llama-3.1-8B Unanimous-IPW SFT Epoch3 − Llama-3.1-8B Unanimous SFT Epoch3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact Match | +48.33 pp [+43.83, +52.83] | +49.67 pp [+45.17, +54.17] | +49.17 pp [+44.67, +53.50] | +1.33 pp [-1.17, +3.83] | +0.83 pp [-1.83, +3.50] | -0.50 pp [-2.33, +1.33] |
| Sample Jaccard | +38.44 pp [+34.44, +42.46] | +38.83 pp [+34.78, +42.88] | +38.54 pp [+34.58, +42.57] | +0.38 pp [-1.70, +2.46] | +0.10 pp [-2.11, +2.33] | -0.29 pp [-1.87, +1.32] |
| Micro F1 | +34.71 pp [+30.74, +38.66] | +34.97 pp [+31.01, +38.92] | +34.58 pp [+30.62, +38.54] | +0.26 pp [-1.74, +2.27] | -0.13 pp [-2.33, +2.05] | -0.39 pp [-1.91, +1.16] |
| Macro F1 · all 15 | +35.72 pp [+28.58, +41.30] | +34.54 pp [+27.46, +40.48] | +32.93 pp [+25.85, +38.89] | -1.19 pp [-6.37, +2.95] | -2.79 pp [-7.96, +1.11] | -1.60 pp [-4.03, +0.60] |
| Macro F1 · crisis 14 | +36.06 pp [+28.49, +41.95] | +34.79 pp [+27.33, +41.13] | +33.02 pp [+25.52, +39.30] | -1.27 pp [-6.78, +3.12] | -3.04 pp [-8.54, +1.09] | -1.77 pp [-4.31, +0.55] |
| Micro Recall | +26.27 pp [+21.89, +30.65] | +25.22 pp [+20.80, +29.63] | +25.07 pp [+20.72, +29.50] | -1.04 pp [-3.13, +1.04] | -1.19 pp [-3.47, +1.03] | -0.15 pp [-1.78, +1.48] |
| Macro Recall · all 15 | +25.41 pp [+17.94, +32.64] | +22.08 pp [+14.94, +29.34] | +21.61 pp [+14.55, +28.70] | -3.33 pp [-8.40, +0.82] | -3.80 pp [-8.89, +0.21] | -0.46 pp [-3.02, +1.90] |
| Macro Recall · crisis 14 | +24.50 pp [+16.49, +32.18] | +20.81 pp [+13.26, +28.54] | +20.32 pp [+12.78, +27.88] | -3.69 pp [-9.16, +0.78] | -4.18 pp [-9.63, +0.10] | -0.50 pp [-3.24, +2.03] |
| Valid Format | +13.17 pp [+10.50, +16.00] | +13.17 pp [+10.33, +16.00] | +13.50 pp [+10.83, +16.33] | +0.00 pp [-0.50, +0.50] | +0.33 pp [+0.00, +0.83] | +0.33 pp [+0.00, +0.83] |

Note: Macro F1 averages 15 labels (or 14 crisis labels), counting a label with no positive samples in a resample as F1 = 0, matching the existing analysis script.
