# CRADLE test-set summary with bootstrap 95% CI

Paired percentile bootstrap: 10,000 resamples of 600 test posts, seed 42. Each model uses the same resampled question IDs. CIs are percentile intervals; they are not significance tests.

All model values below are percentages: point estimate [95% CI].

| Metric | Gemma-4-E4B BF16 | Gemma-4-E4B Consensus SFT Epoch3 | Gemma-4-E4B Unanimous SFT Epoch3 | Gemma-4-E4B Unanimous-IPW SFT Epoch3 | Gemma-4-E4B Unanimous-CB SFT Epoch3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact Match | 57.17% [53.17%, 61.00%] | 76.50% [73.17%, 79.83%] | 75.83% [72.50%, 79.17%] | 76.83% [73.50%, 80.17%] | 76.17% [72.83%, 79.50%] |
| Sample Jaccard | 61.28% [57.57%, 64.90%] | 80.93% [78.09%, 83.78%] | 79.98% [76.96%, 82.95%] | 80.59% [77.64%, 83.53%] | 80.23% [77.21%, 83.18%] |
| Micro F1 | 61.87% [58.28%, 65.42%] | 80.73% [77.90%, 83.60%] | 80.03% [77.06%, 82.97%] | 80.62% [77.59%, 83.53%] | 80.03% [76.98%, 82.98%] |
| Macro F1 · all 15 | 49.85% [45.45%, 53.70%] | 73.63% [67.14%, 78.23%] | 73.14% [66.43%, 77.77%] | 74.58% [68.09%, 78.97%] | 72.61% [66.24%, 77.03%] |
| Macro F1 · crisis 14 | 49.07% [44.45%, 53.10%] | 72.72% [65.80%, 77.60%] | 72.20% [65.08%, 77.09%] | 73.81% [66.92%, 78.46%] | 71.61% [64.80%, 76.26%] |
| Micro Recall | 62.84% [59.15%, 66.47%] | 82.24% [79.45%, 85.09%] | 80.75% [77.75%, 83.65%] | 81.34% [78.36%, 84.20%] | 81.04% [78.02%, 83.96%] |
| Macro Recall · all 15 | 56.25% [51.50%, 61.73%] | 76.06% [69.99%, 82.04%] | 74.17% [67.94%, 80.19%] | 76.03% [69.73%, 81.62%] | 74.30% [68.29%, 80.14%] |
| Macro Recall · crisis 14 | 57.04% [51.97%, 62.84%] | 75.66% [69.21%, 82.03%] | 73.67% [67.00%, 80.11%] | 75.78% [69.05%, 81.72%] | 73.73% [67.31%, 79.97%] |
| Valid Format | 98.50% [97.50%, 99.33%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |

## Paired differences

Positive values favor the later model. Units are percentage points (pp).

| Metric | Gemma-4-E4B Consensus SFT Epoch3 − Gemma-4-E4B BF16 | Gemma-4-E4B Unanimous SFT Epoch3 − Gemma-4-E4B BF16 | Gemma-4-E4B Unanimous-IPW SFT Epoch3 − Gemma-4-E4B BF16 | Gemma-4-E4B Unanimous-CB SFT Epoch3 − Gemma-4-E4B BF16 | Gemma-4-E4B Unanimous SFT Epoch3 − Gemma-4-E4B Consensus SFT Epoch3 | Gemma-4-E4B Unanimous-IPW SFT Epoch3 − Gemma-4-E4B Consensus SFT Epoch3 | Gemma-4-E4B Unanimous-CB SFT Epoch3 − Gemma-4-E4B Consensus SFT Epoch3 | Gemma-4-E4B Unanimous-IPW SFT Epoch3 − Gemma-4-E4B Unanimous SFT Epoch3 | Gemma-4-E4B Unanimous-CB SFT Epoch3 − Gemma-4-E4B Unanimous SFT Epoch3 | Gemma-4-E4B Unanimous-CB SFT Epoch3 − Gemma-4-E4B Unanimous-IPW SFT Epoch3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact Match | +19.33 pp [+15.17, +23.50] | +18.67 pp [+14.50, +22.83] | +19.67 pp [+15.67, +23.83] | +19.00 pp [+15.00, +23.00] | -0.67 pp [-3.00, +1.67] | +0.33 pp [-1.83, +2.67] | -0.33 pp [-2.67, +1.83] | +1.00 pp [-0.67, +2.67] | +0.33 pp [-1.50, +2.00] | -0.67 pp [-2.33, +1.00] |
| Sample Jaccard | +19.65 pp [+15.93, +23.44] | +18.70 pp [+14.88, +22.56] | +19.31 pp [+15.57, +23.11] | +18.95 pp [+15.16, +22.69] | -0.95 pp [-2.96, +1.05] | -0.34 pp [-2.22, +1.49] | -0.70 pp [-2.79, +1.32] | +0.61 pp [-0.89, +2.14] | +0.25 pp [-1.33, +1.78] | -0.36 pp [-1.94, +1.22] |
| Micro F1 | +18.87 pp [+15.37, +22.49] | +18.16 pp [+14.64, +21.79] | +18.76 pp [+15.26, +22.34] | +18.16 pp [+14.57, +21.78] | -0.70 pp [-2.59, +1.18] | -0.11 pp [-1.90, +1.61] | -0.70 pp [-2.75, +1.26] | +0.59 pp [-0.80, +1.99] | -0.00 pp [-1.55, +1.49] | -0.59 pp [-2.20, +0.95] |
| Macro F1 · all 15 | +23.78 pp [+17.80, +29.19] | +23.30 pp [+17.30, +28.50] | +24.73 pp [+18.72, +29.87] | +22.77 pp [+16.98, +27.93] | -0.48 pp [-6.30, +5.20] | +0.95 pp [-2.72, +5.59] | -1.02 pp [-4.00, +1.77] | +1.44 pp [-1.67, +5.70] | -0.53 pp [-6.03, +5.04] | -1.97 pp [-6.71, +1.78] |
| Macro F1 · crisis 14 | +23.64 pp [+17.29, +29.35] | +23.13 pp [+16.77, +28.58] | +24.74 pp [+18.32, +30.16] | +22.53 pp [+16.40, +27.95] | -0.51 pp [-6.72, +5.56] | +1.10 pp [-2.79, +6.07] | -1.11 pp [-4.24, +1.82] | +1.61 pp [-1.71, +6.16] | -0.60 pp [-6.46, +5.35] | -2.21 pp [-7.26, +1.78] |
| Micro Recall | +19.40 pp [+15.78, +23.13] | +17.91 pp [+14.29, +21.64] | +18.51 pp [+14.97, +22.19] | +18.21 pp [+14.54, +21.88] | -1.49 pp [-3.64, +0.61] | -0.90 pp [-2.82, +1.03] | -1.19 pp [-3.40, +0.90] | +0.60 pp [-0.91, +2.16] | +0.30 pp [-1.36, +1.94] | -0.30 pp [-1.97, +1.34] |
| Macro Recall · all 15 | +19.81 pp [+13.87, +26.46] | +17.92 pp [+11.93, +24.33] | +19.78 pp [+13.53, +26.06] | +18.05 pp [+12.26, +24.36] | -1.89 pp [-8.61, +4.76] | -0.03 pp [-4.40, +5.12] | -1.76 pp [-4.94, +1.29] | +1.86 pp [-1.89, +6.65] | +0.13 pp [-6.20, +6.65] | -1.73 pp [-6.92, +2.69] |
| Macro Recall · crisis 14 | +18.62 pp [+12.24, +25.73] | +16.63 pp [+10.19, +23.49] | +18.74 pp [+12.03, +25.43] | +16.69 pp [+10.53, +23.39] | -1.99 pp [-9.17, +5.13] | +0.12 pp [-4.55, +5.65] | -1.93 pp [-5.32, +1.33] | +2.11 pp [-1.93, +7.21] | +0.07 pp [-6.71, +7.05] | -2.04 pp [-7.59, +2.66] |
| Valid Format | +1.50 pp [+0.67, +2.50] | +1.50 pp [+0.67, +2.50] | +1.50 pp [+0.67, +2.50] | +1.50 pp [+0.67, +2.50] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |

Note: Macro F1 averages 15 labels (or 14 crisis labels), counting a label with no positive samples in a resample as F1 = 0, matching the existing analysis script.
