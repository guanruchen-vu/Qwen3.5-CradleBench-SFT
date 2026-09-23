# CRADLE test-set summary with bootstrap 95% CI

Paired percentile bootstrap: 10,000 resamples of 600 test posts, seed 42. Each model uses the same resampled question IDs. CIs are percentile intervals; they are not significance tests.

All model values below are percentages: point estimate [95% CI].

| Metric | BF16 base | SFT Consensus | SFT Unanimous | SFT Unanimous-IPW |
| --- | ---: | ---: | ---: | ---: |
| Exact Match | 46.50% [42.50%, 50.50%] | 75.83% [72.50%, 79.17%] | 75.17% [71.83%, 78.50%] | 73.33% [69.83%, 76.83%] |
| Sample Jaccard | 49.21% [45.28%, 53.14%] | 80.07% [77.06%, 82.99%] | 78.26% [75.11%, 81.32%] | 76.83% [73.58%, 79.99%] |
| Micro F1 | 48.66% [44.77%, 52.52%] | 79.74% [76.71%, 82.72%] | 77.46% [74.11%, 80.68%] | 76.83% [73.64%, 79.97%] |
| Macro F1 · all 15 | 36.96% [32.49%, 40.95%] | 72.08% [65.88%, 76.54%] | 65.22% [59.42%, 69.75%] | 65.70% [59.25%, 70.78%] |
| Macro F1 · crisis 14 | 35.25% [30.50%, 39.43%] | 71.11% [64.56%, 75.87%] | 63.85% [57.71%, 68.66%] | 64.44% [57.58%, 69.85%] |
| Micro Recall | 47.46% [43.56%, 51.42%] | 81.04% [78.05%, 84.04%] | 75.67% [72.25%, 78.97%] | 75.22% [71.98%, 78.50%] |
| Macro Recall · all 15 | 43.28% [38.65%, 48.65%] | 74.69% [68.53%, 80.51%] | 62.95% [58.06%, 68.01%] | 64.06% [58.65%, 69.99%] |
| Macro Recall · crisis 14 | 42.60% [37.68%, 48.25%] | 74.34% [67.78%, 80.58%] | 61.57% [56.33%, 66.93%] | 63.26% [57.49%, 69.64%] |
| Valid Format | 96.50% [95.00%, 98.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |

## Paired differences

Positive values favor the later model. Units are percentage points (pp).

| Metric | SFT Consensus − BF16 base | SFT Unanimous − BF16 base | SFT Unanimous-IPW − BF16 base | SFT Unanimous − SFT Consensus | SFT Unanimous-IPW − SFT Consensus | SFT Unanimous-IPW − SFT Unanimous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact Match | +29.33 pp [+24.83, +33.83] | +28.67 pp [+24.33, +33.00] | +26.83 pp [+22.33, +31.33] | -0.67 pp [-3.67, +2.33] | -2.50 pp [-5.50, +0.50] | -1.83 pp [-4.33, +0.67] |
| Sample Jaccard | +30.86 pp [+26.60, +35.13] | +29.05 pp [+24.81, +33.33] | +27.62 pp [+23.26, +32.03] | -1.81 pp [-4.47, +0.89] | -3.24 pp [-5.97, -0.56] | -1.43 pp [-3.93, +1.05] |
| Micro F1 | +31.07 pp [+26.75, +35.49] | +28.80 pp [+24.42, +33.18] | +28.17 pp [+23.90, +32.55] | -2.27 pp [-5.11, +0.57] | -2.91 pp [-5.68, -0.16] | -0.63 pp [-3.14, +1.91] |
| Macro F1 · all 15 | +35.12 pp [+28.21, +41.26] | +28.26 pp [+22.80, +33.43] | +28.74 pp [+22.06, +35.01] | -6.86 pp [-12.83, -0.36] | -6.38 pp [-12.71, -0.29] | +0.48 pp [-4.85, +5.61] |
| Macro F1 · crisis 14 | +35.86 pp [+28.57, +42.41] | +28.60 pp [+22.84, +34.10] | +29.19 pp [+22.06, +35.82] | -7.26 pp [-13.62, -0.33] | -6.67 pp [-13.43, -0.16] | +0.59 pp [-5.06, +6.08] |
| Micro Recall | +33.58 pp [+29.21, +38.02] | +28.21 pp [+23.92, +32.54] | +27.76 pp [+23.54, +32.18] | -5.37 pp [-8.42, -2.41] | -5.82 pp [-8.66, -3.01] | -0.45 pp [-3.06, +2.23] |
| Macro Recall · all 15 | +31.41 pp [+22.50, +39.34] | +19.67 pp [+13.99, +25.09] | +20.79 pp [+13.45, +28.27] | -11.74 pp [-18.13, -5.10] | -10.62 pp [-17.19, -4.62] | +1.12 pp [-3.74, +6.50] |
| Macro Recall · crisis 14 | +31.74 pp [+22.26, +40.21] | +18.96 pp [+12.91, +24.65] | +20.66 pp [+12.74, +28.67] | -12.77 pp [-19.61, -5.69] | -11.08 pp [-18.09, -4.61] | +1.70 pp [-3.47, +7.45] |
| Valid Format | +3.50 pp [+2.00, +5.00] | +3.50 pp [+2.00, +5.00] | +3.50 pp [+2.00, +5.00] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |

Note: Macro F1 averages 15 labels (or 14 crisis labels), counting a label with no positive samples in a resample as F1 = 0, matching the existing analysis script.
