# CRADLE test-set summary with bootstrap 95% CI

Paired percentile bootstrap: 10,000 resamples of 600 test posts, seed 42. Each model uses the same resampled question IDs. CIs are percentile intervals; they are not significance tests.

All model values below are percentages: point estimate [95% CI].

| Metric | Qwen3.5-4B BF16 | Qwen3.5-4B SFT Epoch3 | Qwen3.5-4B SFT Epoch1 Unanimous |
| --- | ---: | ---: | ---: |
| Exact Match | 45.83% [41.83%, 49.83%] | 75.00% [71.50%, 78.50%] | 74.50% [71.00%, 78.00%] |
| Sample Jaccard | 48.55% [44.63%, 52.47%] | 79.24% [76.16%, 82.23%] | 77.60% [74.42%, 80.69%] |
| Micro F1 | 48.31% [44.41%, 52.18%] | 79.47% [76.44%, 82.41%] | 77.27% [73.91%, 80.49%] |
| Macro F1 · all 15 | 36.50% [32.04%, 40.47%] | 71.85% [65.63%, 76.27%] | 64.91% [59.04%, 69.43%] |
| Macro F1 · crisis 14 | 34.77% [30.05%, 38.98%] | 70.89% [64.29%, 75.59%] | 63.53% [57.32%, 68.32%] |
| Micro Recall | 46.87% [42.96%, 50.76%] | 80.30% [77.22%, 83.33%] | 75.07% [71.64%, 78.42%] |
| Macro Recall · all 15 | 42.57% [37.92%, 47.89%] | 73.77% [67.67%, 79.67%] | 62.24% [57.36%, 67.33%] |
| Macro Recall · crisis 14 | 41.89% [36.96%, 47.55%] | 73.39% [66.84%, 79.73%] | 60.85% [55.66%, 66.23%] |
| Valid Format | 95.67% [94.00%, 97.17%] | 99.00% [98.17%, 99.67%] | 99.00% [98.17%, 99.67%] |

## Paired differences

Positive values favor the later model. Units are percentage points (pp).

| Metric | Qwen3.5-4B SFT Epoch3 − Qwen3.5-4B BF16 | Qwen3.5-4B SFT Epoch1 Unanimous − Qwen3.5-4B BF16 | Qwen3.5-4B SFT Epoch1 Unanimous − Qwen3.5-4B SFT Epoch3 |
| --- | ---: | ---: | ---: |
| Exact Match | +29.17 pp [+24.67, +33.67] | +28.67 pp [+24.33, +33.17] | -0.50 pp [-3.50, +2.50] |
| Sample Jaccard | +30.69 pp [+26.43, +34.99] | +29.05 pp [+24.80, +33.36] | -1.64 pp [-4.31, +1.04] |
| Micro F1 | +31.16 pp [+26.80, +35.58] | +28.96 pp [+24.54, +33.35] | -2.20 pp [-5.00, +0.61] |
| Macro F1 · all 15 | +35.35 pp [+28.47, +41.58] | +28.41 pp [+22.96, +33.60] | -6.94 pp [-12.91, -0.44] |
| Macro F1 · crisis 14 | +36.12 pp [+28.88, +42.67] | +28.76 pp [+22.98, +34.28] | -7.36 pp [-13.71, -0.46] |
| Micro Recall | +33.43 pp [+29.06, +37.87] | +28.21 pp [+23.86, +32.54] | -5.22 pp [-8.27, -2.27] |
| Macro Recall · all 15 | +31.20 pp [+22.32, +39.08] | +19.67 pp [+14.01, +25.09] | -11.53 pp [-17.88, -4.91] |
| Macro Recall · crisis 14 | +31.51 pp [+21.97, +39.97] | +18.96 pp [+12.91, +24.65] | -12.54 pp [-19.33, -5.49] |
| Valid Format | +3.33 pp [+2.00, +4.83] | +3.33 pp [+2.00, +4.83] | +0.00 pp [+0.00, +0.00] |

Note: Macro F1 averages 15 labels (or 14 crisis labels), counting a label with no positive samples in a resample as F1 = 0, matching the existing analysis script.
