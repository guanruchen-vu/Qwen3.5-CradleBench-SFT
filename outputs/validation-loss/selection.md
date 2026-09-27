# Validation loss by epoch (CRADLE Bench checkpoint rule: lowest validation loss)

Answer-token cross-entropy (nats), per-post mean over all validation posts; token-weighted mean in parentheses. ✓ = lowest validation loss.

| Run | Split | Weighting | Epoch 1 | Epoch 2 | Epoch 3 | Selected by loss | Selected by crisis Macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen35-4b-lora-sft-v1` | train_consensus | none | 0.1666 (0.1607) ✓ | 0.2668 (0.2413) | 0.3604 (0.3193) | 1 | 3 |
| `qwen35-4b-lora-sft-unanimous-v1` | train_unanimous | none | 0.2136 (0.2081) ✓ | 0.2387 (0.2213) | 0.3374 (0.3032) | 1 | 1 |
| `qwen35-4b-lora-sft-unanimous-ipw-seed42` | train_unanimous | label_ipw | 0.2395 (0.2263) ✓ | 0.2718 (0.2436) | 0.3343 (0.3040) | 1 | 1 |
| `qwen35-9b-lora-sft-consensus-seed42` | train_consensus | none | 0.1855 (0.1697) ✓ | 0.2922 (0.2623) | 0.3967 (0.3484) | 1 | 2 |
| `qwen35-9b-lora-sft-unanimous-seed42` | train_unanimous | none | 0.2311 (0.2361) ✓ | 0.2588 (0.2399) | 0.3779 (0.3426) | 1 | 3 |
| `qwen35-9b-lora-sft-unanimous-ipw-seed42` | train_unanimous | label_ipw | 0.2140 (0.2158) ✓ | 0.2307 (0.2268) | 0.3575 (0.3304) | 1 | 2 |
