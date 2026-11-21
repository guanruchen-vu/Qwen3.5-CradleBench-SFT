# Label retention, IPW weights, and test recall

Retention = train_unanimous count / train_consensus count. IPW weight = normalized training weight of that label (mean 1 over training posts). Test cells are predicted count / recall / F1 (%). Labels are sorted by retention.

| Label | Consensus | Unanimous | Retention | IPW weight | Test support | SFT Consensus | SFT Unanimous | SFT Unanimous-IPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `childabuse_endangerment_ongoing` | 39 | 15 | 0.385 | 1.827 | 9 | 8 / 44.4 / 47.1 | 8 / 66.7 / 70.6 | 6 / 44.4 / 53.3 |
| `suicideideation_active_past` | 60 | 24 | 0.400 | 1.756 | 11 | 15 / 81.8 / 69.2 | 12 / 63.6 / 60.9 | 12 / 72.7 / 69.6 |
| `suicideideation_passive_past` | 14 | 6 | 0.429 | 1.639 | 4 | 3 / 50.0 / 57.1 | 3 / 50.0 / 57.1 | 4 / 75.0 / 75.0 |
| `childabuse_endangerment_past` | 143 | 66 | 0.462 | 1.522 | 31 | 42 / 71.0 / 60.3 | 35 / 61.3 / 57.6 | 36 / 64.5 / 59.7 |
| `rape_ongoing` | 55 | 31 | 0.564 | 1.246 | 11 | 10 / 63.6 / 66.7 | 10 / 54.5 / 57.1 | 10 / 54.5 / 57.1 |
| `sexualharassment_past` | 156 | 90 | 0.577 | 1.218 | 38 | 39 / 78.9 / 77.9 | 38 / 76.3 / 76.3 | 38 / 76.3 / 76.3 |
| `selfharm_past` | 214 | 127 | 0.593 | 1.184 | 30 | 23 / 60.0 / 67.9 | 20 / 53.3 / 64.0 | 20 / 56.7 / 68.0 |
| `sexualharassment_ongoing` | 219 | 131 | 0.598 | 1.175 | 25 | 33 / 92.0 / 79.3 | 31 / 88.0 / 78.6 | 32 / 92.0 / 80.7 |
| `domesticviolence_past` | 267 | 168 | 0.629 | 1.117 | 33 | 29 / 72.7 / 77.4 | 32 / 81.8 / 83.1 | 30 / 81.8 / 85.7 |
| `domesticviolence_ongoing` | 233 | 150 | 0.644 | 1.091 | 25 | 35 / 88.0 / 73.3 | 36 / 84.0 / 68.9 | 39 / 84.0 / 65.6 |
| `suicideideation_active_ongoing` | 434 | 282 | 0.650 | 1.081 | 54 | 57 / 88.9 / 86.5 | 53 / 83.3 / 84.1 | 57 / 90.7 / 88.3 |
| `suicideideation_passive_ongoing` | 600 | 411 | 0.685 | 1.026 | 77 | 81 / 83.1 / 81.0 | 78 / 80.5 / 80.0 | 78 / 81.8 / 81.3 |
| `selfharm_ongoing` | 703 | 488 | 0.694 | 1.012 | 74 | 92 / 95.9 / 85.5 | 96 / 95.9 / 83.5 | 94 / 95.9 / 84.5 |
| `rape_past` | 352 | 248 | 0.705 | 0.997 | 62 | 62 / 88.7 / 88.7 | 66 / 91.9 / 89.1 | 65 / 90.3 / 88.2 |
| `no_crisis` | 1160 | 1020 | 0.879 | 0.799 | 186 | 166 / 81.7 / 86.4 | 164 / 81.2 / 86.3 | 161 / 79.6 / 85.3 |

## Retention vs. recall difference (14 crisis labels)

Spearman correlation with a two-sided permutation p-value (100,000 permutations, seed 42). Positive rho: the later model loses more recall on labels that lost more posts.

| Recall difference | Spearman rho | Permutation p |
| --- | ---: | ---: |
| SFT Unanimous − SFT Consensus | +0.267 | 0.3560 |
| SFT Unanimous-IPW − SFT Consensus | +0.323 | 0.2595 |
| SFT Unanimous-IPW − SFT Unanimous | -0.241 | 0.4081 |

Exploratory: several labels have fewer than 12 test posts, so per-label recall is noisy.
