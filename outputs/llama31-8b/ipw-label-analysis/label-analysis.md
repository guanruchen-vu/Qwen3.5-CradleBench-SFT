# Label retention, IPW weights, and test recall

Retention = train_unanimous count / train_consensus count. IPW weight = normalized training weight of that label (mean 1 over training posts). Test cells are predicted count / recall / F1 (%). Labels are sorted by retention.

| Label | Consensus | Unanimous | Retention | IPW weight | Test support | SFT Consensus | SFT Unanimous | SFT Unanimous-IPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `childabuse_endangerment_ongoing` | 39 | 15 | 0.385 | 1.827 | 9 | 12 / 44.4 / 38.1 | 6 / 44.4 / 53.3 | 6 / 33.3 / 40.0 |
| `suicideideation_active_past` | 60 | 24 | 0.400 | 1.756 | 11 | 13 / 81.8 / 75.0 | 11 / 72.7 / 72.7 | 12 / 72.7 / 69.6 |
| `suicideideation_passive_past` | 14 | 6 | 0.429 | 1.639 | 4 | 2 / 50.0 / 66.7 | 1 / 25.0 / 40.0 | 1 / 25.0 / 40.0 |
| `childabuse_endangerment_past` | 143 | 66 | 0.462 | 1.522 | 31 | 38 / 64.5 / 58.0 | 28 / 45.2 / 47.5 | 28 / 45.2 / 47.5 |
| `rape_ongoing` | 55 | 31 | 0.564 | 1.246 | 11 | 15 / 72.7 / 61.5 | 16 / 72.7 / 59.3 | 18 / 81.8 / 62.1 |
| `sexualharassment_past` | 156 | 90 | 0.577 | 1.218 | 38 | 37 / 78.9 / 80.0 | 35 / 78.9 / 82.2 | 37 / 76.3 / 77.3 |
| `selfharm_past` | 214 | 127 | 0.593 | 1.184 | 30 | 23 / 56.7 / 64.2 | 20 / 56.7 / 68.0 | 23 / 53.3 / 60.4 |
| `sexualharassment_ongoing` | 219 | 131 | 0.598 | 1.175 | 25 | 32 / 88.0 / 77.2 | 31 / 92.0 / 82.1 | 31 / 92.0 / 82.1 |
| `domesticviolence_past` | 267 | 168 | 0.629 | 1.117 | 33 | 28 / 69.7 / 75.4 | 32 / 75.8 / 76.9 | 31 / 69.7 / 71.9 |
| `domesticviolence_ongoing` | 233 | 150 | 0.644 | 1.091 | 25 | 39 / 84.0 / 65.6 | 41 / 84.0 / 63.6 | 42 / 88.0 / 65.7 |
| `suicideideation_active_ongoing` | 434 | 282 | 0.650 | 1.081 | 54 | 57 / 88.9 / 86.5 | 55 / 85.2 / 84.4 | 54 / 87.0 / 87.0 |
| `suicideideation_passive_ongoing` | 600 | 411 | 0.685 | 1.026 | 77 | 81 / 81.8 / 79.7 | 74 / 80.5 / 82.1 | 77 / 83.1 / 83.1 |
| `selfharm_ongoing` | 703 | 488 | 0.694 | 1.012 | 74 | 92 / 94.6 / 84.3 | 91 / 94.6 / 84.8 | 90 / 93.2 / 84.1 |
| `rape_past` | 352 | 248 | 0.705 | 0.997 | 62 | 58 / 85.5 / 88.3 | 57 / 82.3 / 85.7 | 55 / 82.3 / 87.2 |
| `no_crisis` | 1160 | 1020 | 0.879 | 0.799 | 186 | 160 / 79.6 / 85.5 | 167 / 81.2 / 85.6 | 164 / 81.2 / 86.3 |

## Retention vs. recall difference (14 crisis labels)

Spearman correlation with a two-sided permutation p-value (100,000 permutations, seed 42). Positive rho: the later model loses more recall on labels that lost more posts.

| Recall difference | Spearman rho | Permutation p |
| --- | ---: | ---: |
| SFT Unanimous − SFT Consensus | +0.204 | 0.4839 |
| SFT Unanimous-IPW − SFT Consensus | +0.517 | 0.0614 |
| SFT Unanimous-IPW − SFT Unanimous | +0.245 | 0.3986 |

Exploratory: several labels have fewer than 12 test posts, so per-label recall is noisy.
