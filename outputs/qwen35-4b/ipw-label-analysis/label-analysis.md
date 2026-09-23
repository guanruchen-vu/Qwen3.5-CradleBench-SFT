# Label retention, IPW weights, and test recall

Retention = train_unanimous count / train_consensus count. IPW weight = normalized training weight of that label (mean 1 over training posts). Test cells are predicted count / recall / F1 (%). Labels are sorted by retention.

| Label | Consensus | Unanimous | Retention | IPW weight | Test support | SFT Consensus | SFT Unanimous | SFT Unanimous-IPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `childabuse_endangerment_ongoing` | 39 | 15 | 0.385 | 1.827 | 9 | 9 / 44.4 / 44.4 | 4 / 33.3 / 46.2 | 3 / 22.2 / 33.3 |
| `suicideideation_active_past` | 60 | 24 | 0.400 | 1.756 | 11 | 12 / 81.8 / 78.3 | 3 / 27.3 / 42.9 | 5 / 45.5 / 62.5 |
| `suicideideation_passive_past` | 14 | 6 | 0.429 | 1.639 | 4 | 4 / 50.0 / 50.0 | 6 / 0.0 / 0.0 | 3 / 25.0 / 28.6 |
| `childabuse_endangerment_past` | 143 | 66 | 0.462 | 1.522 | 31 | 40 / 61.3 / 53.5 | 17 / 38.7 / 50.0 | 14 / 35.5 / 48.9 |
| `rape_ongoing` | 55 | 31 | 0.564 | 1.246 | 11 | 9 / 54.5 / 60.0 | 12 / 63.6 / 60.9 | 11 / 54.5 / 54.5 |
| `sexualharassment_past` | 156 | 90 | 0.577 | 1.218 | 38 | 38 / 71.1 / 71.1 | 42 / 81.6 / 77.5 | 29 / 57.9 / 65.7 |
| `selfharm_past` | 214 | 127 | 0.593 | 1.184 | 30 | 25 / 63.3 / 69.1 | 26 / 63.3 / 67.9 | 26 / 56.7 / 60.7 |
| `sexualharassment_ongoing` | 219 | 131 | 0.598 | 1.175 | 25 | 34 / 92.0 / 78.0 | 25 / 80.0 / 80.0 | 32 / 84.0 / 73.7 |
| `domesticviolence_past` | 267 | 168 | 0.629 | 1.117 | 33 | 32 / 75.8 / 76.9 | 34 / 75.8 / 74.6 | 37 / 75.8 / 71.4 |
| `domesticviolence_ongoing` | 233 | 150 | 0.644 | 1.091 | 25 | 38 / 88.0 / 69.8 | 33 / 80.0 / 69.0 | 30 / 80.0 / 72.7 |
| `suicideideation_active_ongoing` | 434 | 282 | 0.650 | 1.081 | 54 | 55 / 90.7 / 89.9 | 41 / 70.4 / 80.0 | 45 / 77.8 / 84.8 |
| `suicideideation_passive_ongoing` | 600 | 411 | 0.685 | 1.026 | 77 | 79 / 83.1 / 82.1 | 80 / 79.2 / 77.7 | 86 / 85.7 / 81.0 |
| `selfharm_ongoing` | 703 | 488 | 0.694 | 1.012 | 74 | 94 / 95.9 / 84.5 | 80 / 86.5 / 83.1 | 95 / 93.2 / 81.7 |
| `rape_past` | 352 | 248 | 0.705 | 0.997 | 62 | 63 / 88.7 / 88.0 | 59 / 82.3 / 84.3 | 76 / 91.9 / 82.6 |
| `no_crisis` | 1160 | 1020 | 0.879 | 0.799 | 186 | 160 / 79.6 / 85.5 | 177 / 82.3 / 84.3 | 150 / 75.3 / 83.3 |

## Retention vs. recall difference (14 crisis labels)

Spearman correlation with a two-sided permutation p-value (100,000 permutations, seed 42). Positive rho: the later model loses more recall on labels that lost more posts.

| Recall difference | Spearman rho | Permutation p |
| --- | ---: | ---: |
| SFT Unanimous − SFT Consensus | +0.319 | 0.2647 |
| SFT Unanimous-IPW − SFT Consensus | +0.763 | 0.0023 |
| SFT Unanimous-IPW − SFT Unanimous | +0.290 | 0.3106 |

Exploratory: several labels have fewer than 12 test posts, so per-label recall is noisy.
