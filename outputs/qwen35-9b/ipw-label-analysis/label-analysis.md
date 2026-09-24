# Label retention, IPW weights, and test recall

Retention = train_unanimous count / train_consensus count. IPW weight = normalized training weight of that label (mean 1 over training posts). Test cells are predicted count / recall / F1 (%). Labels are sorted by retention.

| Label | Consensus | Unanimous | Retention | IPW weight | Test support | Consensus | Unanimous | Unanimous-IPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `childabuse_endangerment_ongoing` | 39 | 15 | 0.385 | 1.827 | 9 | 12 / 55.6 / 47.6 | 3 / 22.2 / 33.3 | 6 / 44.4 / 53.3 |
| `suicideideation_active_past` | 60 | 24 | 0.400 | 1.756 | 11 | 16 / 90.9 / 74.1 | 12 / 72.7 / 69.6 | 11 / 81.8 / 81.8 |
| `suicideideation_passive_past` | 14 | 6 | 0.429 | 1.639 | 4 | 5 / 75.0 / 66.7 | 6 / 50.0 / 40.0 | 4 / 50.0 / 50.0 |
| `childabuse_endangerment_past` | 143 | 66 | 0.462 | 1.522 | 31 | 37 / 64.5 / 58.8 | 32 / 58.1 / 57.1 | 34 / 58.1 / 55.4 |
| `rape_ongoing` | 55 | 31 | 0.564 | 1.246 | 11 | 15 / 72.7 / 61.5 | 14 / 72.7 / 64.0 | 14 / 72.7 / 64.0 |
| `sexualharassment_past` | 156 | 90 | 0.577 | 1.218 | 38 | 33 / 71.1 / 76.1 | 39 / 78.9 / 77.9 | 36 / 73.7 / 75.7 |
| `selfharm_past` | 214 | 127 | 0.593 | 1.184 | 30 | 24 / 66.7 / 74.1 | 26 / 66.7 / 71.4 | 26 / 73.3 / 78.6 |
| `sexualharassment_ongoing` | 219 | 131 | 0.598 | 1.175 | 25 | 38 / 100.0 / 79.4 | 31 / 92.0 / 82.1 | 33 / 96.0 / 82.8 |
| `domesticviolence_past` | 267 | 168 | 0.629 | 1.117 | 33 | 30 / 72.7 / 76.2 | 34 / 84.8 / 83.6 | 35 / 84.8 / 82.4 |
| `domesticviolence_ongoing` | 233 | 150 | 0.644 | 1.091 | 25 | 37 / 88.0 / 71.0 | 34 / 84.0 / 71.2 | 37 / 92.0 / 74.2 |
| `suicideideation_active_ongoing` | 434 | 282 | 0.650 | 1.081 | 54 | 54 / 87.0 / 87.0 | 53 / 90.7 / 91.6 | 57 / 90.7 / 88.3 |
| `suicideideation_passive_ongoing` | 600 | 411 | 0.685 | 1.026 | 77 | 82 / 84.4 / 81.8 | 81 / 88.3 / 86.1 | 80 / 84.4 / 82.8 |
| `selfharm_ongoing` | 703 | 488 | 0.694 | 1.012 | 74 | 90 / 98.6 / 89.0 | 91 / 94.6 / 84.8 | 88 / 98.6 / 90.1 |
| `rape_past` | 352 | 248 | 0.705 | 0.997 | 62 | 60 / 88.7 / 90.2 | 63 / 88.7 / 88.0 | 63 / 88.7 / 88.0 |
| `no_crisis` | 1160 | 1020 | 0.879 | 0.799 | 186 | 165 / 82.3 / 87.2 | 159 / 80.1 / 86.4 | 158 / 79.6 / 86.0 |

## Retention vs. recall difference (14 crisis labels)

Spearman correlation with a two-sided permutation p-value (100,000 permutations, seed 42). Positive rho: the later model loses more recall on labels that lost more posts.

| Recall difference | Spearman rho | Permutation p |
| --- | ---: | ---: |
| Unanimous − Consensus | +0.567 | 0.0380 |
| Unanimous-IPW − Consensus | +0.567 | 0.0368 |
| Unanimous-IPW − Unanimous | -0.300 | 0.2962 |

Exploratory: several labels have fewer than 12 test posts, so per-label recall is noisy.
