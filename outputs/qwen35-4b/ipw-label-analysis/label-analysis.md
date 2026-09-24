# Label retention, IPW weights, and test recall

Retention = train_unanimous count / train_consensus count. IPW weight = normalized training weight of that label (mean 1 over training posts). Test cells are predicted count / recall / F1 (%). Labels are sorted by retention.

| Label | Consensus | Unanimous | Retention | IPW weight | Test support | Consensus | Unanimous | Unanimous-IPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `childabuse_endangerment_ongoing` | 39 | 15 | 0.385 | 1.827 | 9 | 9 / 44.4 / 44.4 | 4 / 22.2 / 30.8 | 5 / 22.2 / 28.6 |
| `suicideideation_active_past` | 60 | 24 | 0.400 | 1.756 | 11 | 12 / 81.8 / 78.3 | 12 / 90.9 / 87.0 | 10 / 72.7 / 76.2 |
| `suicideideation_passive_past` | 14 | 6 | 0.429 | 1.639 | 4 | 4 / 50.0 / 50.0 | 2 / 25.0 / 33.3 | 4 / 50.0 / 50.0 |
| `childabuse_endangerment_past` | 143 | 66 | 0.462 | 1.522 | 31 | 40 / 61.3 / 53.5 | 32 / 51.6 / 50.8 | 35 / 64.5 / 60.6 |
| `rape_ongoing` | 55 | 31 | 0.564 | 1.246 | 11 | 9 / 54.5 / 60.0 | 16 / 63.6 / 51.9 | 18 / 72.7 / 55.2 |
| `sexualharassment_past` | 156 | 90 | 0.577 | 1.218 | 38 | 38 / 71.1 / 71.1 | 40 / 73.7 / 71.8 | 37 / 68.4 / 69.3 |
| `selfharm_past` | 214 | 127 | 0.593 | 1.184 | 30 | 25 / 63.3 / 69.1 | 24 / 50.0 / 55.6 | 23 / 53.3 / 60.4 |
| `sexualharassment_ongoing` | 219 | 131 | 0.598 | 1.175 | 25 | 34 / 92.0 / 78.0 | 29 / 80.0 / 74.1 | 33 / 92.0 / 79.3 |
| `domesticviolence_past` | 267 | 168 | 0.629 | 1.117 | 33 | 32 / 75.8 / 76.9 | 37 / 78.8 / 74.3 | 33 / 78.8 / 78.8 |
| `domesticviolence_ongoing` | 233 | 150 | 0.644 | 1.091 | 25 | 38 / 88.0 / 69.8 | 33 / 80.0 / 69.0 | 38 / 88.0 / 69.8 |
| `suicideideation_active_ongoing` | 434 | 282 | 0.650 | 1.081 | 54 | 55 / 90.7 / 89.9 | 54 / 88.9 / 88.9 | 55 / 88.9 / 88.1 |
| `suicideideation_passive_ongoing` | 600 | 411 | 0.685 | 1.026 | 77 | 79 / 83.1 / 82.1 | 83 / 85.7 / 82.5 | 82 / 87.0 / 84.3 |
| `selfharm_ongoing` | 703 | 488 | 0.694 | 1.012 | 74 | 94 / 95.9 / 84.5 | 97 / 94.6 / 81.9 | 97 / 94.6 / 81.9 |
| `rape_past` | 352 | 248 | 0.705 | 0.997 | 62 | 63 / 88.7 / 88.0 | 64 / 88.7 / 87.3 | 60 / 83.9 / 85.2 |
| `no_crisis` | 1160 | 1020 | 0.879 | 0.799 | 186 | 160 / 79.6 / 85.5 | 160 / 78.0 / 83.8 | 154 / 77.4 / 84.7 |

## Retention vs. recall difference (14 crisis labels)

Spearman correlation with a two-sided permutation p-value (100,000 permutations, seed 42). Positive rho: the later model loses more recall on labels that lost more posts.

| Recall difference | Spearman rho | Permutation p |
| --- | ---: | ---: |
| Unanimous − Consensus | +0.209 | 0.4733 |
| Unanimous-IPW − Consensus | +0.170 | 0.5604 |
| Unanimous-IPW − Unanimous | -0.180 | 0.5380 |

Exploratory: several labels have fewer than 12 test posts, so per-label recall is noisy.
