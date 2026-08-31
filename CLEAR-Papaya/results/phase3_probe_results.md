# Phase 3 — frozen-feature probe: leaky (S1) vs clean (S2)

DINOv2 ViT-S/14 features, 5 seeds, 6709 closed-set images.

| probe | S1 acc % | S2 acc % | acc inflation | S1 mF1 | S2 mF1 | mF1 inflation |
|---|---|---|---|---|---|---|
| knn1 | 86.9 ± 0.6 | 74.8 ± 8.2 | **+12.2** | 88.1 | 73.4 | +14.7 |
| logreg | 82.6 ± 0.7 | 74.2 ± 8.0 | **+8.5** | 82.1 | 72.3 | +9.8 |

S1 = image-level random split (near-duplicates leak across train/test).
S2 = group-disjoint split (same physical leaf never on both sides).
The inflation column is the accuracy a naive protocol reports but a clean one does not.
