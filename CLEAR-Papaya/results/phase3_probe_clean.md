# Phase 3 (clean) — leaky S1 vs group-disjoint S2, raw-only corpus

DINOv2 ViT-S/14 frozen features, 4694 raw closed-set images, 2536 leaf groups. S1 = 5 random splits; S2 = 5-fold StratifiedGroupKFold.

| probe | S1 acc % | S2 acc % | acc inflation | S1 mF1 | S2 mF1 | mF1 infl |
|---|---|---|---|---|---|---|
| knn1 | 82.2 ± 0.5 | 76.9 ± 2.2 | **+5.4** | 84.9 | 77.0 | +8.0 |
| logreg | 80.2 ± 1.0 | 75.9 ± 4.9 | **+4.2** | 80.2 | 74.0 | +6.2 |
