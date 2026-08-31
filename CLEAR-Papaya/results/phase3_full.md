# Phase 3 (full) — leakage inflation with error bars

ResNet-18 (ImageNet-pretrained), 128px, 6 epochs, class-weighted CE, best-epoch test score.
S1 = 3 random stratified splits; S2 = 3 group-disjoint StratifiedGroupKFold folds. Mean ± sd.

| condition | N | protocol | split ρ | test acc % | macro-F1 % |
|---|---|---|---|---|---|
| as_published | 6709 | S1 (random) | 0.73 | 91.7 ± 0.4 | 92.7 ± 0.1 |
| as_published | 6709 | S2 (group-disjoint) | 0.00 | 80.3 ± 4.3 | 79.8 ± 4.6 |
| **as_published** | | **inflation (S1−S2)** | | **+11.3** | **+13.0** |
| raw_only | 4694 | S1 (random) | 0.54 | 87.2 ± 0.8 | 89.2 ± 1.1 |
| raw_only | 4694 | S2 (group-disjoint) | 0.00 | 80.9 ± 2.5 | 79.7 ± 4.2 |
| **raw_only** | | **inflation (S1−S2)** | | **+6.3** | **+9.6** |

*`as_published`* keeps D1's shipped augmentation images (the condition prior papaya-leaf
work operates in); *`raw_only`* removes them. S2 accuracy is ~80% in **both** conditions — the
true task difficulty is stable; only the leaky S1 number inflates. Roughly half the
as_published inflation (11.3 pt) survives augmentation removal (6.3 pt), i.e. ~half is
augmentation-copy leakage and ~half is same-physical-leaf near-duplicate leakage.

ResNet-18 @128px / 6 epochs is a deliberately small CPU-feasible setup; a full
ResNet-50/EfficientNet @224 (where ~99% literature numbers sit) is expected to widen the gap.