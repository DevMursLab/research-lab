# Phase 3 (real) — ResNet-18 fine-tune, leaky S1 vs clean S2

Raw-only corpus, 4694 closed-set images, 128px, 4 epochs, class-weighted CE.

| protocol | split rho | test acc | test macro-F1 | n_test |
|---|---|---|---|---|
| S1 (image-level random) | 0.543 | 86.2% | 87.1% | 1409 |
| S2 (group-disjoint fold) | 0.000 | 80.7% | 83.9% | 939 |
| **inflation (S1 - S2)** | | **+5.5 pts** | **+3.3 pts** | |

Same architecture and hyper-parameters both rows; the only change is how the
train/test line is drawn. ResNet-18 @128px is a deliberately small CPU-feasible
setup — a full ResNet-50/EfficientNet @224 (where the ~99% literature numbers sit)
is expected to widen this gap, not narrow it.
