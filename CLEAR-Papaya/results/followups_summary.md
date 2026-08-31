# Follow-up experiments

## A. Encoder-robustness of the leakage rate (raw-only, mutual-kNN)

| encoder | theta_e=0.95 rho | sweep range (0.90 -> 0.97) |
|---|---|---|
| DINOv2 ViT-S/14 (headline) | 0.55 | 0.84 -> 0.44 |
| CLIP ViT-B/32 | 0.46 | 0.90 -> 0.29 |

The leakage rate is not an artefact of one embedding: an independently trained
CLIP encoder recovers a comparable rho at the same threshold.

## B. Second architecture for Table 6 (raw-only, 160px, 12 epochs, best-epoch)

| model | S1 acc | S2 acc | acc inflation | S1 mF1 | S2 mF1 | mF1 infl |
|---|---|---|---|---|---|---|
| ResNet-18 (pilot) | 87.2 | 80.9 | +6.3 | 89.2 | 79.7 | +9.6 |
| ResNet-50 | 88.4 | 84.6 | +3.8 | 90.5 | 88.1 | +2.3 |
