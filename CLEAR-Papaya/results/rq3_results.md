# RQ3 --- Clever-Hans background probe (raw corpus, S2 fold 0)

Segmentation: GrabCut; mean foreground area = nan. Chance = 16.7%.

| model | Acc full | Acc leaf-only | Acc bg-only | Δ_bg (pp) | Δ_leaf (pp) |
|---|---|---|---|---|---|
| mobilevit_s | 86.1 | 57.0 | 64.2 | +47.5 | +29.1 |
| resnet50 | 83.1 | 60.2 | 64.8 | +48.1 | +23.0 |

Δ_bg >> 0 means the background alone predicts the class (dataset defect).
Large Δ_leaf means the model leans on context that will not transfer.