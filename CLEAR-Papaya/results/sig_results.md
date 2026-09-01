# Significance run: 5 seeds x 5 folds S2

| condition | model | runs | acc | mF1 | ECE | Brier |
|---|---|---|---|---|---|---|
| raw_only | resnet50 | 25 | 84.4±4.9 | 84.1 | 0.162 | 0.292 |
| raw_only | tf_efficientnet_b0 | 25 | 80.8±4.8 | 78.1 | 0.164 | 0.338 |
| raw_only | mobilevit_s | 25 | 87.1±5.2 | 87.9 | 0.162 | 0.244 |
| raw_only | PapayaFormer | 25 | 87.3±4.3 | 88.2 | 0.190 | 0.236 |

## PapayaFormer vs baselines (paired, Holm)
### raw_only  (n=25 paired folds)
- vs tf_efficientnet_b0: t p=0.0000 (Holm 0.0000, sig); Wilcoxon p=0.0000
- vs resnet50: t p=0.0000 (Holm 0.0000, sig); Wilcoxon p=0.0000
- vs mobilevit_s: t p=0.5974 (Holm 0.5974, n.s.); Wilcoxon p=0.7971