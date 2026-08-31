# Significance run: 3 seeds x 5 folds S2

| condition | model | runs | acc | mF1 | ECE | Brier |
|---|---|---|---|---|---|---|
| raw_only | resnet50 | 15 | 84.6±4.8 | 84.4 | 0.166 | 0.291 |
| raw_only | tf_efficientnet_b0 | 15 | 81.5±4.0 | 78.7 | 0.169 | 0.328 |
| raw_only | mobilevit_s | 15 | 87.4±4.4 | 88.3 | 0.161 | 0.247 |
| raw_only | PapayaFormer | 15 | 87.2±4.7 | 87.7 | 0.190 | 0.236 |

## PapayaFormer vs baselines (paired, Holm)
### raw_only  (n=15 paired folds)
- vs tf_efficientnet_b0: t p=0.0000 (Holm 0.0000, sig); Wilcoxon p=0.0001
- vs resnet50: t p=0.0001 (Holm 0.0002, sig); Wilcoxon p=0.0003
- vs mobilevit_s: t p=0.4576 (Holm 0.4576, n.s.); Wilcoxon p=0.7299