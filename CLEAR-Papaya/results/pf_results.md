# PapayaFormer + baselines (224px, GPU, same folds)

| condition | model | S1 acc | S2 acc | Δ | S2 mF1 | S2 ECE | S2 Brier |
|---|---|---|---|---|---|---|---|
| raw_only | resnet50 | 89.8±0.4 | 84.8±3.2 | +5.0 | 83.9 | 0.163 | 0.283 |
| raw_only | tf_efficientnet_b0 | 86.6±0.6 | 80.6±2.5 | +6.0 | 77.3 | 0.171 | 0.338 |
| raw_only | mobilevit_s | 91.6±0.1 | 86.8±2.7 | +4.7 | 87.3 | 0.149 | 0.258 |
| raw_only | PapayaFormer | 90.7±0.0 | 88.3±1.4 | +2.4 | 88.3 | 0.180 | 0.219 |
| as_published | resnet50 | 93.4±0.6 | 81.4±1.5 | +12.0 | 80.2 | 0.171 | 0.354 |
| as_published | tf_efficientnet_b0 | 89.9±0.7 | 78.0±1.9 | +12.0 | 74.2 | 0.173 | 0.378 |
| as_published | mobilevit_s | 94.4±1.0 | 86.8±2.9 | +7.7 | 87.1 | 0.180 | 0.283 |
| as_published | PapayaFormer | 92.9±0.3 | 86.8±2.8 | +6.1 | 86.9 | 0.131 | 0.228 |

## PapayaFormer vs baselines on S2 (paired t, Holm)
### raw_only
- vs tf_efficientnet_b0: p=0.0396, p_holm=0.1189, n.s.
- vs resnet50: p=0.1904, p_holm=0.3808, n.s.
- vs mobilevit_s: p=0.4247, p_holm=0.4247, n.s.
### as_published
- vs tf_efficientnet_b0: p=0.0121, p_holm=0.0362, significant
- vs resnet50: p=0.0718, p_holm=0.1435, n.s.
- vs mobilevit_s: p=0.7251, p_holm=0.7251, n.s.