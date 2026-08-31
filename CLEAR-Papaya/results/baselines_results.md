# Full-scale baselines (224px, GPU)

| condition | model | S1 acc | S2 acc | acc infl | S1 mF1 | S2 mF1 | mF1 infl |
|---|---|---|---|---|---|---|---|
| raw_only | resnet50 | 90.2±0.2 | 84.4±1.2 | +5.8 | 92.1 | 83.2 | +9.0 |
| raw_only | tf_efficientnet_b0 | 86.2±0.1 | 79.8±3.1 | +6.5 | 85.8 | 76.5 | +9.3 |
| raw_only | mobilevit_s | 91.2±0.0 | 87.3±2.0 | +3.9 | 93.2 | 87.0 | +6.2 |
| as_published | resnet50 | 93.0±0.6 | 82.0±2.7 | +11.1 | 93.7 | 79.2 | +14.5 |
| as_published | tf_efficientnet_b0 | 89.3±0.4 | 77.0±2.1 | +12.3 | 86.3 | 73.3 | +13.0 |
| as_published | mobilevit_s | 94.2±0.6 | 86.0±3.5 | +8.2 | 95.2 | 86.4 | +8.7 |