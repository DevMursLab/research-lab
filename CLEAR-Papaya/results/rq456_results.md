# RQ4 / RQ5 / RQ6 / latency  (raw corpus, S2 fold 0)

## tab:rq1 extra metrics (2 seeds)
| model | acc | MCC | kappa | AUROC(ovr) |
|---|---|---|---|---|
| resnet50 | 84.9 | 0.786 | 0.785 | 0.978 |
| mobilevit_s | 87.1 | 0.818 | 0.817 | 0.984 |
| PapayaFormer | 87.9 | 0.835 | 0.833 | 0.977 |

## RQ4 corruption robustness
| model | clean | mean corruption acc | relative robustness |
|---|---|---|---|
| resnet50 | 84.1 | 61.7 | 0.733 |
| mobilevit_s | 88.8 | 66.3 | 0.746 |
| PapayaFormer | 88.0 | 53.9 | 0.613 |

## RQ5 open-set (Phytophthora as unseen class)
| model | AUROC | AUPR | FPR@95TPR |
|---|---|---|---|
| resnet50 | 0.711 | 0.325 | 0.845 |
| mobilevit_s | 0.720 | 0.264 | 0.942 |
| PapayaFormer | 0.708 | 0.309 | 0.794 |

## Latency (PapayaFormer FP32, batch 1)
- GPU median: 12.1 ms
- CPU median: 76.7 ms
- Params: 16.8 M

## RQ6 low-data (PapayaFormer)
| train frac | n_train | acc |
|---|---|---|
| 0.05 | 137 | 55.0 |
| 0.10 | 312 | 55.1 |
| 0.25 | 838 | 74.1 |
| 0.50 | 1747 | 83.2 |
| 1.00 | 3745 | 83.4 |