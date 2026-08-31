# extra1: classical ML / cross-source S3 / temperature scaling

## A. Classical ML (raw S2, 3 folds)
- SVM-RBF: acc 76.2±2.1, mF1 73.5
- RandomForest: acc 71.9±1.9, mF1 67.5

## B. Cross-source S3 (train D1+D2 -> test D3)
- mobilevit_s: acc 25.3, mF1 15.4 (n=924)
- PapayaFormer: acc 22.4, mF1 16.7 (n=924)

## C. Temperature scaling
| model | T | ECE before | ECE after | Brier before | Brier after |
|---|---|---|---|---|---|
| resnet50 | 0.54 | 0.077 | 0.012 | 0.049 | 0.042 |
| mobilevit_s | 0.46 | 0.078 | 0.007 | 0.048 | 0.040 |
| PapayaFormer | 3.08 | 0.156 | 0.381 | 0.083 | 0.239 |