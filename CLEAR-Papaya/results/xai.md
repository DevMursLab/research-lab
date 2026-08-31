# RQ8 faithfulness (mask-free; pointing-game deferred)

| method | Deletion AUC (lower better) | Insertion AUC (higher better) |
|---|---|---|
| Grad-CAM++ (ResNet-50) | 0.367 | 0.640 |
| Grad-CAM++ (PapayaFormer) | 0.436 | 0.627 |
| MSLA mask (intrinsic) | 0.495 | 0.603 |

Attribution stability (Spearman rho under 3% input noise): Grad-CAM++ 0.93, MSLA 0.60.