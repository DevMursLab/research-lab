# Per-class results + cross-source S3 breakdown

## A. Per-class on S2 (raw, fold 0)
### PapayaFormer
| class | precision | recall | F1 | support | AUROC |
|---|---|---|---|---|---|
| anthracnose | 0.951 | 0.895 | 0.922 | 152 | 0.956 |
| bacterial_leaf_spot | 0.683 | 0.947 | 0.794 | 132 | 0.971 |
| healthy | 0.883 | 0.748 | 0.810 | 111 | 0.982 |
| mite_or_deficiency | 0.878 | 0.911 | 0.894 | 79 | 0.984 |
| powdery_mildew | 1.000 | 0.833 | 0.909 | 30 | 1.000 |
| prsv | 0.912 | 0.865 | 0.888 | 445 | 0.972 |

### MobileViT-S
| class | precision | recall | F1 | support | AUROC |
|---|---|---|---|---|---|
| anthracnose | 0.940 | 0.928 | 0.934 | 152 | 0.989 |
| bacterial_leaf_spot | 0.698 | 0.947 | 0.804 | 132 | 0.971 |
| healthy | 0.851 | 0.514 | 0.640 | 111 | 0.973 |
| mite_or_deficiency | 0.901 | 0.924 | 0.912 | 79 | 0.996 |
| powdery_mildew | 1.000 | 0.833 | 0.909 | 30 | 0.999 |
| prsv | 0.848 | 0.852 | 0.850 | 445 | 0.955 |

## B. Cross-source S3 (train D1+D2 -> test D3)
D3 test class counts: [190, 170, 144, 0, 155, 265]

**PapayaFormer**: acc 5.5%, macro-F1 4.0%; per-class F1 = anthracnose 0.00, bacterial_leaf_spot 0.12, healthy 0.00, mite_or_deficiency 0.00, powdery_mildew 0.00, prsv 0.08
**MobileViT-S**: acc 14.0%, macro-F1 11.3%; per-class F1 = anthracnose 0.25, bacterial_leaf_spot 0.34, healthy 0.00, mite_or_deficiency 0.00, powdery_mildew 0.00, prsv 0.09