# Clean re-audit — augmentation removal + chaining control

**Augmentation contamination:** 2015 / 6864 images
(29.4%) are augmentation-derived, **all in D1**
({'D1': 2015}). D1 ships these mixed into the
class folders despite being catalogued as raw.

**theta_e sensitivity (mutual-kNN grouping, 10 random 70/15/15 splits, closed-set):**

| scope | theta_e | rho % | n_groups | largest grp | x-label grps | x-source grps |
|---|---|---|---|---|---|---|
| all_mutualknn | 0.90 | 91.2 ± 0.6 | 1026 | 2947 | 60 | 3 |
| all_mutualknn | 0.92 | 85.4 ± 0.8 | 1526 | 2040 | 82 | 3 |
| all_mutualknn | 0.94 | 78.5 ± 1.4 | 2194 | 814 | 126 | 7 |
| all_mutualknn | 0.95 | 72.7 ± 1.4 | 2643 | 283 | 185 | 5 |
| all_mutualknn | 0.96 | 68.9 ± 1.1 | 3003 | 244 | 233 | 1 |
| all_mutualknn | 0.97 | 65.8 ± 1.2 | 3271 | 244 | 243 | 1 |
| rawonly_mutualknn | 0.90 | 84.3 ± 0.8 | 985 | 2520 | 53 | 2 |
| rawonly_mutualknn | 0.92 | 75.2 ± 0.8 | 1482 | 1370 | 72 | 3 |
| rawonly_mutualknn | 0.94 | 62.7 ± 1.4 | 2129 | 697 | 115 | 5 |
| rawonly_mutualknn | 0.95 | 55.0 ± 1.3 | 2566 | 280 | 175 | 2 |
| rawonly_mutualknn | 0.96 | 48.4 ± 2.0 | 2907 | 123 | 221 | 1 |
| rawonly_mutualknn | 0.97 | 43.5 ± 1.8 | 3136 | 120 | 233 | 0 |

**Chaining check (raw-only, theta_e=0.95):**

| grouping | rho % | largest group |
|---|---|---|
| mutual-kNN | 55.0 | 280 |
| single-linkage kNN | 55.0 | 281 |

Single-linkage inflates the largest component by chaining rotate/copy variants;
mutual-kNN is the reported grouping.

**Headline (raw-only, mutual-kNN, theta_e=0.95):
rho = 55.0%**
— down from the contaminated 72.8% but still far above the 30% gate.
