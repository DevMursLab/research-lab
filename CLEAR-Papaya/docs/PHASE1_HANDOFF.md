# Phase 1 — what's built, what you do next

Status: **all 11 pipeline scripts written and smoke-tested end-to-end** on a
synthetic dataset (steps 1–5, 7, 9, 10, 11 verified locally; step 6 is GPU/Colab).

## Your action items (Week 1)

**Task 1.1 (inventory) — DONE.** `outputs/dataset_inventory.csv` is filled from the
primary sources. Summary:

| ID | Dataset | DOI | Licence | Role |
|----|---------|-----|---------|------|
| D1 | Papaya Leaf Disease Image Dataset (`3kwgxg4stb`) | 10.17632/3kwgxg4stb.1 | CC BY 4.0 | primary — **raw images only** (3626, not the 18130 augmented) |
| D2 | BDPapayaLeaf (`p997fvf526`) | 10.17632/p997fvf526.1 | CC BY 4.0 | primary (2159) |
| D3 | Papaya Diseases Dataset / Sethy (`yvcwypp8rz`) | 10.17632/yvcwypp8rz.1 | CC BY 4.0 | **cross-source (S3)** — India vs BD; only source with Powdery Mildew (983) |
| D4 | Healthy/Unhealthy Papaya, BD Orchards (`44p8v6ywsm`) | 10.17632/44p8v6ywsm.1 | CC BY 4.0 | extra — raw only (1400); **de-dup vs D1 first** |
| D5 | CycleGAN-Balanced Papaya (`h56m9vgv95`) | 10.17632/h56m9vgv95.1 | CC BY 4.0 | **EXCLUDE** — GAN-synthetic images |
| D6 | Kaggle `ajithdari/...` | none | unverified | **EXCLUDE** — no DOI |
| D7 | PlantVillage | arXiv:1511.08060 | CC0 (verify) | OOD control (no papaya) |
| D8 | PlantDoc | 10.1145/3371158.3371196 | CC BY 4.0 (verify) | OOD / background realism |

**Task 1.3 (licences) — DONE (text-verified).** See `docs/licence_verification.md`.
Before submission you still must **screenshot** each licence page (fetched text ≠ screenshot).

**Now you do:**

1. **Task 1.2 — download** D1, D2, D3 (and D4) into `data/D1/`, `data/D2/`, `data/D3/`,
   `data/D6/`... wait — map them to folders `data/D1 data/D2 data/D3` (the inventory
   IDs; put Sethy in `data/D3`). Layout: `data/<id>/<class_name>/*.jpg` — the parent
   folder name is read as the raw label. **For D1 and D4: copy only the raw-image
   folders, not the augmented ones.** Then `python src/02_verify_integrity.py` and
   fill each `data/<id>/README.md`.
2. **Resolve the taxonomy decision** flagged in `docs/licence_verification.md`:
   right now `src/common.py` merges `curl` + `mosaic` + `ringspot` → `prsv` (PRSV
   complex). A reviewer will question this. Decide with the plant-pathology annotator
   (Task 3.5) whether to keep merged or split `leaf_curl` out. Also: D3's
   `Phytophthora` images have no closed-set class — the manifest will mark them
   `UNMAPPED`; exclude or treat as open-set.
3. **Screenshot** the 8 licence pages into `docs/licence_D*.png`.
4. Run locally, in order:
   ```
   python src/03_build_manifest.py     # check outputs/unmapped_labels.csv — extend LABEL_MAP in src/common.py if needed
   python src/04_dataset_stats.py      # prints LaTeX for manuscript Table tab:dist
   python src/05_hashes.py
   ```

## Week 2 — the gate

5. **On Colab (GPU):** upload the repo, then
   ```
   pip install -r requirements.txt
   python src/06_embeddings.py --model dinov2_vits14
   python src/06_embeddings.py --model clip_vitb32     # optional robustness check
   ```
   Download `outputs/embeddings_dinov2_vits14.npy` (+ its `_index.csv`) back.
6. **Calibrate θ_e:**
   ```
   python src/08_calibrate_theta_e.py            # writes outputs/pairs_to_label.csv
   # open it, view path_a vs path_b, set same_leaf = 1 or 0 for ~100–200 pairs
   python src/08_calibrate_theta_e.py --score    # picks smallest θ_e with precision ≥ 0.95
   ```
7. **Build groups + compute ρ:**
   ```
   python src/07_similarity_graph.py --theta_e <chosen>
   python src/09_leakage_rate.py                 # <-- THE GATE
   python src/10_visualize_clusters.py           # paper Figure 2
   python src/11_group_kfold_split.py            # S2 split, verifies ρ=0 per fold
   ```

## Reading the gate — `outputs/leakage_report.json`

| `mean_rho` | verdict |
|---|---|
| > 0.30 | **PROCEED** — strong headline, go to Phase 3 |
| 0.10–0.30 | moderate — keep going but reframe ("protocol matters even when the gap is modest") |
| < 0.05 | **STOP** — rethink the angle before spending more time |

## Deliverables this phase produces

| File | Roadmap task | Goes into paper |
|---|---|---|
| `outputs/dataset_inventory.csv` | 1.1 | — |
| `outputs/integrity_report.json` | 1.2 | — |
| `outputs/master_manifest.csv` | 1.4 | — |
| `outputs/dataset_stats.json` + printed LaTeX | 1.5 | Table `tab:dist`, `tab:datasets` |
| `outputs/hashes.csv` | 2.1 | — |
| `outputs/embeddings_*.npy` | 2.2 | — |
| `outputs/groups.csv`, `groups_summary.json` | 2.3 | — |
| `outputs/theta_e_calibration.json` | 2.3 | §4.6 sensitivity analysis |
| `outputs/leakage_report.json` | 2.4 | **Table `tab:rq2`**, `tab:splits` S1 row |
| `outputs/duplicate_clusters_top10.png` | 2.5 | **Figure 2** |
| `outputs/splits_S2.csv`, `splits_S2_summary.json` | 2.6 | frozen split for all of Phase 3 |

## Notes / assumptions baked in

- `src/common.py` holds every threshold (`THETA_P=5`, `THETA_E=0.95`, `KNN_K=20`,
  `N_RANDOM_SPLITS=10`) and the label-unification map. Edit there, not in scripts.
- `09` and `11` exclude sources `D4`/`D5` (PlantVillage / PlantDoc) from the headline
  ρ — they're OOD controls, not papaya training data.
- `07` recognises D6 filenames of the form `<leafid>_<n>.jpg` and groups by `<leafid>`
  (native leaf identity overrides inferred grouping — this is how you *validate* the
  audit in Task 3.7).
- `07` has a `--no-embed` fallback (pHash+MD5 only) that runs without a GPU; ρ from it
  is a lower bound. Use the real embeddings for the number you report.
- `faiss-cpu` and `open-clip-torch` are optional; scripts fall back to sklearn / skip CLIP.
