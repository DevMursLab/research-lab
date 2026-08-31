# Licence verification (Task 1.3)

Verified via WebFetch of the primary Mendeley Data landing pages, 2026-08-27.
**Before submission:** re-open each URL, screenshot the licence block, and save the
screenshot next to this file (`licence_D1.png`, ...). A fetched text claim is not a
screenshot.

| ID | Source | Licence (as stated on landing page) | DOI | Verdict |
|----|--------|--------------------------------------|-----|---------|
| D1 | Papaya Leaf Disease Image Dataset — `3kwgxg4stb` | **CC BY 4.0** | 10.17632/3kwgxg4stb.1 | OK — attribution only. Use RAW images only. |
| D2 | BDPapayaLeaf — `p997fvf526` | **CC BY 4.0** | 10.17632/p997fvf526.1 | OK. Also published as Data-in-Brief (Elsevier, `S2352340924008734`) + PMC11460515. |
| D3 | Papaya Diseases Dataset (Sethy) — `yvcwypp8rz` | **CC BY 4.0** | 10.17632/yvcwypp8rz.1 | OK. |
| D4 | Healthy/Unhealthy Papaya Leaf, Bangladeshi Orchards — `44p8v6ywsm` | **CC BY 4.0** | 10.17632/44p8v6ywsm.1 | OK. Use RAW only. Decide: separate source or merge with D1. |
| D5 | CycleGAN-Balanced Papaya — `h56m9vgv95` | CC BY 4.0 | 10.17632/h56m9vgv95.1 | **EXCLUDE** — contains GAN-synthetic images; only 1684 of 7000 are real. Non-standard taxonomy. Cite as related work at most. |
| D6 | Kaggle `ajithdari/papaya-leaf-disease-dataset` | **UNVERIFIED** — Kaggle licence field needs manual check (login) | none | **EXCLUDE unless** (a) a licence is shown AND (b) it mirrors a DOI'd source. Roadmap gate 1.3: no DOI ⇒ not citable. |
| D7 | PlantVillage | public / CC0 on the common mirror (verify) | arXiv:1511.08060 | OOD control only (no papaya). |
| D8 | PlantDoc | CC BY 4.0 (verify on the GitHub repo) | 10.1145/3371158.3371196 | OOD / background-realism probe only (no papaya). |

## Decisions

- **Benchmark papaya sources:** D1, D2, D3 (+ D4 pending a de-dup check against D1).
  All CC BY 4.0 — compatible with every target venue including Elsevier hybrid.
- **D3 is the cross-source (S3) test set** — Indian collection vs. Bangladeshi D1/D2,
  and it is the only one with a proper *Powdery Mildew* class.
- **Excluded:** D5 (synthetic), D6 (no DOI / licence unverified).
- **OOD probes:** D7 PlantVillage, D8 PlantDoc — not papaya, used only for open-set / Clever-Hans.

## Taxonomy reconciliation notes (feeds `src/common.py` LABEL_MAP)

| Raw label seen | Source | Canonical |
|---|---|---|
| Curl / Leaf Curl | D1, D2, D4 | `prsv` (leaf curl is a ringspot-complex symptom) — *confirm with plant pathologist* |
| Mosaic / Papaya Mosaic / Mosaic Virus | D1, D4 | `prsv` |
| Ringspot / Ring Spot | D1, D2, D3, D4 | `prsv` |
| Bacterial Spot / Black Spot | D1, D2, D3 | `bacterial_leaf_spot` |
| Mite / Mite Disease / Mites | D1, D4 | `mite_or_deficiency` |
| Mealybug | D1, D4 | `mite_or_deficiency` (pest, non-lesion) — *or drop; decide* |
| Powdery Mildew | D3 only | `powdery_mildew` |
| Phytophthora | D3 only | **no closed-set class** — exclude these images or treat as open-set |
| Anthracnose | D1, D2, D3 | `anthracnose` |
| Healthy | all | `healthy` |

⚠️ Curl vs Mosaic vs Ringspot all collapsing to `prsv` is a modelling choice that a
reviewer will question. Options: (a) keep them merged and justify from virology
(PRSV complex), (b) keep `prsv` = ringspot only and make `leaf_curl` its own class.
Get the plant-pathology annotator (roadmap Task 3.5) to rule on this before Phase 3.
