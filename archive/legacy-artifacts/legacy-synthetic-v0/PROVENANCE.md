# legacy-synthetic-v0 — provenance

**These artifacts were trained on 30 fabricated rows, not on any real dataset.
Do not quote metrics from them.**

They are preserved here byte-for-byte for provenance and comparison only. They are
never the default model and are never loaded unless a caller explicitly asks for
version `legacy-synthetic-v0`.

## What produced them

`archive/legacy-ml/train.py`, which loaded `genomic.csv` + `clinical.csv` (now at
`tests/fixtures/synthetic_genomic.csv` and `tests/fixtures/synthetic_clinical.csv`)
via `archive/legacy-ml/data_fusion.py`.

Those two files are **30 hand-written rows**. They are not clinical data, not a
sample of any registry, and not derived from any patient record.

## Why the metrics are meaningless

In the source fixtures the label is a deterministic function of two input columns:

```
target         0  1        target    0   1
mutation_type              severity
frameshift     7  0        mild      7   0
intron22       0  8        moderate  8   0
missense       8  0        severe    0  15
nonsense       0  7
```

Both columns separate the classes perfectly, so any model fitted here memorises a
rule someone typed by hand. `evaluation_report.json` records the split it came
from: **24 training rows, 6 test rows**, with `"models": {}` — no per-model result
was ever written.

## Known defects in these artifacts

| Artifact | Issue |
|---|---|
| `rf.pkl`, `randomforest.pkl`, `model_ensemble.pkl` | Byte-identical (md5 `1e4ac5601a22968099439900cfb7a12a`). The "best model" is a plain RandomForest. |
| `xgb.pkl`, `xgboost.pkl` | Byte-identical (md5 `05087bbfa3cd123f8dbdefb91a316db7`). |
| `lightgbm.pkl` | Degenerate: one tree, `num_leaves=1`, no splits. Returns a constant. |
| `feature_names.pkl` / `columns.pkl` | 20 features, including `mutation_severity` (a copy-paste duplicate of `mutation_code`) and `exon_risk` (constant `1.0` in all training data). |
| all | Trained with scikit-learn 1.8.0. Loading under a different minor version emits version warnings. |

## Feature space

The 20 features are **not fields of any supplied dataset**. Four of them —
`age_first_treatment`, `dose_intensity`, `exposure_days`, `cumulative_exposure` —
have no counterpart in MMC2 or MMC3 at all. This is why these artifacts cannot
serve the current pipeline and why the `mmc-*-v1` versions exist alongside them
rather than replacing them.

```
exon, age_first_treatment, dose_intensity, exposure_days, mutation_code,
mutation_severity, exon_risk, cumulative_exposure, exposure_risk, age_risk,
severity_code, mutation_severity_interaction, treatment_genetics_interaction,
mutation_type_frameshift, mutation_type_intron22, mutation_type_missense,
mutation_type_nonsense, severity_mild, severity_moderate, severity_severe
```

## Also preserved here

- `X_test.csv` / `y_test.csv` — the 6-row held-out split from that run.
- `evaluation_report.json` — the run's data-split record.
- `shap_values.pkl`, `feature_importance.pkl`, `model_comparison.pkl`,
  `ensemble_hyperparameters.pkl` — downstream outputs of the same run.
