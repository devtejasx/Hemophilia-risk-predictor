# The CHAMP ML pipeline

Every number on this page was produced by `scripts/train_champ.py` and is
reproducible with `python scripts/train_champ.py` (seed 42). Nothing here is
copied from an earlier report or estimated.

## What the model predicts

The probability that an F8 variant has a **reported history of inhibitor
development** in the CHAMP registry.

CHAMP rows are variants, not patients. `History of Inhibitor` aggregates the
reports for a variant. The output is therefore **variant-attributable risk** and
is not an individual patient's probability of developing an inhibitor. Any
wording that implies otherwise is a bug.

## Dataset

`ml/data/champ.csv` — CDC Hemophilia A Mutation Project. The file is never
edited; every normalisation happens in `ml/preprocessing/champ.py`.

| | rows |
|---|---|
| In file | 4,050 |
| Labelled (`Yes`/`No`) | **2,296** |
| — positive (`Yes`) | 461 (20.08%) |
| Excluded: `Not reported` | 1,742 |
| Excluded: blank label | 12 |

Labelled + excluded = 4,050. The suite asserts this, so no record can be dropped
without appearing in the exclusion report.

## Features

Nine source columns, expanding to 47 encoded columns.

| Source column | Type |
|---|---|
| `Variant Type` | categorical |
| `Mechanism` | categorical |
| `Domain` | categorical |
| `Subtype` | categorical |
| `In Poly A` | categorical |
| `Reported Clinical Severity` | categorical |
| `exon_number` | numeric, derived from `Exon` |
| `codon_number` | numeric, derived from `Codon` |
| `is_intron` | binary, derived from `Exon` |

`Exon` is a mixed string column holding both `"14"` and `"Intron 22"`, so it is
split into a number and an intron flag. (The previous pipeline tested
`isinstance(x, str) and 'intron' in x` against an integer column, which made its
`exon_risk` feature constant `1.0` for every row.)

### Deliberately excluded

Recorded in code as `champ.EXCLUDED_COLUMNS` so the reasoning travels with the
pipeline:

- **`Reference Number` — leakage.** This identifies the source publication, not
  the variant. Grouped by reference, some studies are 100% inhibitor-positive
  and others 0%, because inhibitor-focused papers report inhibitor-positive
  variants. Including it would let the model recover the label from the
  citation. Found during Phase 1 and excluded before any model was fitted, so
  no reported result is affected.
- `HGVS cDNA`, `hg19 Coordinates`, `HGVS Protein`, `Mature Protein` — near-unique
  identifiers (2,296 distinct values across 2,296 labelled rows).
- `Year Reported` — a reporting artifact, and the column contains an
  out-of-range `0.0`.
- The four FVIII-level columns — 45–90% null and superseded by
  `Reported Clinical Severity`.
- `Comments` (~96% null), `Newly Added in the Current Version` (bookkeeping).

### One normalisation that matters

`Domain` is **not** case-folded. In FVIII nomenclature the lowercase `a1`, `a2`
and `a3` are the *acidic regions*, biologically distinct from the `A1`/`A2`/`A3`
domains. CHAMP contains both spellings (A1 750 rows vs a1 34; A2 727 vs a2 19;
A3 655 vs a3 43). Blanket case-folding would silently merge 96 acidic-region
rows into the A-domains. Columns where case really is just a typo — `Mechanism`
(`deletion`, `DuplIcation`), `Subtype`, `In Poly A` (`n`), `Reported Clinical
Severity` (`severe`) — are normalised via an explicit frozen map.

## Pipeline order

Split happens before anything is fitted. This is the fix for the audit's
finding B-05, where median/mode imputation ran over the whole dataset before
`train_test_split`.

```
load -> validate -> normalise -> label
     -> SPLIT           train 1468 | validation 368 | test 460
     -> fit preprocessor on TRAIN ONLY
     -> model selection, 5-fold CV, SMOTE inside folds only
     -> isotonic calibration on train
     -> threshold chosen on the VALIDATION split
     -> evaluate ONCE on the held-out test set
```

## Model selection

5-fold cross-validated ROC-AUC on the training split:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **random_forest** | **0.757 ± 0.026** | **0.499** |
| random_forest + SMOTE | 0.753 ± 0.028 | 0.489 |
| logistic_regression | 0.753 ± 0.030 | 0.482 |
| xgboost | 0.746 ± 0.022 | 0.479 |

Selected: `random_forest`, wrapped in `CalibratedClassifierCV(isotonic, cv=5)`.
The four candidates are within one standard deviation of each other — the choice
is not strongly evidenced, and a different seed could reorder them.

SMOTE did not help. Class weighting already handles the 20% positive rate.

## Results — held-out test set, touched once

460 variants, 92 positive.

| Metric | Value |
|---|---|
| ROC-AUC | **0.723** |
| PR-AUC (average precision) | **0.501** |
| Brier score | 0.134 |
| Precision @ threshold | 0.311 |
| Recall @ threshold | 0.772 |
| F1 @ threshold | 0.444 |

Confusion matrix at threshold 0.175: TN 211, FP 157, FN 21, TP 71.

**Read this honestly.** PR-AUC 0.50 against a 0.20 base rate is a 2.5× lift over
chance — real, useful signal. But at the F1-optimal threshold the model flags 228
variants to catch 71 of 92 true positives: roughly two false alarms for every
true one. It is a screening aid that errs toward sensitivity, not a decision
rule.

Accuracy is deliberately not a headline metric: predicting "no inhibitor" for
everything scores 0.80 on this data and is useless.

### Threshold

**0.175**, chosen to maximise F1 on the validation split, recorded in
`metadata.json`, and read from there at inference. Nothing in the system assumes
0.5. Moving it trades recall for precision and is a clinical judgement, not a
technical default.

## Artifacts

```
ml/artifacts/champ-v1/
  model.joblib          calibrated estimator
  preprocessor.joblib   ColumnTransformer fitted on the training split
  background.npy        200 training rows, for SHAP/LIME
  metadata.json         dataset SHA-256, feature names, threshold, versions, limitations
  metrics.json          everything on this page
```

`ml/artifacts.py` refuses to load a bundle whose metadata feature count
disagrees with the estimator's `n_features_in_` — the specific check that would
have caught the old serving path feeding a 20-feature model 9 columns of zeros.

`legacy-synthetic-v0` sits alongside it, marked `preserved-not-servable`.
Requesting it raises with an explanation rather than serving a model fitted on
30 fabricated rows.

## Explainability

`ml/explainability/service.py` — one service, both methods.

- **SHAP**: exact TreeSHAP over the ensemble underneath the isotonic calibrator.
  ~0.4s per explanation. A model-agnostic permutation explainer over the
  calibrated wrapper took ~18s, which is unusable on a request path. Isotonic
  calibration is monotone, so it rescales probabilities without reordering
  feature effects; the payload reports `basis: "uncalibrated ensemble"` so this
  is never implied to be an attribution of the calibrated probability itself.
- **LIME**: local linear approximation over the same background sample.

Both aggregate the 47 encoded columns back onto the 9 source CHAMP columns, so
an explanation never names `cat__Variant Type_Missense`, and never names a
feature the caller did not supply. If an explainer fails, the response says
`available: false` with a reason — it never falls back to something that merely
looks like an attribution.

### Global importance

Mean |SHAP| over the training background:

| Feature | Importance |
|---|---|
| Variant Type | 0.095 |
| Reported Clinical Severity | 0.090 |
| Domain | 0.052 |
| Mechanism | 0.044 |
| codon_number | 0.041 |
| Subtype | 0.038 |
| exon_number | 0.028 |
| In Poly A | 0.003 |
| is_intron | 0.002 |

Consistent with the raw crosstabs: large structural changes are 56%
inhibitor-positive against 8.8% for missense.

## Limitations

1. **Variant-level, not patient-level.** See the top of this page.
2. **Reporting bias.** 1,754 rows (43%) are excluded for having no reported
   inhibitor history, almost certainly non-randomly — variants studied in
   inhibitor-focused work are likelier to have the field filled. The 20.08%
   positive rate describes this registry's reporting, not population incidence,
   and the model inherits that bias.
3. **Registry-reported severity.** `Reported Clinical Severity` is a strong
   predictor and is itself a reported field, with its own inconsistencies
   (`Mild/Moderate/Severe` appears as a category).
4. **No external validation.** Trained and tested on one registry with a random
   split. There is no independent cohort, and no clinical validation is claimed.
5. **Model choice is weakly evidenced.** The four candidates sit within one
   standard deviation of one another.
6. **Genomic only.** There is no authorised clinical dataset in this repository.
   No genomic + clinical fusion is performed or claimed.
