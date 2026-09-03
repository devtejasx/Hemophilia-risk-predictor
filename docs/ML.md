# The MMC2 + MMC3 ML pipeline

Every number on this page was produced by `scripts/train_inhibitor_model.py`
and is reproducible with `python scripts/train_inhibitor_model.py` (seed 42).
Nothing here is copied from a report or estimated.

The previous CHAMP model's numbers are **not** comparable to these and are not
reproduced anywhere on this page. They remain in
`archive/legacy-champ/champ-v1/metrics.json`, clearly labelled.

## What the model predicts

The probability that a **reported clinical record** of an F8 mutation reports
inhibitor development — MMC3's `Inhibitors` field, `Yes` = 1, `No` = 0.

A row of the training data is one clinical report, not one patient and not one
mutation. Several reports may describe the same mutation. The output is
therefore attributable to the record's mutation-and-clinical description, and is
not an individual patient's probability of developing an inhibitor. Any wording
that implies otherwise is a bug.

## Dataset

Two supplementary tables of the source publication, read only through
`ml/preprocessing/hemophilia_a.py`. The files on disk are never edited.

| | MMC2 | MMC3 |
|---|---|---|
| File | `BVTH_VTH-2024-000215-mmc2.csv` | `BVTH_VTH-2024-000215-mmc3.csv` |
| Rows | 6,211 | 10,064 |
| Columns | 34 | 23 |
| Unique `mut_id` | 6,211 | 6,212 |
| Repeated `mut_id` | 0 | 3,852 |
| Duplicate rows | 0 | 0 |
| Missing values | 36,818 | 110,424 |
| Grain | one **mutation** | one **clinical record** |

`python scripts/validate_dataset.py` prints all of this, plus the per-feature
missing rates and the resolved feature sets, before anything is fitted.

### Target

`Inhibitors` is free text. Only an explicit yes/no is a label.

| `Inhibitors` value | Records | Outcome |
|---|---|---|
| `No` (incl. `No ` ) | 4,130 | **0** |
| `Yes` (incl. `yes`) | 836 | **1** |
| `Not reported` | 2,090 | excluded |
| *(blank)* | 1,730 | excluded |
| `Not` | 1,276 | excluded |
| `Severe`, `Mild` | 2 | excluded |

**4,966 labelled records, 836 positive (16.83%).** 5,098 excluded. Labelled +
excluded = 10,064; the suite asserts this, so no record can be dropped without
appearing in the exclusion report. Nothing is imputed — a severity typed into
the inhibitor field is not evidence either way.

### The join

```
MMC2  ── filter g_name == "F8"  (6,205 of 6,211; six F9 rows dropped)
      ── one row per mut_id
                 │
                 └── inner join on mut_id ──►  4,962 merged records
                 │                             2,639 unique mutations
MMC3  ── keep only explicit Yes/No  (4,966)
      ── keep EVERY clinical record
```

| | |
|---|---|
| Merged records | **4,962** |
| Unique `mut_id` | **2,639** |
| Clinical records dropped (mutation not in the F8 table) | 4 |
| Duplicate rows after merge | 0 |
| Most records for one mutation | 104 |
| Mutations whose records disagree about the outcome | 124 |
| Positive | 836 (16.85%) |

The join is deliberately asymmetric. MMC2 is reduced to one row per mutation
first, so the join cannot multiply clinical records; MMC3 is **not**
de-duplicated, because several patients genuinely carry the same mutation and
their reports are separate observations. The 124 mutations whose records
disagree are reported, not removed — that disagreement is data.

## Features

Resolved from the columns that actually exist, never assumed. A candidate that
the files do not carry, or that survives the merge only as a suffixed duplicate,
is dropped and recorded in the artifact's `features.dropped`.

### Genomic — 20 columns, from MMC2

`mut_type` · `mut_effect` · `location` · `e_i_numb` · `locnumb` · `aa_numb_old` ·
`aa_numb` · `codon_change` · `codon_first` · `codon_last` · `n_bp` · `nuc_numb` ·
`ntchange` · `mut_syn` · `aa_change` · `aa_first` · `aa_last` · `aa_syn` ·
`CpG` · `utype`

`aa_numb_old` and `aa_numb` are numeric; the rest are categorical.

**`mutations` is dropped.** It exists in *both* tables, so the join renames it to
`mutations_clinical` / `mutations_genomic` and the candidate column `mutations`
is not present in the merged frame. This matches the reference notebook, whose
`valid_features` filter drops it for the same reason.

### Clinical — 9 columns, from MMC3

`clotting` · `discrep` · `ratio` · `assay` · `antigen` · `act/ant` · `type` ·
`pa_race` · `cli_phe`

`discrep`, `ratio` and `act/ant` are numeric; the rest are categorical.

**`bleed_tool` and `bleed_score` are dropped.** Both columns exist but are 100%
null across all 4,962 merged records, so they carry no information. Again this
matches the notebook's `nunique > 1` filter.

### Merged — 29 columns

The union of the two, used by the default model.

### Deliberately excluded

Recorded in code as `hemophilia_a.EXCLUDED_COLUMNS` so the reasoning travels
with the pipeline. The most important:

- **`uinhibitor` — leakage.** MMC2's own curated inhibitor status for the
  mutation. 97.1% of merged records with `uinhibitor = "Yes"` are positive and
  0.2% of `uinhibitor = "No"` are: it reproduces the label almost exactly. This
  is the notebook's exclusion, kept and documented.
- **`reference`, `ref_id` — leakage.** The source publication. Grouping by
  publication recovers the label, because inhibitor-focused papers report
  inhibitor-positive cases.
- `uclotting`, `udiscrep`, `uratio`, `uantigen`, `useverity`, `ucomments` —
  curated mirrors of the clinical block, from the same curated table as
  `uinhibitor`.
- `pa_id`, `all_id`, `g_id`, `d_id` — identifiers.
- `comments`, `pri_comments` — free text.
- `pub_lab`, `date added`, `Date added to HADB`, `Count_mut_id`, `subgroups` —
  reporting artifacts.
- `mut_id` itself — the grouping key, never a predictor.

The training script prints a leakage probe (weighted |class rate − base rate|)
over every feature plus `uinhibitor`, so a newly leaky column would be visible
rather than silent.

## Preprocessing

One `ColumnTransformer` per feature set, **fitted on the training split only**.
The same fitted object is serialised and used at inference, so training-time and
serving-time preprocessing cannot differ.

| Column type | Treatment |
|---|---|
| Categorical | explicit `Unknown` fill → one-hot, `min_frequency=3`, `handle_unknown="infrequent_if_exist"` |
| Numeric | median impute **with a missing indicator** → standardise |

Before that, two normalisations run over the merged frame:

1. **Whitespace stripping.** The files contain both `"Point"` and `"Point "`,
   `"Exon"` and `"Exon "`, `"Severe"` and `"Severe "`.
2. **Case-only spelling variants folded onto the dominant spelling** — 18 of
   them, e.g. `missense` and `MIssense` → `Missense` (2,674 occurrences),
   `severe` → `Severe` (2,700). Every collapse is listed by
   `scripts/validate_dataset.py`; the winner is always a spelling that occurs in
   the data. This is not a blanket `.lower()`: none of these columns carries a
   meaning that depends on case.

Encoded widths: genomic 1,225 columns, clinical 146, merged 1,371. Several
genomic columns are near-identifiers of the mutation — `mut_syn` has 1,610
distinct training values — so `min_frequency` folds most of them into an
"infrequent" bucket rather than producing thousands of singleton columns.

### Deviations from the reference notebook

`final(1).ipynb` is the reference for the merge, the feature blocks, the split
and the model families. Five things are done differently, each recorded in
`hemophilia_a.NOTEBOOK_DEVIATIONS` and in every artifact's `metadata.json`:

1. **Missing categoricals become an explicit `Unknown` level**, not the mode.
   `assay` is 99.5% null and `discrep` 99.3%: those measurements were never
   taken, and imputing the mode invents one and destroys the only signal the
   column carries.
2. **Whitespace and case-variant normalisation**, as above. The notebook treats
   `Point` and `Point ` as two categories.
3. **Numeric imputation adds a missing indicator**, so an imputed median stays
   distinguishable from an observed value.
4. **Model selection uses `StratifiedGroupKFold` on `mut_id`**, so
   cross-validation obeys the same grouping constraint as the outer split.
5. **The served model is fitted on the training split alone.** The notebook
   refits on train+validation after choosing a threshold; keeping the fit on
   train means the artifact's preprocessor never saw the rows that chose its
   threshold.

The notebook also selects its threshold by maximising accuracy. This pipeline
maximises F1 — at a 16.8% positive rate, answering "no inhibitor" for everything
scores 83% accuracy and is useless.

## The grouped split

**This is the part that matters most.** A mutation appears in up to 104 clinical
records, and every record of one mutation shares an identical genomic block. A
plain row-level shuffle would put some of those records in training and the rest
in test, and the model would be scored on mutations it had memorised.

So `mut_id` is the grouping key for both splits — `GroupShuffleSplit`, 80/20 for
train+val / test, then 80/20 again for train / val, exactly as in the notebook's
cell 5.

```
load MMC2 + MMC3 -> validate -> filter F8 -> encode target
     -> merge on mut_id
     -> GROUPED SPLIT FIRST
     -> fit preprocessor on TRAIN ONLY
     -> model selection, StratifiedGroupKFold(5) on mut_id
     -> isotonic calibration on train
     -> threshold on the VALIDATION split
     -> evaluate ONCE on the held-out test split
```

| | Records | Mutations | Positive |
|---|---|---|---|
| Training | **3,124** | **1,688** | 525 |
| Validation | **812** | **423** | 142 |
| Test | **1,026** | **528** | 169 |
| Total | 4,962 | 2,639 | 836 |

Group overlap, asserted at training time and again by the test suite:

| Pair | Shared `mut_id` |
|---|---|
| train ∩ validation | **0** |
| train ∩ test | **0** |
| validation ∩ test | **0** |

`assert_no_group_overlap` raises rather than continuing, and a test proves that
a plain `train_test_split` on the same frame *would* have leaked.

## Model selection

Cross-validated ROC-AUC on the training split, `StratifiedGroupKFold(5)` grouped
on `mut_id`. Same four families as the previous pipeline — the model was not
replaced, only retrained on the new feature space.

| Model | genomic | clinical | merged |
|---|---|---|---|
| logistic_regression | 0.728 ± 0.055 | 0.742 ± 0.027 | 0.770 ± 0.049 |
| **random_forest** | **0.751 ± 0.061** | **0.746 ± 0.041** | 0.773 ± 0.055 |
| **xgboost** | 0.749 ± 0.062 | 0.745 ± 0.027 | **0.790 ± 0.043** |
| random_forest_smote | 0.750 ± 0.065 | 0.744 ± 0.042 | 0.774 ± 0.054 |

Selected: `random_forest` for genomic and clinical, `xgboost` for merged. Each is
wrapped in `CalibratedClassifierCV(isotonic)` over the same grouped folds.

As before, the candidates sit within about one standard deviation of each other,
so the choice is not strongly evidenced and a different seed could reorder them.
SMOTE did not help; class weighting already handles the 16.8% positive rate.

## Results — held-out test set, touched once

1,026 records of 528 mutations, 169 positive (16.5%).

| Metric | genomic | clinical | **merged** |
|---|---|---|---|
| Accuracy | 0.7982 | 0.8265 | **0.8012** |
| Precision | 0.4087 | 0.4587 | **0.4229** |
| Recall (sensitivity) | 0.5030 | 0.2959 | **0.5680** |
| Specificity | 0.8565 | 0.9312 | **0.8471** |
| F1 | 0.4509 | 0.3597 | **0.4848** |
| Balanced accuracy | 0.6797 | 0.6135 | **0.7076** |
| MCC | 0.3316 | 0.2733 | **0.3710** |
| ROC-AUC | 0.7327 | 0.7156 | **0.7646** |
| PR-AUC | 0.3911 | 0.4144 | **0.4754** |
| Brier score | 0.1220 | 0.1182 | **0.1126** |
| Threshold | 0.2464 | 0.2800 | **0.2453** |

Confusion matrices at those thresholds:

| | TN | FP | FN | TP |
|---|---|---|---|---|
| genomic | 734 | 123 | 84 | 85 |
| clinical | 798 | 59 | 119 | 50 |
| **merged** | **726** | **131** | **73** | **96** |

**Read this honestly.** The merged model's PR-AUC of 0.475 against a 0.165 base
rate is a 2.9× lift — real signal, and the merged block genuinely beats either
block alone on every ranking metric, which is the notebook's central claim. But
at the F1-optimal threshold it flags 227 records to catch 96 of 169 true
positives: more than one false alarm per true one, and it still misses 73. It is
a screening aid that errs toward sensitivity, not a decision rule.

Accuracy is reported because the brief asks for it, but it is not the headline:
answering "no inhibitor" for every record scores 0.835 here and is useless. The
clinical-only model's *higher* accuracy (0.827) comes with the worst recall
(0.296) of the three — it wins on accuracy by predicting negative more often.

### Threshold

Chosen to maximise F1 on the **validation** split, recorded in `metadata.json`,
and read from there at inference. Nothing in the system assumes 0.5. Moving it
trades recall for precision and is a clinical judgement, not a technical
default.

## Artifacts

```
ml/artifacts/
  mmc-genomic-v1/    model.joblib · preprocessor.joblib · background.npy
  mmc-clinical-v1/   metadata.json · metrics.json
  mmc-merged-v1/     (default)
  training_summary.json
```

`metadata.json` carries the SHA-256 of both source CSVs, the merge and label
reports, the leakage probe, the split summary with its overlap counts, the
threshold, the library versions, the notebook deviations, and the `FeatureSpec`
— the exact categorical/numeric column lists the model was fitted on.

Inference reconstructs its column list from that `FeatureSpec` rather than from
a hardcoded constant, so a served model cannot drift away from the feature set
it was trained on. `ml/artifacts.py` additionally refuses to load a bundle whose
metadata feature count disagrees with the estimator's `n_features_in_`.

`legacy-synthetic-v0` sits alongside them, marked `preserved-not-servable`;
requesting it raises rather than serving a model fitted on 30 fabricated rows.

## Explainability

`ml/explainability/service.py` — one service, both methods, driven by the
bundle's own `FeatureSpec`.

- **SHAP**: exact TreeSHAP over the ensemble underneath the isotonic calibrator.
  The unwrapping handles a `RandomForestClassifier`, an `XGBClassifier` and an
  imblearn `Pipeline` wrapping one, because different feature sets select
  different families. A model-agnostic explainer over the calibrated wrapper is
  far too slow for a request path. Isotonic calibration is monotone, so it
  rescales probabilities without reordering feature effects; the payload reports
  `basis: "uncalibrated ensemble"` so this is never implied to be an attribution
  of the calibrated probability itself.
- **LIME**: local linear approximation over the training background sample,
  with continuous discretisation disabled (the matrix is one-hot and
  standardised, not raw measurements).

Both aggregate the encoded columns back onto the source column they came from,
so an explanation never names `cat__mut_type_Point`, and a genomic-only model
can never name a clinical field. Each contribution carries a human-readable
label and a `supplied` flag: a field the caller left blank may appear — its
absence really did move the estimate — but it is shown as *not reported* with a
null value, never as a value the caller entered.

If an explainer fails, the response says `available: false` with a reason. It
never falls back to something that merely looks like an attribution.

### Global importance — merged model

Mean |SHAP| over the training background sample, top 8:

| Feature | Label |
|---|---|
| `cli_phe` | Clinical severity |
| `pa_race` | Reported population / origin |
| `aa_numb` | Amino-acid position |
| `aa_numb_old` | Amino-acid position (legacy numbering) |
| `nuc_numb` | Nucleotide position |
| `codon_first` | Reference codon |
| `aa_last` | Variant amino acid |
| `mut_type` | Mutation type |

Clinical severity dominating is consistent with the literature and with the raw
crosstabs. `pa_race` ranking second is a **reporting** effect, not a biological
one — see the limitations below.

## Limitations

1. **Record-level, not patient-level.** A row is one reported clinical record.
   Several records may describe the same mutation, and the same patient is not
   identifiable across them (`pa_id` is excluded as an identifier).
2. **Reporting bias.** 5,098 of 10,064 MMC3 records (51%) are excluded for
   having no explicit Yes/No inhibitor value. That exclusion is very unlikely to
   be random — records from inhibitor-focused work are likelier to have the
   field filled — so the 16.8% positive rate describes this dataset's reporting,
   not population incidence, and the model inherits that bias.
3. **`pa_race` is a reporting artifact.** It ranks second in global importance,
   and it is 58.7% missing. What it most likely encodes is which cohorts were
   studied and reported, not a biological effect. It is retained because the
   reference notebook's clinical block includes it, but no causal reading of it
   is defensible.
4. **Repeated genomic blocks.** Every record of one mutation carries an
   identical genomic block. The grouped split prevents that from leaking across
   partitions, but within the training split it means the effective number of
   independent genomic observations is 1,688, not 3,124.
5. **Near-identifier features.** `mut_syn`, `aa_syn` and `nuc_numb` effectively
   name the mutation. Because splits are grouped, they cannot leak the test
   label — an unseen mutation's notation lands in the encoder's infrequent
   bucket — but they consume model capacity and generalise poorly.
6. **No external validation.** One dataset, one grouped split. **No clinical
   validation is claimed.**
7. **Modest discrimination.** ROC-AUC 0.765 with 42% precision at the operating
   threshold. Useful for screening, not for deciding.
8. **Model choice is weakly evidenced.** The four candidates sit within roughly
   one standard deviation of each other in cross-validation.
