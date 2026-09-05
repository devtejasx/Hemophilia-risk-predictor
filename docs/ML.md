# The MMC2 + MMC3 ML pipeline

Every number on this page was produced by `scripts/train_inhibitor_model.py`
and is reproducible with `python scripts/train_inhibitor_model.py` (seed 42).
Nothing here is copied from a report or estimated.

## What the model predicts

The probability that an **F8 mutation** is reported with inhibitor development —
MMC3's `Inhibitors` field, `Yes` = 1, `No` = 0.

A row of the training data is one *mutation*, not one patient. The output is
attributable to the mutation-and-assay description supplied, and describes how
often that mutation is reported with an inhibitor in the literature behind MMC3.
It is **not** an individual patient's probability of developing an inhibitor.
Any wording that implies otherwise is a bug.

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
| Contributes | the genomic block | the clinical block **and the target** |

`python scripts/validate_dataset.py` prints all of this, plus the per-feature
missing rates and the resolved feature sets, before anything is fitted.

### Target

`Inhibitors` is free text. Only an explicit yes/no is a label.

| `Inhibitors` value | Records | Outcome |
|---|---|---|
| `No` (incl. `No `) | 4,130 | **0** |
| `Yes` (incl. `yes`) | 836 | **1** |
| `Not reported` | 2,090 | excluded |
| *(blank)* | 1,730 | excluded |
| `Not` | 1,276 | excluded |
| `Severe`, `Mild` | 2 | excluded |

**4,966 labelled records, 836 positive (16.83%).** 5,098 excluded. Labelled +
excluded = 10,064; the suite asserts this, so no record can be dropped without
appearing in the exclusion report. Nothing is imputed — a severity typed into
the inhibitor field is not evidence either way.

## Fusion

```
MMC2  ── filter g_name == "F8"  (6,205 of 6,211; six F9 rows dropped)
      ── one row per mut_id
                 │
                 └── inner join on mut_id ──►  4,962 merged records
                 │                             2,639 unique mutations
MMC3  ── keep only explicit Yes/No  (4,966)          │
      ── keep EVERY clinical record                  │
                                                     ▼
                              AGGREGATE per mut_id ──►  2,639 mutations
                                                     │
                              drop 124 conflicting ──►  2,515 modelled
```

| | |
|---|---|
| Merged records | 4,962 |
| Clinical records dropped (mutation not in the F8 table) | 4 |
| Duplicate rows after merge | 0 |
| Unique mutations with a label | **2,639** |
| Most clinical records for one mutation | 104 |
| Mean clinical records per mutation | 1.88 |

### Why aggregation happens before anything else

A mutation appears in up to 104 clinical records, and every record of one
mutation carries an **identical genomic block**. Modelling raw records means a
single well-studied mutation contributes 104 near-duplicate rows, and a model
scores well by recognising the mutation rather than by learning anything about
inhibitor risk.

So MMC3's records are collapsed to one row per `mut_id` *before* the split.
After that, a mutation is one row and cannot be split across train and test even
in principle. The grouped split and its assertion are kept anyway — they are what
would catch a future change that reverts to record-level rows.

### The mutation census

| | Mutations |
|---|---|
| F8 mutations in MMC2 | 6,205 |
| …with no usable inhibitor label at all | **3,566** |
| …with a label | **2,639** |
| — positive (every record reports an inhibitor) | **496** |
| — negative (no record reports one) | **2,019** |
| — conflicting (records disagree) → **excluded** | **124** |
| **Final modelling population** | **2,515** |
| Positive rate among modelled mutations | **19.72%** |

**Conflicting mutations are excluded, not resolved by majority vote.** When two
patients carry the same mutation and one develops an inhibitor while the other
does not, the data does not contain a single answer. Voting would manufacture a
certainty the source does not support, and it would systematically resolve
toward the majority class. They are counted and reported instead.

This is a real cost, and it cuts the other way from most exclusions: the 124
conflicting mutations are plausibly the *hardest* cases, so the modelled
population is easier than the whole. See limitation 3.

## Features

Resolved from the columns that actually exist, never assumed.

### Genomic — 14 features, from MMC2

`mut_type` · `mut_effect` · `location` · `e_i_numb` · `locnumb` · `aa_numb` ·
`codon_first` · `codon_last` · `n_bp` · `nuc_numb` · `ntchange` · `aa_first` ·
`aa_last` · `CpG`

`e_i_numb`, `locnumb`, `aa_numb`, `n_bp` and `nuc_numb` are **parsed as
measurements** rather than one-hot encoded — an exon number is genuinely between
its neighbours, and `n_bp` records large deletions with unit suffixes (`7kb`).
The rest are categorical.

### Clinical — 33 features, aggregated from 6 MMC3 fields

Five measurements — `clotting`, `antigen`, `ratio`, `act/ant`, `discrep` — each
become five features:

| Feature | Meaning |
|---|---|
| `<field>_mean` · `_median` · `_min` · `_max` | the distribution of readings for this mutation |
| `<field>_censored_rate` | share of readings given as a bound or range rather than a point value |

Plus the severity phenotype `cli_phe`:

| Feature | Meaning |
|---|---|
| `severity_score_mean` · `_min` · `_max` | ordinal score (Mild 1, Moderate 2, Severe 3; combinations score between) |
| `severity_prop_mild` · `_moderate` · `_severe` | share of this mutation's records in each bucket |
| `severity_n_distinct` | how many distinct severities were reported |
| `cli_phe_mode` | the dominant reported severity (categorical) |

A mutation with one clinical record gets the same features as one with 104; only
the aggregates differ. That is what lets inference build an identical row from a
single record.

### Merged — 47 features

The union of the two, used by the default model.

### Censored and range measurements are parsed, not treated as text

This is the single largest change to the clinical block. `clotting` — residual
factor VIII activity, and the most informative clinical field there is — writes
**4,237 of its 6,492 non-null entries** as a bound or a range:

| Raw value | Parsed | Censoring |
|---|---|---|
| `<1` | 1.0 | left (below the assay's detection limit) |
| `>5` | 5.0 | right |
| `1 to 5`, `1--5`, `48-53` | midpoint | range |
| `13 (post)` | 13.0 | observed |
| `7kb`, `>55kb` | 7000.0, 55000.0 | observed, right |
| `-860` | −860.0 | observed — a negative number, **not** a range |
| `Not reported`, `?`, `Jul-13` | *missing* | — |

One-hot encoding these would have discarded the strongest clinical signal in the
dataset and produced a category per distinct string. A bound is kept **as** the
bound rather than halved, and its direction is carried by the censoring rate
feature: `<1` is not evidence of 0.5, it is evidence of "at most 1". Across the
modelled population, 76.3% of clotting readings are censored in some way.

Anything unparseable becomes missing — never a guess.

### Deliberately excluded

Recorded in code as `hemophilia_a.EXCLUDED_COLUMNS`, each with its reason, so the
justification travels with the pipeline. Four groups:

**The label's curated mirrors in MMC2**

- **`uinhibitor` — leakage.** MMC2's own curated inhibitor status. 97.1% of
  merged records with `uinhibitor = "Yes"` are positive and 0.2% of
  `uinhibitor = "No"` are: it reproduces the label almost exactly.
- `uclotting`, `udiscrep`, `uratio`, `uantigen`, `useverity`, `ucomments` —
  curated mirrors of the clinical block, from the same curated table.

**Fields that exist only because inhibitor testing happened**

- **`type` — leakage.** The inhibitor kinetic type (`I` / `II` / `NU`). It is
  recorded only when an inhibitor assay was run, so both its value *and its
  presence* are post-outcome information: 3,652 of 4,962 merged records leave it
  blank, and the non-blank ones carry a different positive rate (NU 22.6%,
  II 6.4%, I 3.8%) against a 16.8% base rate.
- **`utype` — leakage.** MMC2's curated copy of the same field.
- **`assay` — leakage.** Which inhibitor assay produced the reading. 99.5% null,
  and non-null only for tested patients.

**Reporting and administrative bookkeeping**

- **`reference`, `ref_id` — leakage.** The source publication. Grouping by
  publication recovers the label, because inhibitor-focused papers report
  inhibitor-positive cases.
- **`pa_race`.** Nominally an ancestry, in practice the reporting country — it
  repeats `pub_lab` verbatim in most rows. Its positive rate swings from 8.8%
  (India) to 48.7% (Italy), which tracks which cohorts were published rather
  than biology. In the previous record-level pipeline this column ranked
  **second** in global importance; excluding it is why the current model's top
  features are clinical and genomic rather than geographic.
- `pa_id`, `all_id`, `g_id`, `d_id`, `subgroups` — identifiers.
- `comments`, `pri_comments` — free text.
- `pub_lab`, `date added`, `Date added to HADB`, `Count_mut_id` — reporting
  artifacts.
- `n_clinical_records` — how many records back a mutation. A reporting artifact
  like `Count_mut_id`, and after conflicting mutations are excluded it is
  actively misleading: multi-record mutations that survive are unanimous, and
  unanimous-negative far more often than unanimous-positive (22.2% positive at
  one record, 0% above four). It encodes the exclusion rule, not biology.
- `bleed_tool`, `bleed_score`, `mutations` — empty or constant.

**Near-unique identifiers and redundant composites**

- **`mut_syn` — identifier.** The HGVS coding string: 2,509 distinct values
  across 2,639 mutations (95.1% unique). It *names* the mutation rather than
  describing it, so a model given it can score well by memorising the training
  set. Its content is kept as `nuc_numb` and `ntchange`.
- **`aa_syn` — identifier.** The HGVS protein string, 81.4% unique. Same
  reasoning; `aa_first`, `aa_numb` and `aa_last` carry the same content.
- `aa_change`, `codon_change` — redundant composites of columns that are kept
  separately.
- `aa_numb_old` — the pre-2001 amino-acid numbering of `aa_numb`, correlation
  0.999 with it.

This is enforced, not just documented. `hemophilia_a.identifier_like_columns`
refuses any categorical feature whose distinct-value count exceeds
`MAX_CATEGORY_UNIQUENESS_RATIO` (50%) of the rows, `build_feature_specs` raises
rather than emitting one, and the test suite asserts the resolved feature sets
are clean. Adding a column to the candidate lists cannot silently reintroduce
`mut_syn`-style memorisation.

The training script also prints a leakage probe (weighted |class rate − base
rate|) over every feature that reaches the model, so a newly leaky column is
visible rather than silent. The top scores on the current feature set are
`nuc_numb` 0.195 and `aa_numb` 0.171 — position along the gene, which is
biology, not leakage.

## Preprocessing

One `ColumnTransformer` per feature set, **fitted on the training split only**.
The same fitted object is serialised and used at inference, so training-time and
serving-time preprocessing cannot differ.

| Column type | Treatment |
|---|---|
| Categorical | explicit `Unknown` fill → one-hot, `min_frequency=3`, `handle_unknown="infrequent_if_exist"` |
| Numeric | median impute **with a missing indicator** → standardise |

Before that, two normalisations run over the merged records:

1. **Whitespace stripping.** The files contain both `"Point"` and `"Point "`,
   `"Exon"` and `"Exon "`, `"Severe"` and `"Severe "`.
2. **Case-only spelling variants folded onto the dominant spelling** — 10 of
   them, e.g. `missense` and `MIssense` → `Missense`, `severe` → `Severe`,
   `MIld` → `Mild`. Every collapse is listed by `scripts/validate_dataset.py`;
   the winner is always a spelling that occurs in the data. This is not a blanket
   `.lower()`: none of these columns carries a meaning that depends on case.

Encoded widths: genomic 248 columns, clinical 72, merged 320.

### Pipeline decisions

Recorded in `hemophilia_a.PIPELINE_DECISIONS` and in every artifact's
`metadata.json`:

1. **The modelling unit is the mutation**, not the clinical record.
2. **Conflicting mutations are excluded**, not majority-voted.
3. **Censored and range measurements are parsed**, not treated as text.
4. **Missing categoricals become an explicit `Unknown` level**, not the mode.
   `discrep` is 99.6% missing at mutation level: that measurement was never
   taken, and imputing the mode invents one.
5. **Whitespace and case-variant normalisation**, as above.
6. **Numeric imputation adds a missing indicator**, so an imputed median stays
   distinguishable from an observed value.
7. **Model selection uses `StratifiedGroupKFold` on `mut_id`.**
8. **The served model is fitted on the training split alone**, and its threshold
   is chosen on the validation split. The test split is touched exactly once.

The threshold maximises F1 rather than accuracy: at a 19.7% positive rate,
answering "no inhibitor" for everything scores 80% accuracy and is useless.

## The grouped split

`mut_id` is the grouping key for both splits — `GroupShuffleSplit`, 80/20 for
train+val / test, then 80/20 again for train / val.

```
load MMC2 + MMC3 -> validate -> filter F8 -> encode target
     -> merge on mut_id
     -> AGGREGATE to one row per mutation, drop conflicting labels
     -> GROUPED SPLIT
     -> fit preprocessor on TRAIN ONLY
     -> model selection, StratifiedGroupKFold(5) on mut_id
     -> isotonic calibration on train, same grouped folds
     -> threshold on the VALIDATION split
     -> evaluate ONCE on the held-out test split
```

| | Mutations | Positive |
|---|---|---|
| Training | **1,609** | 318 |
| Validation | **403** | 76 |
| Test | **503** | 102 |
| Total | 2,515 | 496 |

Group overlap, asserted at training time and again by the test suite:

| Pair | Shared `mut_id` |
|---|---|
| train ∩ validation | **0** |
| train ∩ test | **0** |
| validation ∩ test | **0** |

`assert_no_group_overlap` raises rather than continuing.

## Model selection

Cross-validated ROC-AUC on the training split, `StratifiedGroupKFold(5)` grouped
on `mut_id`. Eight families are compared; every one is given the class imbalance
explicitly (`class_weight`, `scale_pos_weight` or `auto_class_weights`) computed
from the split actually being trained on.

| Model | genomic | clinical | merged |
|---|---|---|---|
| logistic_regression | 0.709 ± 0.029 | 0.717 ± 0.018 | 0.740 ± 0.018 |
| **random_forest** | **0.762 ± 0.025** | **0.726 ± 0.028** | 0.783 ± 0.015 |
| xgboost | 0.744 ± 0.029 | 0.720 ± 0.021 | 0.774 ± 0.017 |
| lightgbm | 0.730 ± 0.035 | 0.724 ± 0.018 | 0.755 ± 0.026 |
| **catboost** | 0.762 ± 0.018 | 0.722 ± 0.021 | **0.787 ± 0.019** |
| svm_rbf | 0.732 ± 0.024 | 0.717 ± 0.026 | 0.768 ± 0.011 |
| mlp | 0.626 ± 0.065 | 0.724 ± 0.022 | 0.766 ± 0.019 |
| random_forest_smote | 0.760 ± 0.018 | 0.726 ± 0.026 | 0.776 ± 0.016 |

Selected: `random_forest` for genomic and clinical, `catboost` for merged. Each
is wrapped in `CalibratedClassifierCV(isotonic)` over the same grouped folds.

SMOTE did not help — class weighting already handles the 19.7% positive rate.
Where it is evaluated, it runs **inside** an imblearn pipeline, so resampling
happens within each cross-validation fold and never touches the fold being
scored. It is never applied before the split.

The top candidates sit within about one standard deviation of each other, so the
choice is not strongly evidenced and a different seed could reorder them.

## Results — held-out test set, touched once

503 mutations, 102 positive (20.3%).

| Metric | genomic | clinical | **merged** |
|---|---|---|---|
| ROC-AUC | 0.7793 | 0.7380 | **0.7998** |
| PR-AUC | 0.4850 | 0.3991 | **0.5042** |
| Brier score | 0.1331 | 0.1402 | **0.1296** |
| Precision | 0.3232 | 0.4286 | **0.4167** |
| Recall (sensitivity) | 0.8333 | 0.2941 | **0.6863** |
| Specificity | 0.5561 | 0.9002 | **0.7556** |
| F1 | 0.4658 | 0.3488 | **0.5185** |
| Balanced accuracy | 0.6947 | 0.5972 | **0.7209** |
| Accuracy | 0.6123 | 0.7773 | **0.7416** |
| MCC | 0.3135 | 0.2258 | **0.3767** |
| Threshold | 0.1499 | 0.2516 | **0.2384** |

Confusion matrices at those thresholds:

| | TN | FP | FN | TP |
|---|---|---|---|---|
| genomic | 223 | 178 | 17 | 85 |
| clinical | 361 | 40 | 72 | 30 |
| **merged** | **303** | **98** | **32** | **70** |

### Fusion is worth it

The fused model beats **both** single-source models on every ranking metric:

| | ROC-AUC | PR-AUC |
|---|---|---|
| MMC2 genomic only | 0.7793 | 0.4850 |
| MMC3 clinical only | 0.7380 | 0.3991 |
| **MMC2 + MMC3 fused** | **0.7998** | **0.5042** |

That is the central claim of this architecture, and it is the reason the fused
model is the default. Genomic information alone outperforms clinical information
alone here; together they beat either.

### Read this honestly

PR-AUC 0.504 against a 0.203 base rate is a **2.5× lift** — real signal. But at
the F1-optimal threshold the fused model flags 168 mutations to catch 70 of 102
true positives: more false alarms than true ones, and it still misses 32. It is
a screening aid that errs toward sensitivity, not a decision rule.

Accuracy is reported because the brief asks for it, but it is not the headline.
Answering "no inhibitor" for every mutation scores 0.797 here — more than the
clinical-only model (0.777) manages by actually trying. That model wins on
accuracy, precision and specificity precisely because it predicts negative more
often; its recall is 0.294, so it misses seven of every ten positives.

Note also that accuracy *fell* relative to the previous record-level pipeline
while ROC-AUC and PR-AUC *rose*. That is expected and is not a regression: the
old numbers were computed over repeated records of the same mutations, which
inflated apparent accuracy. These numbers are over 503 distinct mutations, each
counted once.

### Threshold

Chosen to maximise F1 on the **validation** split, recorded in `metadata.json`,
and read from there at inference. Nothing in the system assumes 0.5. Moving it
trades recall for precision and is a clinical judgement, not a technical default.

## Artifacts

```
ml/artifacts/
  mmc2-genomic-v1/    model.joblib · preprocessor.joblib · background.npy
  mmc3-clinical-v1/   metadata.json · metrics.json
  mmc2-mmc3-v1/       (default — the fused model)
  training_summary.json
```

The names say which source table each model sees, so a served version cannot be
mistaken for one fitted on a different block.

`metadata.json` carries the SHA-256 of both source CSVs, the label, merge,
mutation-level and population reports, the leakage probe, the split summary with
its overlap counts, the threshold, the library versions, the pipeline decisions,
and the `FeatureSpec` — the exact categorical/numeric column lists the model was
fitted on, plus the raw input fields a caller supplies.

Inference reconstructs its column list from that `FeatureSpec` rather than from a
hardcoded constant, so a served model cannot drift away from the feature set it
was trained on. `ml/artifacts.py` additionally refuses to load a bundle whose
metadata feature count disagrees with the estimator's `n_features_in_`.

## Train/serve consistency

A caller supplies **raw fields** (`clotting`, `cli_phe`, `mut_type`), not the
aggregates the model consumes. `hemophilia_a.build_input_row` bridges the two by
wrapping the record in a one-row frame and pushing it through
`aggregate_to_mutations` — *the same function training used*. A single record
therefore produces exactly the aggregates a mutation with one clinical record
produced during training: `mean = median = min = max = the value`,
`censored_rate` 0 or 1.

Reusing the training function rather than reimplementing the arithmetic is the
point: there is no second copy of the aggregation to drift out of sync. The test
suite asserts this directly, by comparing a real single-record mutation's row in
the aggregated table against the row `build_input_row` builds from its raw
fields.

## Explainability

`ml/explainability/service.py` — one service, both methods, driven by the
bundle's own `FeatureSpec`.

- **SHAP**: TreeSHAP over the ensemble underneath the isotonic calibrator. The
  unwrapping handles a `RandomForestClassifier`, a `CatBoostClassifier`, an
  `XGBClassifier` and an imblearn `Pipeline` wrapping one, because different
  feature sets select different families.
- **LIME**: local linear approximation over the training background sample, with
  continuous discretisation disabled (the matrix is one-hot and standardised).

Both aggregate the encoded columns back onto the **raw field** they came from, so
an explanation names `clotting`, never `num__clotting_median` and never
`cat__mut_type_Point`. A genomic-only model can never name a clinical field. Each
contribution carries a human-readable label and a `supplied` flag: a field the
caller left blank may appear — its absence really did move the estimate — but it
is shown as *not reported* with a null value, never as a value the caller
entered.

If an explainer fails, the response says `available: false` with a reason. It
never falls back to something that merely looks like an attribution.

### Global importance — fused model

Mean |SHAP| over the training background sample, top 8:

| Feature | Label | Mean \|SHAP\| |
|---|---|---|
| `clotting` | FVIII clotting activity (%) | 0.0468 |
| `cli_phe` | Clinical severity | 0.0401 |
| `e_i_numb` | Exon / intron number | 0.0265 |
| `ratio` | Activity ratio | 0.0260 |
| `mut_effect` | Mutation effect | 0.0252 |
| `mut_type` | Mutation type | 0.0240 |
| `aa_first` | Reference amino acid | 0.0202 |
| `nuc_numb` | Nucleotide position | 0.0181 |

Residual clotting activity and clinical severity leading, followed by mutation
type and effect, is consistent with the literature: null mutations that leave no
circulating factor VIII protein carry the highest inhibitor risk. Both blocks
appear in the top eight, which is what the fused model is for.

This is a marked change from the previous record-level pipeline, whose second
most important feature was `pa_race` — a reporting artifact. Excluding it, and
parsing `clotting` instead of one-hot encoding its 86 distinct strings, is what
moved biology to the top of this list.

## Limitations

1. **Mutation-level, not patient-level.** A row is one F8 mutation. An estimate
   describes how often that mutation is reported with an inhibitor, not an
   individual's risk. Patients are not identifiable (`pa_id` is excluded).
2. **Reporting bias.** 5,098 of 10,064 MMC3 records (51%) have no explicit
   Yes/No inhibitor value, leaving **3,566 F8 mutations with no usable label at
   all**. That exclusion is very unlikely to be random — records from
   inhibitor-focused work are likelier to have the field filled — so the 19.7%
   positive rate describes this dataset's reporting, not population incidence,
   and the model inherits that bias.
3. **Conflicting mutations are excluded.** The 124 dropped mutations are those
   whose records disagree, which are plausibly the hardest cases. The modelled
   population is therefore easier than the full one, and the reported metrics are
   optimistic to that extent.
4. **Thin clinical evidence per mutation.** The mean is 1.88 clinical records per
   mutation, so for most mutations the mean, median, min and max coincide and the
   aggregate features carry no more information than a single reading would.
5. **Censored values are bounds, not measurements.** 76.3% of clotting readings
   are censored. Treating `<1` as 1.0 with an indicator is defensible but it is
   still an approximation of an interval-censored quantity; a proper survival-
   style treatment would be better.
6. **No external validation.** One dataset, one grouped split. **No clinical
   validation is claimed.**
7. **Modest discrimination.** ROC-AUC 0.800 with 42% precision at the
   operating threshold. Useful for screening, not for deciding.
8. **Model choice is weakly evidenced.** CatBoost and random forest are within
   0.005 ROC-AUC of each other on the merged block, well inside one standard
   deviation.
