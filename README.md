# Hemophilia Inhibitor-Risk Predictor

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/tests-230%20passing-2C6642)
![Status](https://img.shields.io/badge/status-research%20prototype-8A5D0B)

An explainable Hemophilia A inhibitor-risk prediction **research prototype**,
built on the **MMC2** (genomic) and **MMC3** (clinical) supplementary tables,
fused on `mut_id` into **one row per mutation**.

> **Medical disclaimer.** This is research decision-support software, not a
> diagnostic device. It is **not clinically validated**, has no external
> validation cohort, and must not be used for standalone diagnosis or treatment
> decisions. It never recommends a course of treatment. Estimates are attributed
> to a **reported mutation**, not to an individual patient — see
> [Limitations](#limitations).

## Contents

- [What it does](#what-it-does)
- [Project phase](#project-phase)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Prediction target](#prediction-target)
- [Feature groups](#feature-groups)
- [ML pipeline and results](#ml-pipeline-and-results)
- [Explainability](#explainability)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Reproducing the whole pipeline](#reproducing-the-whole-pipeline)
- [Running locally](#running-locally)
- [Running the tests](#running-the-tests)
- [Docker](#docker)
- [API](#api)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Repository history](#repository-history)

## What it does

A clinician signs in, records a patient, and describes that patient's F8
mutation, their clinical picture, or both. The application returns a calibrated
probability that a **mutation** with that description is reported with an
inhibitor, together with SHAP and LIME explanations of which genomic and
clinical features drove the estimate. Predictions and explanations are stored
and viewable as history.

## Project phase

Stated plainly so the scope is not read off the ambition.

**Current version — MMC2 + MMC3, fused at the mutation level.** Implemented and
serving:

- MMC2 genomic features and MMC3 clinical features, joined on `mut_id`
- genomic-only, clinical-only and merged prediction modes, one trained artifact
  each, selectable per request
- calibrated probability, decision threshold chosen on a validation split, risk
  band, SHAP and LIME explanations, stored prediction history
- doctor sign-in, patient records scoped per account, audit log

**Not implemented. The application does not have these and does not claim to:**

- **patient-level data of any kind.** A row is one F8 mutation as the source
  literature reports it. There is no patient-level cohort behind any estimate,
  so nothing here is a statement about an individual.
- **treatment history, exposure days (ED), product type or switching.** None of
  these columns exists in MMC2 or MMC3, so no model was fitted on them and no
  form collects them.
- **HLA typing or immune biomarkers.** Same reason.
- **external validation.** One dataset, one grouped split.

The patient form therefore asks for an identifier and a display name and nothing
else clinical: every predictive field is an MMC2/MMC3 column, and the form is
built from the served model's own schema (`GET /api/predictions/schema`) so it
cannot offer a field the model was never fitted on.

**Future work** would need a patient-level cohort with treatment and exposure
records, and an external validation set. Adding those means new datasets and a
retrained model, not new form fields.

## Architecture

```
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │ TypeScript / Vite   │
                    │ form built from the │
                    │ model's own schema  │
                    └──────────┬──────────┘
                               │ REST /api
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │ auth · patients     │
                    │ predictions         │
                    │ explanations        │
                    │ analytics · health  │
                    └──────────┬──────────┘
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          ┌─────────────┐          ┌──────────────────┐
          │  SQLite DB  │          │   ml/ package    │
          │  users      │          │  preprocessing   │
          │  patients   │          │  inference       │
          │  case_      │          │  explainability  │
          │   records   │          └────────┬─────────┘
          │  predictions│                   ▼
          │  explanations│         ┌──────────────────┐
          │  audit_logs │          │ mmc2-genomic-v1  │
          └─────────────┘          │ mmc3-clinical-v1 │
                                   │ mmc2-mmc3-v1 ★   │
                                   │ model + preproc  │
                                   │ + metadata       │
                                   └──────────────────┘
```

Three deployable pieces: frontend, backend, database. ML inference runs inside
the backend process — a separate model service would add operational cost with
no benefit at this size. Every model is loaded **once at startup**, not per
request.

## Dataset

Two supplementary tables of the source publication, in `ml/data/`. The files are
never modified; all interpretation happens in
`ml/preprocessing/hemophilia_a.py`.

| | MMC2 | MMC3 |
|---|---|---|
| `BVTH_VTH-2024-000215-mmc2.csv` | 6,211 rows × 34 cols | |
| `BVTH_VTH-2024-000215-mmc3.csv` | | 10,064 rows × 23 cols |
| Grain | one **mutation** | one **clinical record** |
| Unique `mut_id` | 6,211 | 6,212 |
| Repeated `mut_id` | 0 | 3,852 |

They are fused on **`mut_id`** in two steps. MMC2 is reduced to one row per
mutation and joined to MMC3; then every clinical record behind a mutation is
**aggregated per `mut_id`**, so the modelling unit is the **mutation** — not the
clinical record, and not the patient. A mutation reported by many records
contributes one row rather than many near-identical rows carrying an identical
genomic block, which is what keeps the evaluation from rewarding a model for
memorising a frequently reported mutation.

A mutation whose clinical records **disagree** about the inhibitor outcome has
no single label. Those mutations are counted, reported and **excluded** from
supervised training rather than resolved by a majority vote, which would
manufacture a certainty the source data does not contain.

`python scripts/validate_dataset.py` prints the full report — row counts,
duplicates, missing values, the join and aggregation outcome, the conflicting
mutations, the target distribution, and every resolved feature — before anything
is fitted. The post-fusion counts (mutations after the join, mutations excluded
as conflicting, mutations carrying a usable label, positive rate) come from that
report and from each artifact's `metadata.json`. Measured: **2,639** F8
mutations carry a usable label and **3,566** carry none; **124** conflict and
are excluded; the **final modelling population is 2,515 mutations — 496
positive, 2,019 negative, a 19.72% positive rate.**

Because aggregation leaves exactly one row per mutation, no mutation can appear
in two partitions even in principle. **`mut_id` is still the grouping key for
every split**, and the zero-overlap property is asserted anyway.

## Prediction target

MMC3's `Inhibitors` column:

```
Yes  ->  1
No   ->  0
```

Case- and whitespace-insensitive. Everything else — `Not reported`, blank,
`Not`, and the records where a severity was typed into the field — is
**excluded and counted, never imputed**.

That label is then resolved **per mutation**, because the modelling unit is the
mutation:

```
every labelled record reports an inhibitor  ->  1
no labelled record reports one              ->  0
the records disagree                        ->  excluded, counted, never voted on
```

`Inhibitors` never appears as an input feature, and neither do the columns that
would leak it: `uinhibitor` (MMC2's curated inhibitor status, which reproduces
the label almost exactly), `type` and `utype` (inhibitor kinetic type, recorded
only when an assay was run), `assay` (which assay produced the reading),
`pa_race` (the reporting country, which tracks which cohorts were published),
and the free-text identifiers `mut_syn`, `aa_syn`, `aa_change` and
`codon_change`, which name a mutation rather than describe it. The full list,
with the reason for each, lives in `EXCLUDED_COLUMNS` in
`ml/preprocessing/hemophilia_a.py` — in code, not only in documentation.

## Feature groups

Three blocks, resolved from the columns that actually exist in the files. Each
is a separately trained model and a selectable prediction mode.

Three counts, because they differ and the difference matters: **fields** is what
a caller supplies, **columns** is what those become after the per-mutation
clinical aggregation expands each assay into mean / median / min / max and a
censored rate, and **encoded features** is the width of the matrix the model
sees after one-hot encoding. Every number here is read from the artifact's own
`metadata.json`.

| Mode | Source | Model version | Input columns |
|---|---|---|---|
| **Genomic** | MMC2 | `mmc2-genomic-v1` | 14 fields → 14 columns → 248 encoded features |
| **Clinical** | MMC3, aggregated per mutation | `mmc3-clinical-v1` | 6 fields → 33 columns → 72 encoded features |
| **Merged** | both | `mmc2-mmc3-v1` | 20 fields → 47 columns → 320 encoded features — **the default** |

Genomic — one value per mutation, straight from MMC2: `mut_type`, `mut_effect`,
`location`, `e_i_numb`, `locnumb`, `aa_numb`, `codon_first`, `codon_last`,
`n_bp`, `nuc_numb`, `ntchange`, `aa_first`, `aa_last`, `CpG`. The positional
ones are parsed as measurements rather than one-hot encoded, because exon 14 is
genuinely between exon 13 and exon 15.

Clinical — summarised over every record behind the mutation: `clotting`,
`antigen`, `ratio`, `act/ant` and `discrep`, each written as free text in the
source (censored readings like `<1`, ranges like `1 to 5`) and therefore parsed
into a value plus a censoring flag, then reduced to mean / median / min / max
and a censored rate per mutation; and `cli_phe`, the severity phenotype, which
contributes an ordinal score and the proportion of the mutation's records in
each severity bucket.

**Candidate columns that are dropped, with the reason recorded in code rather
than hidden:**

- `mutations` — a single constant value for every merged record, so it carries
  nothing.
- `bleed_tool`, `bleed_score` — present in the file but effectively empty.
- `aa_numb_old` — the pre-2001 amino-acid numbering of `aa_numb`, differing
  only by the signal-peptide offset.
- `Count_mut_id`, `n_clinical_records` — how many records mention a mutation.
  A reporting artifact, and after conflicting mutations are excluded it encodes
  the exclusion rule rather than biology.

Every one of these, and every leakage exclusion above, sits in
`EXCLUDED_COLUMNS` with its justification, so the reason travels with the
pipeline instead of living only on this page.

## ML pipeline and results

```
MMC2 + MMC3 → validate → filter F8 → encode target → merge on mut_id
     → AGGREGATE per mut_id (one row per mutation, conflicting ones excluded)
     → GROUPED SPLIT on mut_id (no mutation crosses a boundary)
     → fit preprocessor on TRAIN ONLY
     → model selection, StratifiedGroupKFold(5) on mut_id
     → isotonic calibration on train
     → threshold chosen on the validation split
     → evaluate ONCE on the held-out test set
     → save artifact → serve → SHAP / LIME
```

One row is one mutation, so the split table counts mutations and nothing else:

| | Mutations | Positive |
|---|---|---|
| Training | 1,609 | 318 |
| Validation | 403 | 76 |
| Test | 503 | 102 |

Overlapping mutations between any two splits: **0**, asserted at training time
and again by the test suite.

Held-out test set, produced by `scripts/train_inhibitor_model.py`:

| Metric | Genomic | Clinical | **Merged** |
|---|---|---|---|
| Accuracy | 0.6123 | 0.7773 | **0.7416** |
| Precision | 0.3232 | 0.4286 | **0.4167** |
| Recall (sensitivity) | 0.8333 | 0.2941 | **0.6863** |
| Specificity | 0.5561 | 0.9002 | **0.7556** |
| F1 | 0.4658 | 0.3488 | **0.5185** |
| ROC-AUC | 0.7793 | 0.7380 | **0.7998** |
| PR-AUC | 0.4850 | 0.3991 | **0.5042** |
| Brier | 0.1331 | 0.1402 | **0.1296** |

Merged confusion matrix at the operating threshold (0.2384): TN 303, FP 98, FN 32, TP 70.

> Every number above is copied from the artifacts' own `metrics.json`,
> produced by a real run of `scripts/train_inhibitor_model.py` (seed 42).
> Nothing is typed by hand or carried over from an earlier pipeline.
>
> **The fused model beats both single-source models on ROC-AUC and PR-AUC**
> — ROC-AUC 0.7998 against 0.7793 (genomic) and 0.7380 (clinical),
> PR-AUC 0.5042 against 0.4850 and 0.3991. That is the point of the fusion.

**Read this honestly.** Read PR-AUC against the positive rate rather than
reading accuracy. PR-AUC 0.5042 against a 20.3% base rate is a 2.5× lift — real
signal. But at the operating threshold the fused model flags 168 mutations to
catch 70 of 102 true positives: more false alarms than true ones, and it still
misses 32. Answering "no inhibitor" for every mutation would score 0.797
accuracy here — more than the clinical-only model (0.777) manages by trying.
It is a screening aid that errs toward sensitivity, not a decision rule.

Full detail — the model comparison, the preprocessing, the censored-value
parsing, the leakage exclusions and every caveat: **[docs/ML.md](docs/ML.md)**.

## Explainability

One service, `ml/explainability/service.py`, provides both:

- **SHAP** — exact TreeSHAP over the ensemble beneath the isotonic calibrator
  (~0.4 s). Global feature importance and per-prediction local contributions.
- **LIME** — local linear approximation for individual predictions.

Both aggregate the encoded columns back onto the source MMC2/MMC3 columns and
attach a human-readable label, so an explanation never names
`cat__mut_type_Point`, and the column list comes from the artifact's own feature
spec — a genomic-only model cannot name a clinical field. A field the caller
left blank may still appear, since its absence really did move the estimate, but
it is marked `supplied: false` with a null value rather than being shown as
something the caller entered. If an explainer fails, the response says so rather
than returning a fabricated attribution.

## Installation

Requires Python 3.12 and Node 20+.

```bash
git clone https://github.com/devtejasx/Hemophilia-risk-predictor.git
cd Hemophilia-risk-predictor
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

For training and tests, use `pip install -r requirements-dev.txt` instead.

`scikit-learn` is pinned to 1.8.0 exactly: the serialised artifacts were fitted
with it and a minor-version drift changes unpickling behaviour.

## Environment variables

Copy `.env.example` to `.env`. Nothing is committed with a real value, and
**every path defaults inside the checkout**, so the project runs after a clone
with nothing configured.

| Variable | Default | Notes |
|---|---|---|
| `JWT_SECRET_KEY` | — | **Required in production.** In development a random per-process key is generated, so tokens do not survive a restart. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ENVIRONMENT` | `development` | `production` makes a missing/short secret fatal |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Exact origins, comma separated. Never `*`. |
| `DATABASE_PATH` | `hemophilia.db` | |
| `DATA_DIR` | `ml/data` | Where the two CSVs live |
| `MMC2_PATH` | `$DATA_DIR/BVTH_VTH-2024-000215-mmc2.csv` | Overrides `DATA_DIR` for one file |
| `MMC3_PATH` | `$DATA_DIR/BVTH_VTH-2024-000215-mmc3.csv` | |
| `ML_ARTIFACTS_DIR` | `ml/artifacts` | |
| `ML_DEFAULT_FEATURE_SET` | `merged` | `genomic`, `clinical` or `merged` |
| `ML_MODEL_VERSION` | *(empty)* | Pins one artifact and disables mode routing |
| `BOOTSTRAP_ADMIN_EMAIL` / `_PASSWORD` | empty | If unset, **no account is seeded at all** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | |

`DATA_DIR`, `MMC2_PATH` and `MMC3_PATH` are read only by training and dataset
validation. Serving a built model does not open the CSVs at all.

Frontend (`frontend/.env`): `VITE_API_URL` — only `VITE_*` reaches the browser
bundle, so never put a secret there.

## Reproducing the whole pipeline

Nothing depends on running a notebook by hand. Two scripts cover the data path
end to end:

```bash
# 1. What is in the files, before anything is fitted
python scripts/validate_dataset.py            # add --json for machine output

# 2. Merge, aggregate, split, train, evaluate, save (all three feature sets)
python scripts/train_inhibitor_model.py

# ...or one at a time
python scripts/train_inhibitor_model.py --feature-set merged
python scripts/train_inhibitor_model.py --seed 7 --version-suffix v2
```

`train_inhibitor_model.py` prints the source reports, the merge report, the
aggregation report with the mutations it excluded as conflicting, the grouped
split with its overlap counts, the leakage probe, the cross-validation
comparison and the held-out test metrics, then writes
`ml/artifacts/mmc2-genomic-v1/`, `ml/artifacts/mmc3-clinical-v1/` and
`ml/artifacts/mmc2-mmc3-v1/` (model, preprocessor, background sample,
`metadata.json`, `metrics.json`) plus `ml/artifacts/training_summary.json`.

Everything in `metrics.json` comes from that run. Nothing is hand-edited.

## Running locally

Two terminals:

```bash
uvicorn backend.main:app --reload --port 8000
```

```bash
cd frontend && npm run dev
```

Open http://localhost:3000, create an account, add a patient, choose a
prediction mode, and describe the mutation and/or its clinical readings.
`GET /health` reports database and model status for every mode.

Retraining is optional — all three artifacts are committed.

## Running the tests

```bash
pytest
```

230 tests: dataset loading, the join on `mut_id`, censored-measurement parsing,
mutation-level aggregation and the exclusion of conflicting mutations, target
encoding and its exclusions, feature resolution, the identifier guard,
preprocessing and feature ordering, grouped-split leakage, train/serve
consistency, artifact consistency, inference, thresholding, input validation,
SHAP/LIME, the API contract for all three modes, authentication, authorisation,
the database migration, failure handling, and end-to-end tests that walk
input → API → preprocessing → model → prediction → explanation → database →
response.

Frontend type-check and build:

```bash
cd frontend && npm run build
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Frontend on `:3000` (nginx, proxying `/api` to the backend), backend on `:8000`,
SQLite on a named volume. `JWT_SECRET_KEY` must be supplied — there is no
default.

> The Docker path is configured but has **not** been executed in this
> environment; the local and test paths above have been.

## API

Interactive docs at `/docs`. Full reference: **[docs/API.md](docs/API.md)**.

```
POST   /api/auth/register · /api/auth/login      GET /api/auth/me
POST   /api/patients                             GET /api/patients · /api/patients/{id}
GET    /api/predictions/schema?feature_set=…
POST   /api/patients/{id}/predictions
GET    /api/predictions/{id}
GET    /api/predictions/{id}/explanation
GET    /api/patients/{id}/history
GET    /api/explanations/global
GET    /api/analytics                            GET /health
```

A prediction request:

```bash
curl -X POST http://localhost:8000/api/patients/1/predictions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"feature_set":"merged","features":{
        "mut_type":"Point","mut_effect":"Missense","location":"Exon",
        "e_i_numb":"14","locnumb":"14","n_bp":"1","nuc_numb":"1834",
        "ntchange":"C>T","aa_numb":612,
        "cli_phe":"Severe","clotting":"<1"}}'
```

```json
{ "prediction": 0, "risk": "Low", "probability": 0.198477,
  "risk_category": "Lower estimated risk", "threshold": 0.2384,
  "model_version": "mmc2-mmc3-v1", "feature_set": "merged" }
```

The accepted keys are not hardcoded anywhere in the API layer — call
`GET /api/predictions/schema` for the fields, labels, groups, required set and
accepted values of whichever mode you want.

## Project structure

```
frontend/            React + TypeScript + Vite SPA (the production UI)
  components/PredictionForm.tsx   the form, built from the model's own schema
backend/             one FastAPI application
  core/              config, security
  routers/           auth, patients, predictions, analytics
  services/ml.py     adapter over the ml package; one service per mode
  db.py              the single SQLite layer; its migration renames a superseded
                     schema's tables aside (legacy_pre_mmc_*) with data intact
ml/
  data/              BVTH_VTH-2024-000215-mmc2.csv + …-mmc3.csv
  preprocessing/hemophilia_a.py   load, validate, merge, aggregate per mutation,
                                  feature sets, transformer
  explainability/    SHAP + LIME
  artifacts/
    mmc2-genomic-v1/  mmc3-clinical-v1/  mmc2-mmc3-v1/  (nothing else is servable)
  inference.py       the single predict() entrypoint
scripts/
  validate_dataset.py         dataset report, exits non-zero on a bad file
  train_inhibitor_model.py    merge -> aggregate -> split -> train -> evaluate -> save
tests/               230 tests, including end-to-end
docker/              Dockerfiles, compose, nginx
docs/                CANONICAL.md · ML.md · API.md
archive/             superseded implementations, kept for provenance; nothing
                     here is imported, served or maintained
```

## Limitations

1. **Mutation-level, not patient-level.** A row is one F8 mutation as reported
   in the literature behind MMC3, not a person. The output describes how often
   that mutation is reported with an inhibitor, not an individual's future.
2. **Reporting bias.** Roughly half of MMC3's records are excluded for having no
   explicit Yes/No inhibitor value, almost certainly non-randomly — records from
   inhibitor-focused work are likelier to have the field filled. The positive
   rate describes this dataset's reporting, not population incidence.
3. **Excluding conflicting mutations is not free.** A mutation whose records
   disagree is dropped rather than voted on, which is the honest choice but
   removes exactly the ambiguous cases a clinician would most want help with.
   The surviving multi-record mutations are unanimous by construction.
4. **Reporting artifacts are excluded, and so is their signal.** `pa_race` (the
   reporting country, which largely repeats the reporting laboratory) and the
   per-mutation record counts predict the label in this file, but they encode
   which cohorts were published rather than biology, so the model never sees
   them. Nothing recovers whatever real effect they may have been standing in
   for.
5. **No external validation.** One dataset, one grouped split. **No clinical
   validation is claimed.**
6. **Modest discrimination.** ROC-AUC 0.7998 with 42% precision at
   the operating threshold. Useful for screening, not for deciding.
7. **Model choice is weakly evidenced.** CatBoost and random forest are within
   0.005 ROC-AUC of each other on the merged block — well inside one standard
   deviation — and six of the eight candidate families sit inside that band.

## Repository history

This repository previously contained four competing application stacks, three
non-importing FastAPI backends, four Streamlit UIs, five chatbot
implementations, and around 100 aspirational markdown documents. The canonical
choices that resolved it are in **[docs/CANONICAL.md](docs/CANONICAL.md)**.

Two findings worth knowing:

- **The models committed before the audit were never trained on a real dataset.**
  They were fitted on 30 fabricated rows whose label was a deterministic function
  of two input columns. They are preserved, unmodified and clearly labelled, in
  `archive/legacy-artifacts/legacy-synthetic-v0/`. They were moved out of
  `ml/artifacts/` during this migration, so they are no longer a loadable model
  version at all.
- **`hemophilia_clinic.db` was committed to git** containing four accounts that
  shared the unsalted MD5 of `password123`. The file is now untracked, but
  untracking does not remove it from git history: those accounts should be
  treated as compromised.

Superseded implementations were moved into `archive/` with `git mv`, so history
follows each file. The one exception is the retired variant-level dataset and
its pipeline, which were **deleted outright** rather than archived: keeping a
second dataset in the tree invited exactly the confusion this migration set out
to end. It remains reachable in git history, but nothing in the working tree
refers to it.

Only what this page describes is live: the MMC2 + MMC3 fusion, aggregated per
mutation, behind one FastAPI application and one React UI. Metrics produced by
any earlier pipeline describe a different dataset and a different modelling
unit, and must never be quoted for the current models.

## License

MIT — see [LICENSE](LICENSE).
