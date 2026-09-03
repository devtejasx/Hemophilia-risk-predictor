# Hemophilia Inhibitor-Risk Predictor

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/tests-161%20passing-2C6642)
![Status](https://img.shields.io/badge/status-research%20prototype-8A5D0B)

An explainable Hemophilia A inhibitor-risk prediction **research prototype**,
built on the **MMC2 + MMC3** mutation and clinical-record tables.

> **Medical disclaimer.** This is research decision-support software, not a
> diagnostic device. It is **not clinically validated**, has no external
> validation cohort, and must not be used for standalone diagnosis or treatment
> decisions. It never recommends a course of treatment. Estimates are attributed
> to a **reported record**, not to an individual patient — see
> [Limitations](#limitations).

## Contents

- [What it does](#what-it-does)
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
mutation, their clinical record, or both. The application returns a calibrated
probability that a record with that description reports inhibitor development,
together with SHAP and LIME explanations of which mutation and clinical features
drove the estimate. Predictions and explanations are stored and viewable as
history.

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
          │  audit_logs │          │ mmc-genomic-v1   │
          └─────────────┘          │ mmc-clinical-v1  │
                                   │ mmc-merged-v1 ★  │
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

They are joined on **`mut_id`**: MMC2 is reduced to one row per mutation, MMC3
keeps every clinical record. **4,962 merged records over 2,639 unique
mutations**, 836 positive (16.85%).

`python scripts/validate_dataset.py` prints the full report — row counts,
duplicates, missing values, the join outcome, the target distribution, and every
resolved feature — before anything is fitted.

Because one mutation can appear in up to 104 records, **`mut_id` is the grouping
key for every split**. No mutation is allowed to appear in two partitions.

## Prediction target

MMC3's `Inhibitors` column:

```
Yes  ->  1
No   ->  0
```

Case- and whitespace-insensitive. Everything else — `Not reported` (2,090),
blank (1,730), `Not` (1,276), and two records where a severity was typed into
the field — is **excluded and counted, never imputed**. 4,966 records carry an
explicit label; 836 of them are positive.

`Inhibitors` never appears as an input feature, and neither does `uinhibitor`,
MMC2's curated inhibitor status, which reproduces the label almost exactly
(97.1% positive when `Yes`) and is on the exclusion list as leakage.

## Feature groups

Three blocks, resolved from the columns that actually exist in the files. Each
is a separately trained model and a selectable prediction mode.

| Mode | Source | Columns |
|---|---|---|
| **Genomic** | MMC2 | 20 |
| **Clinical** | MMC3 | 9 |
| **Merged** | both | 29 — **the default** |

Genomic: `mut_type`, `mut_effect`, `location`, `e_i_numb`, `locnumb`,
`aa_numb_old`, `aa_numb`, `codon_change`, `codon_first`, `codon_last`, `n_bp`,
`nuc_numb`, `ntchange`, `mut_syn`, `aa_change`, `aa_first`, `aa_last`, `aa_syn`,
`CpG`, `utype`.

Clinical: `clotting`, `discrep`, `ratio`, `assay`, `antigen`, `act/ant`, `type`,
`pa_race`, `cli_phe`.

**Two candidate columns are dropped, and the reason is recorded rather than
hidden:**

- `mutations` — present in *both* tables, so the join renames it to
  `mutations_clinical` / `mutations_genomic` and the candidate column does not
  exist in the merged frame.
- `bleed_tool`, `bleed_score` — present but 100% null across all 4,962 merged
  records.

Both match the reference notebook's own `valid_features` filter.

## ML pipeline and results

```
MMC2 + MMC3 → validate → filter F8 → encode target → merge on mut_id
     → GROUPED SPLIT on mut_id (no mutation crosses a boundary)
     → fit preprocessor on TRAIN ONLY
     → model selection, StratifiedGroupKFold(5) on mut_id
     → isotonic calibration on train
     → threshold chosen on the validation split
     → evaluate ONCE on the held-out test set
     → save artifact → serve → SHAP / LIME
```

| | Records | Mutations | Positive |
|---|---|---|---|
| Training | 3,124 | 1,688 | 525 |
| Validation | 812 | 423 | 142 |
| Test | 1,026 | 528 | 169 |

Overlapping mutations between any two splits: **0**, asserted at training time
and again by the test suite.

Held-out test set, produced by `scripts/train_inhibitor_model.py`:

| Metric | Genomic | Clinical | **Merged** |
|---|---|---|---|
| Accuracy | 0.7982 | 0.8265 | **0.8012** |
| Precision | 0.4087 | 0.4587 | **0.4229** |
| Recall (sensitivity) | 0.5030 | 0.2959 | **0.5680** |
| Specificity | 0.8565 | 0.9312 | **0.8471** |
| F1 | 0.4509 | 0.3597 | **0.4848** |
| ROC-AUC | 0.7327 | 0.7156 | **0.7646** |
| PR-AUC | 0.3911 | 0.4144 | **0.4754** |
| Brier | 0.1220 | 0.1182 | **0.1126** |

Merged confusion matrix at threshold 0.245: TN 726, FP 131, FN 73, TP 96.

**Read this honestly.** The merged model's PR-AUC of 0.475 against a 0.165 base
rate is a 2.9× lift — real signal, and combining both blocks genuinely beats
either alone on every ranking metric. But at the operating threshold it flags
227 records to catch 96 of 169 true positives: more than one false alarm per
true one, and it still misses 73. It is a screening aid that errs toward
sensitivity, not a decision rule. Accuracy is deliberately not the headline:
answering "no inhibitor" for everything scores 0.835 here and is useless — which
is exactly how the clinical-only model wins on accuracy while having the worst
recall of the three.

Full detail, including the model comparison, the preprocessing, the deviations
from the reference notebook and every caveat: **[docs/ML.md](docs/ML.md)**.

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

# 2. Merge, split, train, evaluate, save (all three feature sets)
python scripts/train_inhibitor_model.py

# ...or one at a time
python scripts/train_inhibitor_model.py --feature-set merged
python scripts/train_inhibitor_model.py --seed 7 --version-suffix v2
```

`train_inhibitor_model.py` prints the source reports, the merge report, the
grouped split with its overlap counts, the leakage probe, the cross-validation
comparison and the held-out test metrics, then writes
`ml/artifacts/mmc-<set>-v1/` (model, preprocessor, background sample,
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
prediction mode, and describe the mutation and/or the clinical record.
`GET /health` reports database and model status for every mode.

Retraining is optional — all three artifacts are committed.

## Running the tests

```bash
pytest
```

161 tests: dataset loading, the join on `mut_id`, target encoding and its
exclusions, feature resolution, preprocessing and feature ordering, grouped-split
leakage, artifact consistency, inference, thresholding, input validation,
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
        "mut_syn":"c.1834C>T","cli_phe":"Severe",
        "aa_numb_old":593,"aa_numb":612}}'
```

```json
{ "prediction": 0, "risk": "Low", "probability": 0.093977,
  "risk_category": "Lower estimated risk", "threshold": 0.2453,
  "model_version": "mmc-merged-v1", "feature_set": "merged" }
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
  db.py              the single SQLite layer, incl. the CHAMP-era migration
ml/
  data/              BVTH_VTH-2024-000215-mmc2.csv + …-mmc3.csv
  preprocessing/hemophilia_a.py   load, validate, merge, feature sets, transformer
  explainability/    SHAP + LIME
  artifacts/
    mmc-genomic-v1/  mmc-clinical-v1/  mmc-merged-v1/
    legacy-synthetic-v0/   preserved for provenance; see its PROVENANCE.md
  inference.py       the single predict() entrypoint
scripts/
  validate_dataset.py         dataset report, exits non-zero on a bad file
  train_inhibitor_model.py    merge -> split -> train -> evaluate -> save
tests/               161 tests, including end-to-end
docker/              Dockerfiles, compose, nginx
docs/                AUDIT.md · CANONICAL.md · ML.md · API.md
archive/             superseded implementations, kept for reference
  legacy-champ/      the retired CHAMP dataset, preprocessor, script and model
```

## Limitations

1. **Record-level, not patient-level.** A row is one reported clinical record;
   several records may describe the same mutation. The output is attributable to
   that description, not to an individual's future.
2. **Reporting bias.** 5,098 of 10,064 MMC3 records (51%) are excluded for
   having no explicit Yes/No inhibitor value, almost certainly non-randomly —
   records from inhibitor-focused work are likelier to have the field filled.
   The 16.8% positive rate describes this dataset's reporting, not population
   incidence.
3. **`pa_race` ranks second in global importance and is 58.7% missing.** What it
   most likely encodes is which cohorts were studied and reported, not a
   biological effect. It is retained because the reference analysis's clinical
   block includes it; no causal reading of it is defensible.
4. **Repeated genomic blocks.** Every record of one mutation carries an
   identical genomic block, so the training split contains 1,688 independent
   mutations behind its 3,124 rows.
5. **No external validation.** One dataset, one grouped split. **No clinical
   validation is claimed.**
6. **Modest discrimination.** ROC-AUC 0.765 with 42% precision at the operating
   threshold. Useful for screening, not for deciding.
7. **Model choice is weakly evidenced.** The four candidate families sit within
   roughly one standard deviation of each other in cross-validation.

## Repository history

This repository previously contained four competing application stacks, three
non-importing FastAPI backends, four Streamlit UIs, five chatbot
implementations, and around 100 aspirational markdown documents. A full audit is
in **[docs/AUDIT.md](docs/AUDIT.md)**; the resulting canonical choices are in
**[docs/CANONICAL.md](docs/CANONICAL.md)**.

Two findings worth knowing:

- **The models committed before the audit were never trained on a real dataset.**
  They were fitted on 30 fabricated rows whose label was a deterministic function
  of two input columns. They are preserved, unmodified and clearly labelled, in
  `ml/artifacts/legacy-synthetic-v0/` and are not servable.
- **`hemophilia_clinic.db` was committed to git** containing four accounts that
  shared the unsalted MD5 of `password123`. The file is now untracked, but
  untracking does not remove it from git history: those accounts should be
  treated as compromised.

The CHAMP pipeline that preceded this one was moved to `archive/legacy-champ/`
with `git mv`, so history follows each file. Its metrics are not comparable to
the ones on this page and must not be quoted for the current model — see
[archive/legacy-champ/README.md](archive/legacy-champ/README.md). Nothing was
deleted.

## License

MIT — see [LICENSE](LICENSE).
