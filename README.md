# Hemophilia Inhibitor-Risk Predictor

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/tests-98%20passing-2C6642)
![Status](https://img.shields.io/badge/status-research%20prototype-8A5D0B)

An explainable Hemophilia A inhibitor-risk prediction **research prototype**
based on CHAMP genomic data.

> **Medical disclaimer.** This is research decision-support software, not a
> diagnostic device. It is **not clinically validated**, has no external
> validation cohort, and must not be used for standalone diagnosis or treatment
> decisions. It never recommends a course of treatment. Estimates are attributed
> to an F8 **variant**, not to an individual patient — see
> [Limitations](#limitations).

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML pipeline and results](#ml-pipeline-and-results)
- [Explainability](#explainability)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Running locally](#running-locally)
- [Running the tests](#running-the-tests)
- [Docker](#docker)
- [API](#api)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Repository history](#repository-history)

## What it does

A clinician signs in, records a patient, enters that patient's F8 variant as
described in CHAMP, and receives a calibrated probability that such a variant
has a reported history of inhibitor development — together with SHAP and LIME
explanations of which genomic features drove the estimate. Predictions and
explanations are stored and viewable as history.

## Architecture

```
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │ TypeScript / Vite   │
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
          │  genomic_   │          │  explainability  │
          │   profiles  │          └────────┬─────────┘
          │  predictions│                   ▼
          │  explanations│         ┌──────────────────┐
          │  audit_logs │          │ champ-v1 artifact│
          └─────────────┘          │ model + preproc  │
                                   │ + metadata       │
                                   └──────────────────┘
```

Three deployable pieces: frontend, backend, database. ML inference runs inside
the backend process — a separate model service would add operational cost with
no benefit at this size. The model is loaded **once at startup**, not per
request.

## Dataset

`ml/data/champ.csv` — the CDC Hemophilia A Mutation Project variant registry.
**CHAMP is the only dataset used.** No WBDR, ATHN, hospital data, or synthetic
augmentation. The file is never modified; all normalisation happens in
`ml/preprocessing/champ.py`.

| | rows |
|---|---|
| In file | 4,050 |
| Labelled (`History of Inhibitor` = Yes/No) | **2,296** |
| — positive | 461 (20.08%) |
| Excluded: `Not reported` | 1,742 |
| Excluded: blank label | 12 |

Labelled + excluded = 4,050, asserted by the test suite so no record can be
dropped without appearing in the exclusion report.

Nine source features: `Variant Type`, `Mechanism`, `Domain`, `Subtype`,
`In Poly A`, `Reported Clinical Severity`, plus `exon_number`, `codon_number`
and `is_intron` derived from the `Exon` and `Codon` columns.

## ML pipeline and results

```
CHAMP → validate → normalise → label → SPLIT FIRST
      → fit preprocessor on TRAIN ONLY
      → model selection (5-fold CV, SMOTE inside folds)
      → isotonic calibration
      → threshold chosen on a validation split
      → evaluate ONCE on held-out test
      → predict → SHAP / LIME
```

Held-out test set (460 variants, 92 positive), produced by
`scripts/train_champ.py`:

| Metric | Value |
|---|---|
| ROC-AUC | **0.723** |
| PR-AUC | **0.501** (base rate 0.201) |
| Brier score | 0.134 |
| Precision @ threshold 0.175 | 0.311 |
| Recall @ threshold 0.175 | 0.772 |

**Read this honestly.** PR-AUC 0.50 against a 0.20 base rate is a 2.5× lift —
real signal. But at the F1-optimal threshold the model flags 228 variants to
catch 71 of 92 true positives: roughly two false alarms per true one. It is a
screening aid that errs toward sensitivity, not a decision rule. Accuracy is
deliberately not a headline metric — predicting "no inhibitor" for everything
scores 0.80 here and is useless.

Full detail, including model comparison and every caveat: **[docs/ML.md](docs/ML.md)**.

## Explainability

One service, `ml/explainability/service.py`, provides both:

- **SHAP** — exact TreeSHAP over the ensemble beneath the calibrator (~0.4s).
  Global feature importance and per-prediction local contributions.
- **LIME** — local linear approximation for individual predictions.

Both aggregate the 47 encoded columns back onto the 9 source CHAMP columns, so
an explanation never names `cat__Variant Type_Missense` and never names a
feature the caller did not supply. If an explainer fails, the response says so
rather than returning a fabricated attribution.

## Installation

Requires Python 3.11 and Node 20+.

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

Copy `.env.example` to `.env`. Nothing is committed with a real value.

| Variable | Default | Notes |
|---|---|---|
| `JWT_SECRET_KEY` | — | **Required in production.** In development a random per-process key is generated, so tokens do not survive a restart. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ENVIRONMENT` | `development` | `production` makes a missing/short secret fatal |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Exact origins, comma separated. Never `*`. |
| `DATABASE_PATH` | `hemophilia.db` | |
| `ML_MODEL_VERSION` | `champ-v1` | |
| `ML_ARTIFACTS_DIR` | `ml/artifacts` | |
| `BOOTSTRAP_ADMIN_EMAIL` / `_PASSWORD` | empty | If unset, **no account is seeded at all** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | |

Frontend (`frontend/.env`): `VITE_API_URL` — only `VITE_*` reaches the browser
bundle, so never put a secret there.

## Running locally

Two terminals:

```bash
uvicorn backend.main:app --reload --port 8000
```

```bash
cd frontend && npm run dev
```

Open http://localhost:3000, create an account, add a patient, then enter a
variant. `GET /health` reports database and model status.

To retrain (optional — `champ-v1` is committed):

```bash
python scripts/train_champ.py
```

## Running the tests

```bash
pytest
```

98 tests: data integrity, preprocessing, feature ordering, artifact
consistency, inference, thresholding, input validation, SHAP/LIME, API
contract, authentication, authorisation, failure handling, and one end-to-end
test that walks input → API → preprocessing → model → prediction → explanation
→ database → response.

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
GET    /api/predictions/schema
POST   /api/patients/{id}/predictions
GET    /api/predictions/{id}
GET    /api/predictions/{id}/explanation
GET    /api/patients/{id}/history
GET    /api/analytics                            GET /health
```

## Project structure

```
frontend/            React + TypeScript + Vite SPA (the production UI)
backend/             one FastAPI application
  core/              config, security
  routers/           auth, patients, predictions, analytics
  services/          adapter over the ml package
  db.py              the single SQLite layer
ml/
  data/champ.csv     the authoritative dataset
  preprocessing/     validation, normalisation, fitted ColumnTransformer
  models/            model registry and imbalance handling
  explainability/    SHAP + LIME
  artifacts/
    champ-v1/        model, preprocessor, metadata.json, metrics.json
    legacy-synthetic-v0/   preserved for provenance; see its PROVENANCE.md
  inference.py       the single predict() entrypoint
tests/               98 tests, including one end-to-end
scripts/train_champ.py
docker/              Dockerfiles, compose, nginx
docs/                AUDIT.md · CANONICAL.md · ML.md · API.md
archive/             superseded implementations, kept for reference
```

## Limitations

1. **Variant-level, not patient-level.** CHAMP rows are F8 variants and
   `History of Inhibitor` is a registry aggregation over reports for that
   variant. The output is variant-attributable risk, not an individual's
   probability of developing an inhibitor.
2. **Reporting bias.** 1,754 rows (43%) are excluded for having no reported
   inhibitor history, almost certainly non-randomly — variants studied in
   inhibitor-focused work are likelier to have the field filled. The 20.08%
   positive rate describes this registry's reporting, not population incidence.
3. **No external validation.** One registry, one random split. **No clinical
   validation is claimed.**
4. **Genomic only.** There is no authorised clinical dataset in this
   repository. No genomic + clinical fusion is performed or claimed.
5. **Modest discrimination.** ROC-AUC 0.72 with low precision at the operating
   threshold. Useful for screening, not for deciding.
6. **Model choice is weakly evidenced.** The four candidate models sat within
   one standard deviation of each other in cross-validation.

## Repository history

This repository previously contained four competing application stacks, three
non-importing FastAPI backends, four Streamlit UIs, five chatbot
implementations, and around 100 aspirational markdown documents. A full audit
is in **[docs/AUDIT.md](docs/AUDIT.md)**; the resulting canonical choices are in
**[docs/CANONICAL.md](docs/CANONICAL.md)**.

Two findings worth knowing:

- **The previously committed models were never trained on CHAMP.** They were fitted
  on 30 fabricated rows whose label was a deterministic function of two input
  columns. They are preserved, unmodified and clearly labelled, in
  `ml/artifacts/legacy-synthetic-v0/` and are not servable.
- **`hemophilia_clinic.db` was committed to git** containing four accounts that
  shared the unsalted MD5 of `password123`. The file is now untracked, but
  untracking does not remove it from git history: those accounts should be
  treated as compromised.

Superseded code was moved to `archive/` with `git mv`, so history follows each
file. Nothing was deleted.

## License

MIT — see [LICENSE](LICENSE).
