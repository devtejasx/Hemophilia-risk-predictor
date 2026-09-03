# Canonical components

One implementation per concern. Anything not listed here as canonical is either
superseded (and lives under `archive/`) or does not exist yet.

Decisions were made from the Phase 0 audit (`docs/AUDIT.md`) on evidence of actual
use and code quality — not on which implementation was newest.

| Concern | Canonical | Chosen over | Why |
|---|---|---|---|
| **Frontend** | `frontend/` — React 18 + TypeScript + Vite | 4 Streamlit apps (`app.py`, `app_frontend.py`, `streamlit_app.py`, `streamlit_medical_app.py`), `clean_project/` | Only implementation with a typed API client, routing, state management and real endpoint calls. The Streamlit apps were mocks or duplicates. |
| **Backend** | `backend_api.py` | `backend/`, `fastapi_backend/`, root `main.py` | Only backend that (a) imports at all, (b) serves the `/api/*` routes the SPA already calls, (c) has working JWT auth and per-user row scoping. `backend/` fails on import; `fastapi_backend/` has no auth. |
| **Database** | `database.py` + the schema in `backend_api.py`, consolidated | `database_optimized.py`, `database_old/`, `backend/models_orm.py`, `auth_models.py` | `database.py` has the richest schema and binds every SQL parameter. The competing layers had no live callers. |
| **Authentication** | JWT in `backend_api.py`, moving to bcrypt | 8 `auth_*.py` modules, `user_auth.py`, `security.py`, `backend/auth.py` | The `auth_*` stack belonged to root `main.py`, which cannot start (imports an archived module). Its bcrypt approach is kept; its plumbing is not. |
| **ML inference** | `ml/inference.py` | `backend/ml_utils.py`, `fastapi_backend/services/prediction_service.py`, the formula in `backend_api.py` | None of the three was correct. `ml_utils` silently zero-filled 9 of 20 features; the others had no model at all. Now driven by the artifact's own `FeatureSpec`, so it cannot name a column the model was not fitted on. |
| **Dataset** | `ml/data/BVTH_VTH-2024-000215-mmc2.csv` + `…-mmc3.csv`, read only through `ml/preprocessing/hemophilia_a.py` | `ml/data/champ.csv` (now `archive/legacy-champ/`), `HemophiliaA_*.csv` derivatives | MMC3 is a clinical-record table joinable to the MMC2 mutation description, which gives the model clinical assay values alongside the genomic block. CHAMP is variant-level only. |
| **Preprocessing** | `ml/preprocessing/hemophilia_a.py` — one fitted `ColumnTransformer` per feature set | `ml/preprocessing/champ.py` (archived), `data_fusion.py` | `data_fusion` imputed over the full dataset before splitting, and shipped a duplicate and a constant feature. `champ.py` was correct for CHAMP's nine columns but cannot express the MMC2/MMC3 feature blocks or the grouped split. |
| **Train/test split** | `GroupShuffleSplit` on `mut_id`, in `scripts/train_inhibitor_model.py` | stratified row-level `train_test_split` | One mutation appears in up to 104 clinical records that share an identical genomic block. A row-level split puts some of them in train and the rest in test. |
| **Model artifacts** | `ml/artifacts/<version>/` with `metadata.json` | 14 loose `.pkl` files at the repo root | Versioned, self-describing, and loadable without guessing which file matches which feature list. |
| **SHAP / LIME** | `ml/explainability/service.py` | `ml/explainability/explainers.py`, `shap_explainability.py`, `backend/services/explainability.py`, `pages/shap_explainability*.py` | `service.py` is what the API and tests call: one service, both methods, reporting against source columns rather than encoded names. `explainers.py` holds the generic explainer classes it grew out of and is no longer on any call path. |
| **Configuration** | `config.py` → `backend/core/config.py` | `fastapi_backend/config.py`, `auth_config.py`, hardcoded literals | Existing env-driven `Settings` class kept and hardened; the duplicates are archived. |
| **Logging** | `logging_config.py` | scattered `print()` calls | Already structured; retained. |
| **Testing** | `tests/` (pytest) | 3 root scripts with 0 assertions | There was nothing to preserve. |

## Model versions

One artifact per prediction mode, all trained by
`scripts/train_inhibitor_model.py` on the same grouped split.

| Version | Feature set | Status |
|---|---|---|
| `mmc-merged-v1` | genomic + clinical, 29 columns | **Default.** Best on every ranking metric. |
| `mmc-genomic-v1` | MMC2 only, 20 columns | Selectable per request. |
| `mmc-clinical-v1` | MMC3 only, 9 columns | Selectable per request. |
| `legacy-synthetic-v0` | — | Preserved, never default. Trained on 30 fabricated rows; see its `PROVENANCE.md`. Requesting it raises rather than serving. |

`champ-v1` was retired with the CHAMP pipeline and lives in
`archive/legacy-champ/champ-v1/`. It is outside `ml/artifacts/`, so
`available_versions()` no longer lists it and the API cannot serve it. Its
metrics describe a different dataset and must not be quoted for these models.

## Database

`case_records` replaced `genomic_profiles`. The old table's fixed CHAMP columns
(`variant_type`, `mechanism`, `domain`, …) cannot hold an MMC2/MMC3 record, and
a fixed-column replacement would need migrating every time the feature
resolution changes — so the validated feature dict is stored as JSON alongside
its feature set. `db.migrate_legacy_champ_tables` renames the CHAMP-era tables
to `legacy_champ_*` rather than dropping them; users and patients are untouched.

## What is deliberately absent

Per the "no unnecessary complexity" constraint, this project does **not** use Redis,
Celery, message queues, Kubernetes, GraphQL, an ORM layer, or a second database.
ML inference runs inside the backend process; a separate model service would add
operational cost with no benefit at this size.

The chatbot (5 competing implementations, ~130 KB) is archived. It was not part of
the system's stated purpose and none of the implementations was reachable from the
SPA.
