# Canonical components

One implementation per concern. Anything not listed here as canonical is either
superseded (and lives under `archive/`) or does not exist yet.

Decisions were made from the Phase 0 audit on evidence of actual use and code
quality — not on which implementation was newest. That audit described the
retired variant-level pipeline and was removed with it; its conclusions survive
as the table below.

| Concern | Canonical | Chosen over | Why |
|---|---|---|---|
| **Frontend** | `frontend/` — React 18 + TypeScript + Vite | 4 Streamlit apps (`app.py`, `app_frontend.py`, `streamlit_app.py`, `streamlit_medical_app.py`), `clean_project/` | Only implementation with a typed API client, routing, state management and real endpoint calls. The Streamlit apps were mocks or duplicates. |
| **Backend** | `backend_api.py` | `backend/`, `fastapi_backend/`, root `main.py` | Only backend that (a) imports at all, (b) serves the `/api/*` routes the SPA already calls, (c) has working JWT auth and per-user row scoping. `backend/` fails on import; `fastapi_backend/` has no auth. |
| **Database** | `database.py` + the schema in `backend_api.py`, consolidated | `database_optimized.py`, `database_old/`, `backend/models_orm.py`, `auth_models.py` | `database.py` has the richest schema and binds every SQL parameter. The competing layers had no live callers. |
| **Authentication** | JWT in `backend_api.py`, moving to bcrypt | 8 `auth_*.py` modules, `user_auth.py`, `security.py`, `backend/auth.py` | The `auth_*` stack belonged to root `main.py`, which cannot start (imports an archived module). Its bcrypt approach is kept; its plumbing is not. |
| **ML inference** | `ml/inference.py` | `backend/ml_utils.py`, `fastapi_backend/services/prediction_service.py`, the formula in `backend_api.py` | None of the three was correct. `ml_utils` silently zero-filled 9 of 20 features; the others had no model at all. Now driven by the artifact's own `FeatureSpec`, so it cannot name a column the model was not fitted on. |
| **Dataset** | `ml/data/BVTH_VTH-2024-000215-mmc2.csv` + `…-mmc3.csv`, read only through `ml/preprocessing/hemophilia_a.py` | a superseded variant-level dataset and its `HemophiliaA_*.csv` derivatives (all under `archive/`) | MMC3 is a clinical-record table joinable to the MMC2 mutation description, which gives the model clinical assay values alongside the genomic block. The retired predecessor was variant-level only and carried no assay values. MMC3's records are aggregated per `mut_id`, so a modelling row is one mutation. |
| **Preprocessing** | `ml/preprocessing/hemophilia_a.py` — one fitted `ColumnTransformer` per feature set | the retired pre-MMC2/MMC3 preprocessor (archived), `data_fusion.py` | `data_fusion` imputed over the full dataset before splitting, and shipped a duplicate and a constant feature. The retired preprocessor was correct for its own variant-level column set but cannot express the MMC2/MMC3 feature blocks, the per-`mut_id` aggregation, or the grouped split. |
| **Train/test split** | `GroupShuffleSplit` on `mut_id`, in `scripts/train_inhibitor_model.py` | stratified row-level `train_test_split` | The unit of observation is a **mutation**, not a clinical record: MMC3's records are aggregated per `mut_id` before the split, so each mutation contributes exactly one row. Grouping on `mut_id` is what enforces that — splitting the raw clinical records row-wise would put several records of the same mutation, sharing an identical genomic block, on both sides of the split. |
| **Model artifacts** | `ml/artifacts/<version>/` with `metadata.json` | 14 loose `.pkl` files at the repo root | Versioned, self-describing, and loadable without guessing which file matches which feature list. |
| **SHAP / LIME** | `ml/explainability/service.py` | `ml/explainability/explainers.py`, `shap_explainability.py`, `backend/services/explainability.py`, `pages/shap_explainability*.py` | `service.py` is what the API and tests call: one service, both methods, reporting against source columns rather than encoded names. `explainers.py` holds the generic explainer classes it grew out of and is no longer on any call path. |
| **Configuration** | `config.py` → `backend/core/config.py` | `fastapi_backend/config.py`, `auth_config.py`, hardcoded literals | Existing env-driven `Settings` class kept and hardened; the duplicates are archived. |
| **Logging** | `logging_config.py` | scattered `print()` calls | Already structured; retained. |
| **Testing** | `tests/` (pytest) | 3 root scripts with 0 assertions | There was nothing to preserve. |

## Model versions

One artifact per prediction mode, all trained by
`scripts/train_inhibitor_model.py` on the same grouped, one-row-per-mutation
split.

| Version | Feature set | Status |
|---|---|---|
| `mmc2-mmc3-v1` | MMC2 genomic + MMC3 clinical, aggregated per `mut_id` | **Default.** <!-- TODO: state how it compares to the single-source artifacts, from a real training run. Do not write a number here until then. --> |
| `mmc2-genomic-v1` | MMC2 only | Selectable per request. |
| `mmc3-clinical-v1` | MMC3 only, aggregated per `mut_id` | Selectable per request. |
| `legacy-synthetic-v0` | — | Preserved, never default. Trained on fabricated rows; see its `PROVENANCE.md`. Requesting it raises rather than serving. |

The artifact from the retired pre-MMC2/MMC3 pipeline went with that pipeline and
now lives under `archive/`, outside `ml/artifacts/` — so `available_versions()`
no longer lists it and the API cannot serve it. Its recorded metrics describe a
superseded variant-level dataset and must not be quoted for these models.

## Database

`case_records` replaced `genomic_profiles`. The old table's fixed variant-level
columns (`variant_type`, `mechanism`, `domain`, …) cannot hold an MMC2/MMC3
record, and a fixed-column replacement would need migrating every time the
feature resolution changes — so the validated feature dict is stored as JSON
alongside its feature set. `db.migrate_legacy_schema_tables` renames the
pre-MMC2/MMC3 tables to `legacy_pre_mmc_*` rather than dropping them; users and
patients are untouched.

## What is deliberately absent

Per the "no unnecessary complexity" constraint, this project does **not** use Redis,
Celery, message queues, Kubernetes, GraphQL, an ORM layer, or a second database.
ML inference runs inside the backend process; a separate model service would add
operational cost with no benefit at this size.

The chatbot (5 competing implementations, ~130 KB) is archived. It was not part of
the system's stated purpose and none of the implementations was reachable from the
SPA.
