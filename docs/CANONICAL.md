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
| **ML inference** | `ml/inference.py` (Phase 1, commit 2) | `backend/ml_utils.py`, `fastapi_backend/services/prediction_service.py`, the formula in `backend_api.py` | None of the three was correct. `ml_utils` silently zero-filled 9 of 20 features; the others had no model at all. |
| **Preprocessing** | `ml/preprocessing/` — one fitted `ColumnTransformer` | `data_fusion.py` | `data_fusion` imputed over the full dataset before splitting, and shipped a duplicate and a constant feature. Its feature *ideas* are reused; its leakage is not. |
| **Model artifacts** | `ml/artifacts/<version>/` with `metadata.json` | 14 loose `.pkl` files at the repo root | Versioned, self-describing, and loadable without guessing which file matches which feature list. |
| **SHAP / LIME** | `ml/explainability/explainers.py` | `shap_explainability.py`, `backend/services/explainability.py`, `pages/shap_explainability*.py` | The only module implementing both SHAP **and** LIME. The alternatives were Streamlit-coupled or duplicated. |
| **Configuration** | `config.py` → `backend/core/config.py` | `fastapi_backend/config.py`, `auth_config.py`, hardcoded literals | Existing env-driven `Settings` class kept and hardened; the duplicates are archived. |
| **Logging** | `logging_config.py` | scattered `print()` calls | Already structured; retained. |
| **Testing** | `tests/` (pytest) | 3 root scripts with 0 assertions | There was nothing to preserve. |

## Model versions

| Version | Status | Notes |
|---|---|---|
| `champ-v1` | **Default.** Trained on `ml/data/champ.csv`. | The model the API serves. |
| `legacy-synthetic-v0` | Preserved, never default. | Trained on 30 fabricated rows. See its `PROVENANCE.md`. Loadable only by explicit version request, for comparison. |

## What is deliberately absent

Per the "no unnecessary complexity" constraint, this project does **not** use Redis,
Celery, message queues, Kubernetes, GraphQL, an ORM layer, or a second database.
ML inference runs inside the backend process; a separate model service would add
operational cost with no benefit at this size.

The chatbot (5 competing implementations, ~130 KB) is archived. It was not part of
the system's stated purpose and none of the implementations was reachable from the
SPA.
