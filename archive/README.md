# Archive

Superseded code, kept for reference and provenance. **Nothing here is imported,
deployed, or tested.** The canonical implementations are listed in
`docs/CANONICAL.md`.

| Directory | Contents | Why archived |
|---|---|---|
| `legacy-backends/` | `backend/`, `fastapi_backend/`, `main.py`, 8 `auth_*.py`, `user_auth.py`, `security.py`, `models_schema.py`, `init_db.py`, `logging.py` | Three competing FastAPI applications plus a fourth auth stack. `backend/` could not import; `fastapi_backend/` had no auth; `main.py` imported an already-archived module. Canonical backend is `backend_api.py`. |
| `legacy-streamlit/` | `app.py`, `app_frontend.py`, `streamlit_app.py`, `streamlit_medical_app.py`, `clean_project/`, `pages/`, `streamlit_pages/`, `streamlit_utils/`, `components/`, `styles/`, `utils/`, `.streamlit/` | Four Streamlit UIs and two parallel page sets. The canonical production UI is the React SPA in `frontend/`. `app.py` — previously the deployed entrypoint — was a mock with `numpy.random` predictions. |
| `legacy-chatbot/` | `gpt_chatbot.py`, `simple_chatbot.py`, `clinical_*.py`, `chatbot_*.py`, `local_model.py`, `services/` | Five chatbot implementations, none reachable from the SPA and none part of the system's stated purpose. |
| `legacy-infra/` | `database_optimized.py`, `database_old/`, `cache_*.py`, `background_tasks.py`, `dashboard_persistence.py`, `gunicorn_config.py`, 4 redundant `requirements_*.txt`, setup/quickstart scripts, the 3 assertion-free test scripts | Competing data and infrastructure layers with no live callers. |
| `legacy-ml/` | `train.py`, `data_fusion.py`, `evaluation.py`, `predict.py`, `shap_explainability.py`, `explainers.py`, `ensemble_models.py`, `imbalance_handler.py`, `train_optimized.py`, `catboost_info/`, `temp_*` | The synthetic-data training pipeline. Preserved because it documents exactly how `ml/artifacts/legacy-synthetic-v0/` was produced. `explainers.py` is the generic `SHAPExplainer` / `LIMEExplainer` / `ExplainabilityComparison` trio that `ml/explainability/service.py` grew out of; nothing imported it. `ensemble_models.py` and `imbalance_handler.py` came from `ml/models/`, where nothing in the active tree imported them — `train.py` here is their only caller, and its bare `from ensemble_models import ...` now resolves from this directory. |
| `legacy-frontend/` | `Charts.tsx`, `Chatbot.tsx`, `ChatBox.tsx`, `FormField.tsx`, `Navbar.tsx`, `PatientCard.tsx`, `Predictions.tsx`, `SHAPAnalysis.tsx` | Earlier React components superseded by the current `frontend/src/` tree. |
| `legacy-dbs/` | `hemophilia.db`, `hemophilia_clinic.db` | Database files committed before the Phase 0 audit. `hemophilia_clinic.db` held four accounts sharing the unsalted MD5 of a common password; treat those credentials as compromised. |
| `legacy-artifacts/` | `legacy-synthetic-v0/` | The model files committed before the Phase 0 audit. Fitted on 30 fabricated rows whose label was a deterministic function of two input columns, so every metric attached to them describes a memorised hand-written rule. Moved out of `ml/artifacts/` so they cannot be loaded as a model version. |
| `session-docs/` | ~90 one-shot status/summary documents from earlier sessions | Aspirational; none authoritative. |
| `variants/` | `app_*.py`, `api*.py` iterations | Superseded app and API variants. |

## Restoring something

Everything was moved with `git mv`, so history follows the file:

```bash
git log --follow archive/legacy-ml/train.py
```

Most of these files are not merely superseded but actively broken, so check why
a thing was archived before restoring it — the table above is the short version.
