# Hemophilia Risk Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-prototype-yellow)

A Streamlit-based demo UI for a hypothetical hemophilia clinical risk-assessment tool. This README documents **what is actually deployed and runnable today**, verified directly against the code and deployment config — not the more ambitious system described in this repo's many other docs (see [Repository State](#repository-state--important) below, which you should read before relying on anything else in this repo).

## Table of Contents

- [Repository State (important)](#repository-state--important)
- [What Actually Runs](#what-actually-runs)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration-environment-variables)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker-experimental-unused-path)
- [Testing](#testing)
- [Deployment](#deployment)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Known Issues / Inconsistencies](#known-issues--inconsistencies)
- [Contributing](#contributing)
- [License](#license)
- [Future Improvements](#future-improvements)

## Repository State (important)

This repository contains **at least four different, mutually inconsistent stories about what the application is**, plus roughly 100 top-level markdown files documenting features that are not present in the code that's actually deployed. Before using this repo, understand:

1. **What's actually deployed** (per `Procfile` and `render.yaml`): a single-file Streamlit app, `app.py`, run standalone with no backend service.
2. **What the previous root README described**: a "Hemophilia AI Platform" with a real GPT-4 chatbot, a SQL database with 6 tables, ML risk prediction via `rf.pkl`/`xgb.pkl`, and PDF report generation. **This does not match `app.py`.**
3. **What `docker-compose.yml` describes**: a split architecture — `app_frontend.py` (Streamlit) calling `backend_api.py` (FastAPI) over HTTP, plus a placeholder database container. This is a real, self-consistent architecture, but it is **not** what `Procfile`/`render.yaml` deploy.
4. **What `clean_project/` describes**: a from-scratch modular rewrite (its own `README.md` frames it as "v2.0", consolidating "8 duplicate app files, 4 duplicate API files, 8 auth modules, 5 chatbot implementations" into one app). It is fully self-contained but **not referenced by anything outside its own folder** — nothing wires it up as the real entrypoint.

This README documents story #1 (the one that's actually live), and calls out the others so you don't spend time debugging code paths that were never wired into deployment.

## What Actually Runs

`app.py` at the repo root, verified by reading it directly:

- A **single-page Streamlit app** (auto-logs in a hardcoded user, "Dr. Sarah Chen" — there is no real authentication).
- Patient data is **hardcoded in-memory** (`PAT001`, `PAT002`, `PAT003`) — there is no database read/write despite `hemophilia.db` and `hemophilia_clinic.db` files sitting in the repo.
- "Risk prediction" is a hand-written weighted formula over `numpy.random` values (`generate_sample_prediction()`) — **no ML model is loaded**. `app.py` imports `pickle` but never calls `pickle.load`, and none of the 14 committed `.pkl` files (`catboost.pkl`, `rf.pkl`, `xgb.pkl`, `lightgbm.pkl`, ensemble models, etc.) are read anywhere in this file.
- SHAP explainability charts are **hardcoded static numbers**, not real SHAP output, despite `shap` being a dependency.
- The chatbot is **simple keyword matching** (e.g. `if "help" in message`) — no OpenAI/GPT-4 call, despite prior docs describing a GPT-4 integration.

In short: this is a UI mockup with realistic-looking but fabricated data, useful for demoing the intended workflow, not a working clinical risk model.

## Tech Stack

From the root `requirements.txt` (a full `pip freeze` dump, ~140 packages — see [Known Issues](#known-issues--inconsistencies)):

- **Streamlit 1.55** — the only UI actually deployed
- **Plotly** — charts within `app.py`
- **pandas / numpy** — data handling
- Present as dependencies but **not exercised by the deployed app**: `fastapi`, `uvicorn`, `scikit-learn`, `xgboost`, `shap`, `torch`, `transformers`, a full Supabase client stack (`supabase`, `postgrest`, `realtime`, `storage3`), `SQLAlchemy`, `psycopg2-binary`, `python-jose`, `reportlab`, `pygame`, `pytest`

## Project Structure

```
Hemophilia-risk-predictor/
├── app.py                    # ACTUAL deployed entrypoint (Procfile/render.yaml) — mock Streamlit demo
├── requirements.txt           # full pip-freeze dump used by the real deployment
├── Procfile / render.yaml     # deploy app.py as a standalone Streamlit service
├── build.sh                    # pip install -r requirements.txt
├── runtime.txt                 # python-3.11.9 (conflicts with .python-version, see Known Issues)
│
├── app_backup.py, app_frontend.py, app_optimized.py,
│   app_refactored.py, app_unified.py, app_updated.py    # unused alternate/iterated versions of app.py
├── api.py, api_optimized.py, api_production.py, api_updated.py  # unused alternate FastAPI stubs
│
├── backend/                   # a full FastAPI package (auth, routers, services) — not deployed
├── fastapi_backend/           # a SECOND, independent FastAPI package with its own docs — not deployed
├── backend_api.py             # a THIRD standalone FastAPI app — only used via Dockerfile.backend
│
├── frontend/                  # separate Vite + React + TypeScript SPA, expects a backend on :8000/api — not deployed
├── clean_project/              # self-contained "v2.0" modular rewrite — not wired to anything outside itself
│
├── pages/, streamlit_pages/    # two separate sets of Streamlit multipage files, neither imported by app.py
├── components/, services/, styles/, utils/   # root-level packages, duplicated again inside clean_project/
│
├── *.pkl                       # 14 committed model artifacts, unused by the deployed app.py
├── hemophilia.db, hemophilia_clinic.db       # committed SQLite DBs, unused by the deployed app.py
├── champ.csv, clinical.csv, genomic.csv, X_test.csv, y_test.csv, ...  # committed datasets
├── catboost_info/              # committed CatBoost training logs
│
├── docker-compose.yml, Dockerfile.backend, Dockerfile.frontend  # describes a DIFFERENT architecture
│                                                                    (app_frontend.py + backend_api.py), not used by Procfile/render.yaml
│
└── ~100 top-level *.md files   # overlapping guides/summaries for features not present in the deployed app
```

## Prerequisites

- Python 3.11 (matches `runtime.txt`, used by the real Render deployment)

## Installation

```bash
git clone https://github.com/devtejasx/Hemophilia-risk-predictor.git
cd Hemophilia-risk-predictor
pip install -r requirements.txt
```

## Configuration (Environment Variables)

`render.yaml` declares one env var for the deployed service:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Declared in `render.yaml`/`.env.example`, but **not actually read anywhere in `app.py`** — the deployed chatbot uses keyword matching, not the OpenAI API. |

`.env.example` at the root additionally documents a large set of production-style settings (`DATABASE_URL`, SMTP, Sentry/Datadog keys, rate limiting, etc.) that correspond to the `backend_api.py`/`fastapi_backend/` code paths, not to the deployed `app.py`. If you're only running `app.py`, none of these are required.

## Running Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. No `.env`, database, or model files are required for this to work, since `app.py` doesn't read any of them.

## Running with Docker (experimental, unused path)

`docker-compose.yml` defines a three-service topology that is **not the same application as `app.py`**:

```bash
docker-compose up --build
```

| Service | Runs | Port |
|---|---|---|
| `backend` | `backend_api.py` via `Dockerfile.backend` (uvicorn) | 8000 |
| `frontend` | `app_frontend.py` via `Dockerfile.frontend` (Streamlit), configured with `API_BASE_URL=http://backend:8000` | 8501 |
| `db` | a bare Alpine placeholder container — does not actually run a database engine | — |

This is a coherent design for a real frontend/backend split, but it's disconnected from what `Procfile`/`render.yaml` actually deploy, and the `db` service is a stub.

## Testing

No real automated test suite exists. Three loose scripts sit at the repo root:
- `auth_test.py`, `test_model_loading.py`, `test_pickle_load.py` — manual smoke-test scripts (print statements, no `assert`s, not organized as a pytest suite)

There is no `tests/` directory, no `pytest.ini`/`conftest.py`, and no CI to run anything. `clean_project/README.md` references `pytest --cov` and a `tests/` folder, but that test infrastructure does not exist in the repo.

## Deployment

The only deployment path actually configured:

1. **Render** (`render.yaml`): builds via `build.sh` (`pip install -r requirements.txt`), starts with `streamlit run app.py --server.port=$PORT --server.headless=true --server.enableCORS=false`.
2. **Procfile** (Heroku-style): identical `streamlit run app.py` command.

Both deploy the mock demo described in [What Actually Runs](#what-actually-runs) — no backend, no database, no ML inference.

## Security Considerations

- The deployed `app.py` **auto-logs in a hardcoded user** with no credential check — there is no real authentication on the live path.
- `.gitignore` excludes `.env/` (a directory) but **not `.env`** (a file) — if a real `.env` were ever created here, it would not be excluded from git by the current `.gitignore`.
- `.gitignore` also lists `.streamlit/` as ignored, yet `.streamlit/config.toml` is committed anyway — the ignore rule was added after the file was already tracked.
- Real secrets-handling code exists (`auth_security.py`, `security.py`, `backend/security.py`, JWT/bcrypt dependencies) but belongs to the unused `backend`/`fastapi_backend`/`backend_api.py` paths, not the deployed app — don't assume the live app has this protection.
- Sample/synthetic patient data is committed (`champ.csv`, `clinical.csv`, `genomic.csv`, `patients_backup_20260327_124514.csv`) — confirm this is genuinely synthetic before treating it as safe to keep in a public repo, given the healthcare subject matter.

## Troubleshooting

- **"Where's the real ML prediction?"** — there isn't one wired up in the deployed app; see [What Actually Runs](#what-actually-runs).
- **`streamlit run app.py` shows different behavior than the old README described** — the old README documented a different, unimplemented version of the app; trust this README and the code instead.
- **Python version mismatches** — `.python-version` says `3.10.13` while `runtime.txt` (used by Render) says `python-3.11.9`. Use 3.11 to match what's actually deployed.
- **`requirements.txt` fails to parse in some editors** — the file is UTF-16LE encoded (an artifact of `pip freeze > requirements.txt` on Windows PowerShell); re-save as UTF-8 if you need to hand-edit it.

## Known Issues / Inconsistencies

Flagged here rather than silently fixed, since resolving them requires deciding which of the competing implementations is the "real" project:

1. **The deployed app doesn't match any of the project's own documentation.** The (previous) root README, `clean_project/README.md`, and the various `*_SUMMARY.md`/`*_GUIDE.md` files describe GPT-4 chatbots, real ML models, and database persistence that don't exist in the code path Render/Procfile actually run.
2. **At least 4 competing "canonical app" implementations**, none pointing to each other: root `app.py` (deployed), `clean_project/app.py` (self-described rewrite, unreferenced), `docker-compose.yml`'s `app_frontend.py` + `backend_api.py` pair (a different real architecture, unreferenced by the Streamlit-only deploy), and a React `frontend/` SPA expecting a backend on port 8000 (also unreferenced by the deploy).
3. **Three separate, non-shared FastAPI backends** exist: `backend/`, `fastapi_backend/`, and root `backend_api.py`. They overlap in responsibility (patients, predictions, chat, auth) but share no code.
4. **Six or more alternate/duplicate app files** (`app_backup.py`, `app_optimized.py`, `app_refactored.py`, `app_unified.py`, `app_updated.py`, plus 4 `api*.py` variants) sit at the root with no indication of which — if any — should be kept.
5. **14 `.pkl` model files, 2 SQLite databases, and 8 CSV datasets are committed directly to git**, none of them read by the deployed `app.py`.
6. **~100 top-level markdown files** with heavy overlap (e.g. 5 `AUTHENTICATION_*.md` files, 8 `SHAP_*.md` files, 4 `FASTAPI_*.md` files) make it very hard to find authoritative information.
7. **Inconsistent Python version pinning** — `.python-version` (3.10.13) vs `runtime.txt` (3.11.9).
8. **`.gitignore` gaps** — doesn't exclude `.env` (only `.env/`), and excludes `.streamlit/` after `.streamlit/config.toml` was already committed.
9. **7 different `requirements*.txt` files** at various paths (`requirements.txt`, `requirements_auth.txt`, `requirements_optimized.txt`, `requirements_production.txt`, `requirements_streamlit.txt`, `backend/requirements.txt`, `fastapi_backend/requirements.txt`, `clean_project/requirements.txt`) with no documentation of which applies where.
10. **No LICENSE file, no `.github/` CI workflows, no real automated test suite.**

## Contributing

1. Fork the repository
2. Before adding new features, read [Repository State](#repository-state--important) — pick one of the existing implementations to build on rather than adding a fifth
3. Open a pull request

## License

No `LICENSE` file exists in this repository.

## Future Improvements

- Pick one canonical implementation (the deployed `app.py`, the `docker-compose.yml` split architecture, or `clean_project/`) and delete or clearly archive the others
- Wire the deployed app to an actual trained model (the `.pkl` files already exist) instead of `numpy.random`-based mock predictions
- Consolidate the ~100 markdown files into a single, current set of docs
- Remove committed model/database/dataset artifacts from git history if they contain anything beyond synthetic sample data, and add them to `.gitignore` going forward
- Fix the `.env` gitignore gap and the Python version mismatch
- Add a real automated test suite and CI
