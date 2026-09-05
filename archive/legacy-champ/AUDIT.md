# Phase 0 audit — what was actually in this repository

Audited 2026-08-31 against `main @ 05e81d5`. Every claim below was checked by
running the code, loading the artifact, or querying the data — not by reading
docstrings, which in this repository were frequently aspirational.

The consolidation decisions that follow from this audit are in
[CANONICAL.md](CANONICAL.md).

> **Historical record.** This page describes the repository *as it was on
> 2026-08-31*. Its references to CHAMP are deliberate and are not describing the
> current system: the application has since moved to the MMC2 + MMC3 dataset.
> The CHAMP pipeline it audits now lives in `archive/legacy-champ/`. For what
> the system does today, read [ML.md](ML.md) and the
> [README](../README.md).

## Headline finding

**No committed model was trained on CHAMP.** `train.py` loaded `genomic.csv` +
`clinical.csv` — 30 hand-written rows. `champ.csv` was read by nothing; it appeared
once in the whole codebase, as an unused default argument in `data_fusion.py`.

That synthetic label is a deterministic function of two input columns:

```
target         0  1        target    0   1
mutation_type              severity
frameshift     7  0        mild      7   0
intron22       0  8        moderate  8   0
missense       8  0        severe    0  15
nonsense       0  7
```

Perfect separation, 24 training rows. Any accuracy figure attached to those `.pkl`
files measures a rule someone typed by hand. See
`ml/artifacts/legacy-synthetic-v0/PROVENANCE.md`.

The real CHAMP data is sound: 4,050 rows, **2,296 labelled at 20.1% positive**, one
row per distinct `HGVS cDNA` variant, and the signal matches the published
literature (large structural change 56% inhibitor incidence, missense 8.8%).

## Architecture found

Five disconnected stacks, none referencing another:

| Stack | Entrypoint | State when audited |
|---|---|---|
| Deployed (`Procfile`/`render.yaml`) | `app.py` Streamlit | Ran, but a mock: `numpy.random` risk, hardcoded SHAP bars, auto-login |
| `docker-compose.yml` | `backend_api.py` + `app_frontend.py` | Build failed (`COPY .env`, no `.env`); `db` service was a bare Alpine container |
| `backend/` | FastAPI package | **Could not import** |
| `fastapi_backend/` | 2nd FastAPI package | Unreferenced; no auth; empty `database/connection.py` |
| `clean_project/` | 3rd Streamlit app | Unreferenced |

Plus root `main.py`, a 4th FastAPI app importing an already-archived module.

## Findings

Severity: **C** critical, **W** warning, **I** informational.

| # | Sev | Finding | Evidence |
|---|---|---|---|
| B-01 | C | Models not trained on CHAMP | `champ.csv` has zero readers |
| B-02 | C | Training label is a lookup table | crosstabs above |
| B-03 | C | 24 training rows / 6 test rows | `evaluation_report.json`, `"models": {}` |
| B-04 | C | **Train/inference mismatch:** `backend/ml_utils.py` filled 9 of 20 features with `0`, including `exon_risk` which was constant `1.0` in training. Still returned a confident probability. | reproduced: rf 0.715/xgb 0.854 vs 0.985/0.903 for a correctly-shaped row |
| B-05 | C | Leakage: median/mode imputation over the full dataset before `train_test_split` | `data_fusion.engineer_features()` |
| B-06 | W | `mutation_severity` is a copy-paste duplicate of `mutation_code`; `exon_risk` is constant (tests `isinstance(x, str)` on an int column) | |
| B-07 | W | `lightgbm.pkl` is a single-leaf constant model | `num_leaves=1`, no splits |
| B-08 | W | `rf.pkl` = `randomforest.pkl` = `model_ensemble.pkl`; `xgb.pkl` = `xgboost.pkl` | byte-identical (md5) |
| B-09 | W | **No calibration and no threshold selection existed anywhere** | grep returns nothing |
| B-10 | C | `backend/` failed on line 8 | `ImportError: cannot import name 'logging' from 'fastapi'`; `TypeError: FastAPI.get() got an unexpected keyword argument 'redirect_url'` |
| B-11 | C | React frontend did not build | `tsc`: 49 diagnostics (13 real type errors); `vite build`: `[postcss] The 'focus-visible' class does not exist` |
| B-12 | W | SPA called `/api/*`; only `backend_api.py` served that shape. 401 interceptor redirected to `/login`, a route that did not exist | |
| B-13 | W | The one reachable backend had no ML — risk was a hand-weighted formula over non-CHAMP fields | `backend_api.py:429` |
| B-14 | C | **Risk adjusted by ethnicity** (`+0.05` for "african", `−0.02` for "asian") and blood type. No source, no citation, no CHAMP column. | `backend/ml_utils.calculate_clinical_adjustment()` |
| B-15 | C | System recommended treatment ("Consider hospitalization for immune tolerance induction") as a function of a model score | `generate_recommendations()` |
| B-16 | W | Model failure fell through to a hardcoded formula in the same response shape; `confidence` was set equal to the probability | |
| B-17 | W | Models re-read from disk on every request | `load_models()` inside `predict_inhibitor_risk()` |
| B-18 | W | `requirements.txt` missing `lime`, `lightgbm`, `catboost`, `imbalanced-learn`, `python-jose`; also UTF-16LE | `train.py` could not run as declared |
| B-19 | C | **Working credentials committed to git**: 4 accounts sharing unsalted MD5 `ef92b778…` = `password123` | `hemophilia_clinic.db` |
| B-20 | W | Three competing password schemes: MD5 (in the DB), unsalted SHA-256, bcrypt | |
| B-21 | W | **Zero tests** — 0 `assert` statements across all 3 `*test*.py` files | `pytest --collect-only`: 1 collected, 1 error |
| B-22 | I | No `predictions` or `explanations` table in the canonical schema | |

## Security

Fixed in Phase 1 commit 1 unless noted:

- `.gitignore` excluded `.env/` (directory) but not `.env` (file) — **fixed**
- `hemophilia_clinic.db` / `hemophilia.db` tracked in git — **untracked**
- `requirements.txt` unusable — **rewritten**
- Both Dockerfiles `COPY .env .env` — **removed**
- `uvicorn --reload` in the production container — **removed**
- `allow_origins=["*"]` with `allow_credentials=True` — fixed in commit 3
- Unsalted SHA-256 password hashing — replaced with bcrypt in commit 3
- JWT secret defaulting to a literal string — made required in commit 3

**Not fixable by this refactor:** untracking the database does not remove those
password hashes from git history in a public repository. The four accounts must be
treated as compromised regardless of what the working tree now contains. History
rewriting is the repository owner's decision.

Good news from the audit: no real `.env` and no API key was ever tracked, and
`database.py` bound every SQL parameter.

## Constraints this audit establishes

1. **CHAMP is variant-level, not patient-level.** Each row is an F8 variant;
   `History of Inhibitor` is a registry aggregation over reports for that variant.
   The honest claim is *variant-attributable* inhibitor risk, never an individual
   patient's probability.
2. **The label has reporting bias.** 1,741 rows (43%) are excluded as
   "Not reported", almost certainly non-randomly — variants studied in
   inhibitor-focused work are likelier to have the field filled. The 20.1%
   positive rate describes this registry's reporting, not population incidence.
3. **The system is genomic-only.** There is no authorised clinical dataset in this
   repository — only the 30 fabricated rows. Any "genomic + clinical fusion" claim
   in older docstrings is removed.
