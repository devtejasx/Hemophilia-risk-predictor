# Archive

Historical material moved out of the repository root, kept for reference:

- `session-docs/` — ~90 one-shot status/summary/guide documents produced
  during earlier AI-assisted development sessions. They describe features
  and architectures in various stages of aspiration; none of them are
  authoritative. The root `README.md` documents what actually runs.
- `variants/` — superseded iterations of the main app and API
  (`app_backup.py`, `app_optimized.py`, `app_refactored.py`,
  `app_unified.py`, `app_updated.py`, `api*.py`). None of these are
  imported or deployed by anything. The deployed entrypoint is `app.py`
  (see `Procfile`/`render.yaml`); the docker-compose path uses
  `app_frontend.py` + `backend_api.py`, which remain at the root.
