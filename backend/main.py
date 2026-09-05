"""FastAPI application for the Hemophilia A inhibitor-risk research prototype.

Promoted from the previous single-file backend_api.py, which the Phase 0 audit
selected as the canonical backend: it was the only one that imported, the only
one serving the /api/* routes the SPA calls, and the only one with working JWT
auth and per-user row scoping. Its hand-written risk formula is replaced by the
MMC2 + MMC3 models in ml/; its auth moves from unsalted SHA-256 to bcrypt.

Run:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import db
from backend.core import security
from backend.core.config import settings
from backend.routers import analytics, auth, patients, predictions
from backend.schemas import HealthResponse
from backend.services import ml

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
Explainable Hemophilia A inhibitor-risk prediction, built on the **MMC2 +
MMC3** Hemophilia A tables fused on `mut_id`. MMC2 supplies the genomic
description of the mutation, MMC3 the clinical records reported for it and the
target — MMC3's `Inhibitors` field (Yes = 1, No = 0). MMC3's records are
aggregated per mutation, so the unit of prediction is one **F8 mutation**.

Three prediction modes are served, one per feature block:

* `genomic` — the MMC2 mutation description alone
* `clinical` — the aggregated MMC3 clinical record alone
* `merged` — both, fused (the default; it discriminates best)

**This is a research prototype.** An estimate is attributed to a *mutation* as
the source literature reports it, not to an individual patient. It is not
clinically validated and must not be used for standalone diagnosis or treatment
decisions. The API never recommends a course of treatment.
"""


def _bootstrap_admin() -> None:
    """Create the configured admin account if it does not exist.

    Nothing is seeded unless both env vars are set - the previous system shipped
    a committed database containing four accounts that all shared the unsalted
    MD5 of "password123".
    """
    email = settings.bootstrap_admin_email.strip()
    password = settings.bootstrap_admin_password
    if not email or not password:
        return
    if db.get_user_by_email(email) is not None:
        return
    problems = security.password_problems(password)
    if problems:
        logger.error(
            "Refusing to create bootstrap admin: password %s.", ", ".join(problems)
        )
        return
    db.create_user(
        email=email,
        full_name="Administrator",
        password_hash=security.hash_password(password),
        role="admin",
    )
    logger.info("Created bootstrap admin account %s", email)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for warning in settings.validate():
        logger.warning("Configuration: %s", warning)

    db.init_database()
    _bootstrap_admin()
    ml.startup()  # loads the model ONCE, not per request

    if not ml.is_ready():
        logger.error(
            "Starting without a usable model for the '%s' mode. Prediction "
            "endpoints will return 503 and /health will report unhealthy.",
            ml.default_feature_set(),
        )
    yield
    ml.shutdown()


app = FastAPI(
    title="Hemophilia Inhibitor-Risk Research Prototype",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

# Explicit origins. The archived backends used allow_origins=["*"] together
# with allow_credentials=True, which browsers reject outright.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(predictions.router)
app.include_router(analytics.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the detail, return a generic message.

    The archived handlers returned str(exc) to the client, which leaks file
    paths and SQL. They also returned bare dicts rather than Responses.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "detail": "The request could not be completed. The error has been logged.",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness plus real dependency checks."""
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        database = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check: database unavailable: %s", exc)
        database = "unavailable"

    model_ok = ml.is_ready()
    versions = ml.model_versions()
    # A missing non-default mode is reported but does not make the service
    # unhealthy: the default mode is what the UI uses.
    healthy = database == "ok" and model_ok
    return HealthResponse(
        status="healthy" if healthy else "degraded",
        database=database,
        model="loaded" if model_ok else "unavailable",
        model_version=ml.model_version(),
        model_versions=versions,
        detail=(
            ml.load_error()
            if not healthy
            else (ml.load_error() or None)
        ),
    )


@app.get("/", tags=["System"])
def root() -> dict:
    return {
        "name": "Hemophilia Inhibitor-Risk Research Prototype",
        "docs": "/docs",
        "health": "/health",
        "dataset": "MMC2 + MMC3 Hemophilia A, joined on mut_id",
        "target": "Inhibitors (Yes = 1, No = 0)",
        "disclaimer": (
            "Research decision-support prototype. Not clinically validated. "
            "Estimates describe a reported record, not an individual patient."
        ),
    }
