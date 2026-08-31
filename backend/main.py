"""FastAPI application for the Hemophilia A inhibitor-risk research prototype.

Promoted from the previous single-file backend_api.py, which the Phase 0 audit
selected as the canonical backend: it was the only one that imported, the only
one serving the /api/* routes the SPA calls, and the only one with working JWT
auth and per-user row scoping. Its hand-written risk formula is replaced by the
CHAMP model in ml/; its auth moves from unsalted SHA-256 to bcrypt.

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
Explainable Hemophilia A inhibitor-risk prediction, built on the CHAMP
(CDC Hemophilia A Mutation Project) variant registry.

**This is a research prototype.** Estimates are attributed to an F8 *variant*,
not to an individual patient, are not clinically validated, and must not be used
for standalone diagnosis or treatment decisions. The API never recommends a
course of treatment.
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
            "Starting without a usable model. Prediction endpoints will return "
            "503 and /health will report unhealthy."
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
    healthy = database == "ok" and model_ok
    return HealthResponse(
        status="healthy" if healthy else "degraded",
        database=database,
        model="loaded" if model_ok else "unavailable",
        model_version=ml.model_version(),
        detail=None if healthy else (ml.load_error() or "database unavailable"),
    )


@app.get("/", tags=["System"])
def root() -> dict:
    return {
        "name": "Hemophilia Inhibitor-Risk Research Prototype",
        "docs": "/docs",
        "health": "/health",
        "disclaimer": (
            "Research decision-support prototype. Not clinically validated. "
            "Estimates are variant-attributable, not patient-level."
        ),
    }
