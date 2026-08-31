"""The single database layer: SQLite connection, schema and queries.

Consolidated from the previous root-level database.py (which owned the richest
schema and bound every SQL parameter) and the table definitions inside
backend_api.py. Three other competing data layers are archived.

Design notes:

* Every table that belongs to a user is scoped by ``user_id`` in the query
  itself, so one account cannot read another's records. That pattern came from
  backend_api.py and is preserved deliberately.
* No fabricated clinical variables are stored. A patient row holds an
  identifier and a display name; the clinical signal lives in
  ``genomic_profiles`` because those are the only fields the model consumes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.core.config import settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    full_name     TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'clinician',
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    identifier  TEXT NOT NULL,
    display_name TEXT NOT NULL,
    notes       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, identifier)
);

-- The CHAMP-shaped variant record a prediction was made from. Columns mirror
-- ml.preprocessing.champ.FEATURE_COLUMNS exactly.
CREATE TABLE IF NOT EXISTS genomic_profiles (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id                  INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    variant_type                TEXT NOT NULL,
    mechanism                   TEXT NOT NULL,
    domain                      TEXT NOT NULL,
    subtype                     TEXT NOT NULL,
    in_poly_a                   TEXT NOT NULL,
    reported_clinical_severity  TEXT NOT NULL,
    exon_number                 REAL,
    codon_number                REAL,
    is_intron                   INTEGER NOT NULL DEFAULT 0,
    created_at                  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    genomic_profile_id    INTEGER NOT NULL REFERENCES genomic_profiles(id) ON DELETE CASCADE,
    probability           REAL NOT NULL,
    risk_category         TEXT NOT NULL,
    threshold             REAL NOT NULL,
    model_version         TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS explanations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    method        TEXT NOT NULL,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (prediction_id, method)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action     TEXT NOT NULL,
    entity     TEXT,
    entity_id  INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_patients_user       ON patients(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_patient    ON genomic_profiles(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_explanations_pred   ON explanations(prediction_id);
"""

_db_path: str = settings.database_path


def set_database_path(path: str | Path) -> None:
    """Point the layer at a different file. Used by tests for isolation."""
    global _db_path
    _db_path = str(path)


def database_path() -> str:
    return _db_path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """A connection with row access by name, foreign keys on, committed on success."""
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database ready at %s", _db_path)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------


def create_user(email: str, full_name: str, password_hash: str,
                role: str = "clinician") -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, full_name, password_hash, role) VALUES (?,?,?,?)",
            (email.lower().strip(), full_name.strip(), password_hash, role),
        )
        user_id = cursor.lastrowid
    return get_user_by_id(user_id)  # type: ignore[return-value]


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _row(
            conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        )


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _row(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def touch_last_login(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?", (user_id,)
        )


# --------------------------------------------------------------------------
# Patients  (every query is scoped by user_id)
# --------------------------------------------------------------------------


def create_patient(user_id: int, identifier: str, display_name: str,
                   notes: str | None = None) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO patients (user_id, identifier, display_name, notes) "
            "VALUES (?,?,?,?)",
            (user_id, identifier.strip(), display_name.strip(), notes),
        )
        patient_id = cursor.lastrowid
    return get_patient(patient_id, user_id)  # type: ignore[return-value]


def list_patients(user_id: int, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.*,
                   (SELECT COUNT(*) FROM predictions pr WHERE pr.patient_id = p.id)
                       AS prediction_count,
                   (SELECT pr.probability FROM predictions pr
                     WHERE pr.patient_id = p.id
                     ORDER BY pr.created_at DESC, pr.id DESC LIMIT 1)
                       AS latest_probability,
                   (SELECT pr.risk_category FROM predictions pr
                     WHERE pr.patient_id = p.id
                     ORDER BY pr.created_at DESC, pr.id DESC LIMIT 1)
                       AS latest_risk_category
              FROM patients p
             WHERE p.user_id = ?
             ORDER BY p.created_at DESC
             LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_patient(patient_id: int, user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _row(
            conn.execute(
                "SELECT * FROM patients WHERE id = ? AND user_id = ?",
                (patient_id, user_id),
            ).fetchone()
        )


def delete_patient(patient_id: int, user_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM patients WHERE id = ? AND user_id = ?", (patient_id, user_id)
        )
        return cursor.rowcount > 0


# --------------------------------------------------------------------------
# Genomic profiles + predictions
# --------------------------------------------------------------------------


def create_genomic_profile(patient_id: int, features: dict[str, Any]) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO genomic_profiles
                (patient_id, variant_type, mechanism, domain, subtype, in_poly_a,
                 reported_clinical_severity, exon_number, codon_number, is_intron)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                patient_id,
                features["Variant Type"],
                features["Mechanism"],
                features["Domain"],
                features["Subtype"],
                features["In Poly A"],
                features["Reported Clinical Severity"],
                features.get("exon_number"),
                features.get("codon_number"),
                int(bool(features.get("is_intron", 0))),
            ),
        )
        return int(cursor.lastrowid)


def create_prediction(patient_id: int, genomic_profile_id: int,
                      result: dict[str, Any]) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (patient_id, genomic_profile_id, probability, risk_category,
                 threshold, model_version, preprocessing_version)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                patient_id,
                genomic_profile_id,
                result["probability"],
                result["risk_category"],
                result["threshold"],
                result["model_version"],
                result["preprocessing_version"],
            ),
        )
        return int(cursor.lastrowid)


def get_prediction(prediction_id: int, user_id: int) -> dict[str, Any] | None:
    """Joined through patients so another user's prediction is simply not found."""
    with get_connection() as conn:
        return _row(
            conn.execute(
                """
                SELECT pr.*, g.variant_type, g.mechanism, g.domain, g.subtype,
                       g.in_poly_a, g.reported_clinical_severity,
                       g.exon_number, g.codon_number, g.is_intron,
                       p.display_name AS patient_name
                  FROM predictions pr
                  JOIN patients p          ON p.id = pr.patient_id
                  JOIN genomic_profiles g  ON g.id = pr.genomic_profile_id
                 WHERE pr.id = ? AND p.user_id = ?
                """,
                (prediction_id, user_id),
            ).fetchone()
        )


def list_predictions_for_patient(patient_id: int, user_id: int,
                                 limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT pr.*, g.variant_type, g.reported_clinical_severity
              FROM predictions pr
              JOIN patients p         ON p.id = pr.patient_id
              JOIN genomic_profiles g ON g.id = pr.genomic_profile_id
             WHERE pr.patient_id = ? AND p.user_id = ?
             ORDER BY pr.created_at DESC, pr.id DESC
             LIMIT ?
            """,
            (patient_id, user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------


def save_explanation(prediction_id: int, method: str, payload: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO explanations (prediction_id, method, payload) "
            "VALUES (?,?,?)",
            (prediction_id, method, json.dumps(payload)),
        )


def get_explanations(prediction_id: int) -> dict[str, Any]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT method, payload FROM explanations WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchall()
    return {r["method"]: json.loads(r["payload"]) for r in rows}


# --------------------------------------------------------------------------
# Analytics + audit
# --------------------------------------------------------------------------


def analytics_for_user(user_id: int) -> dict[str, Any]:
    """Counts computed from real stored predictions. Nothing is synthesised."""
    with get_connection() as conn:
        patients = conn.execute(
            "SELECT COUNT(*) AS n FROM patients WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]

        rows = conn.execute(
            """
            SELECT pr.risk_category, pr.probability, pr.created_at, g.variant_type
              FROM predictions pr
              JOIN patients p         ON p.id = pr.patient_id
              JOIN genomic_profiles g ON g.id = pr.genomic_profile_id
             WHERE p.user_id = ?
            """,
            (user_id,),
        ).fetchall()

    probabilities = [r["probability"] for r in rows]
    distribution: dict[str, int] = {}
    variants: dict[str, int] = {}
    for r in rows:
        distribution[r["risk_category"]] = distribution.get(r["risk_category"], 0) + 1
        variants[r["variant_type"]] = variants.get(r["variant_type"], 0) + 1

    return {
        "total_patients": patients,
        "total_predictions": len(rows),
        "mean_probability": (
            round(sum(probabilities) / len(probabilities), 6) if probabilities else None
        ),
        "risk_distribution": distribution,
        "variant_type_distribution": variants,
    }


def write_audit_log(user_id: int | None, action: str, entity: str | None = None,
                    entity_id: int | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, action, entity, entity_id) VALUES (?,?,?,?)",
            (user_id, action, entity, entity_id),
        )
