"""The single database layer: SQLite connection, schema and queries.

Consolidated from the previous root-level database.py (which owned the richest
schema and bound every SQL parameter) and the table definitions inside
backend_api.py. Three other competing data layers are archived.

Design notes:

* Every table that belongs to a user is scoped by ``user_id`` in the query
  itself, so one account cannot read another's records. That pattern came from
  backend_api.py and is preserved deliberately.
* No fabricated clinical variables are stored. A patient row holds an
  identifier and a display name; every predictive field lives in
  ``case_records`` because those are the only fields the model consumes.
* ``case_records`` stores the submitted features as JSON rather than as fixed
  columns. The three feature sets take 14, 6 and 20 input fields, and the
  exact list is decided at training time by what the MMC2/MMC3 files contain —
  so a fixed-column table would have to be migrated every time a model is
  retrained. The feature set and model version are stored alongside, and
  reading a record back goes through the same ml.inference validation that
  produced it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
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

-- The MMC2/MMC3-shaped record a prediction was made from. `features` is the
-- JSON object that ml.inference validated, keyed by source column name;
-- `feature_set` says which of genomic / clinical / merged it belongs to.
-- `mutation_label` is a display convenience only and never reaches the model.
CREATE TABLE IF NOT EXISTS case_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    feature_set    TEXT NOT NULL,
    features       TEXT NOT NULL,
    mutation_label TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    case_record_id        INTEGER NOT NULL REFERENCES case_records(id) ON DELETE CASCADE,
    probability           REAL NOT NULL,
    prediction            INTEGER NOT NULL,
    risk                  TEXT NOT NULL,
    risk_category         TEXT NOT NULL,
    threshold             REAL NOT NULL,
    model_version         TEXT NOT NULL,
    feature_set           TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_records_patient     ON case_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_explanations_pred   ON explanations(prediction_id);
"""

#: The one feature the analytics page breaks predictions down by. It is the
#: lowest-cardinality genomic column (Point / Deletion / Insertion / …), so a
#: count over it is readable; every other column is either near-unique or absent
#: from most submissions.
ANALYTICS_BREAKDOWN_COLUMN = "mut_type"

#: Tables belonging to the retired pre-MMC2/MMC3 feature space. Their columns
#: (variant_type, mechanism, domain, …) cannot hold an MMC2/MMC3 record, and the
#: rows in them were produced by a superseded model version that is no longer
#: servable. They are renamed rather than dropped: the data is a user's own
#: history and deleting it to make room for a new schema is not this migration's
#: call to make.
LEGACY_SCHEMA_TABLES = {
    "genomic_profiles": "legacy_pre_mmc_genomic_profiles",
    "predictions": "legacy_pre_mmc_predictions",
    "explanations": "legacy_pre_mmc_explanations",
}

#: The file every connection opens. Tests point this at a tmp_path copy.
_db_path: str = settings.database_path


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
        migrate_legacy_schema_tables(conn)
        conn.executescript(SCHEMA)
    logger.info("Database ready at %s", _db_path)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate_legacy_schema_tables(conn: sqlite3.Connection) -> list[str]:
    """Move tables of the retired pre-MMC2/MMC3 schema aside for the new one.

    Detection is by column, not by a version number: a ``genomic_profiles``
    table belongs to the retired pre-MMC2/MMC3 feature space if it has a
    ``variant_type`` column, and a ``predictions`` table does if it references
    ``genomic_profile_id``. Both are renamed with their data intact, together
    with the explanations that point at them, so a
    `SELECT * FROM legacy_pre_mmc_predictions` still returns every estimate the
    superseded model version produced.

    Nothing outside those three tables is touched: users and patients carry over
    unchanged, and their identifiers are unaffected by the rename.

    Returns the tables that were renamed, so the caller can log or assert on it.
    """
    existing = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    legacy = (
        "genomic_profiles" in existing
        and "variant_type" in _table_columns(conn, "genomic_profiles")
    ) or (
        "predictions" in existing
        and "genomic_profile_id" in _table_columns(conn, "predictions")
    )
    if not legacy:
        return []

    renamed: list[str] = []
    for old, new in LEGACY_SCHEMA_TABLES.items():
        if old not in existing or new in existing:
            continue
        conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
        renamed.append(old)

    if renamed:
        logger.warning(
            "Migrated tables of the retired pre-MMC2/MMC3 feature space aside: "
            "%s. Their rows were produced by a superseded model version and are "
            "preserved read-only under their legacy_pre_mmc_* names; new "
            "predictions use case_records.",
            ", ".join(renamed),
        )
    return renamed


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
                       AS latest_risk_category,
                   (SELECT pr.risk FROM predictions pr
                     WHERE pr.patient_id = p.id
                     ORDER BY pr.created_at DESC, pr.id DESC LIMIT 1)
                       AS latest_risk
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
# Case records + predictions
# --------------------------------------------------------------------------


def create_case_record(
    patient_id: int,
    feature_set: str,
    features: dict[str, Any],
    mutation_label: str | None = None,
) -> int:
    """Store the exact feature dict ml.inference validated.

    The JSON is written verbatim, keys and all, so the record can be replayed
    through the same validation later — which is how the explanation endpoint
    rebuilds its input without a second copy of the feature logic.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO case_records
                (patient_id, feature_set, features, mutation_label)
            VALUES (?,?,?,?)
            """,
            (patient_id, feature_set, json.dumps(features), mutation_label),
        )
        return int(cursor.lastrowid)


def create_prediction(patient_id: int, case_record_id: int,
                      result: dict[str, Any]) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (patient_id, case_record_id, probability, prediction, risk,
                 risk_category, threshold, model_version, feature_set,
                 preprocessing_version)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                patient_id,
                case_record_id,
                result["probability"],
                int(result["prediction"]),
                result["risk"],
                result["risk_category"],
                result["threshold"],
                result["model_version"],
                result["feature_set"],
                result["preprocessing_version"],
            ),
        )
        return int(cursor.lastrowid)


def _with_features(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Decode the stored feature JSON into a `features` dict on the row."""
    if row is None:
        return None
    data = dict(row)
    raw = data.pop("features", None)
    data["features"] = json.loads(raw) if raw else {}
    return data


def get_prediction(prediction_id: int, user_id: int) -> dict[str, Any] | None:
    """Joined through patients so another user's prediction is simply not found."""
    with get_connection() as conn:
        return _with_features(
            conn.execute(
                """
                SELECT pr.*, c.features, c.mutation_label,
                       p.display_name AS patient_name
                  FROM predictions pr
                  JOIN patients p      ON p.id = pr.patient_id
                  JOIN case_records c  ON c.id = pr.case_record_id
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
            SELECT pr.*, c.features, c.mutation_label
              FROM predictions pr
              JOIN patients p     ON p.id = pr.patient_id
              JOIN case_records c ON c.id = pr.case_record_id
             WHERE pr.patient_id = ? AND p.user_id = ?
             ORDER BY pr.created_at DESC, pr.id DESC
             LIMIT ?
            """,
            (patient_id, user_id, limit),
        ).fetchall()
    return [_with_features(r) for r in rows]  # type: ignore[misc]


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
            SELECT pr.risk_category, pr.probability, pr.created_at,
                   pr.feature_set, c.features
              FROM predictions pr
              JOIN patients p     ON p.id = pr.patient_id
              JOIN case_records c ON c.id = pr.case_record_id
             WHERE p.user_id = ?
            """,
            (user_id,),
        ).fetchall()

    probabilities = [r["probability"] for r in rows]
    distribution: dict[str, int] = {}
    mutation_types: dict[str, int] = {}
    feature_sets: dict[str, int] = {}
    for r in rows:
        distribution[r["risk_category"]] = distribution.get(r["risk_category"], 0) + 1
        feature_sets[r["feature_set"]] = feature_sets.get(r["feature_set"], 0) + 1
        # mut_type is absent from a clinical-only submission; counting it as
        # "Not supplied" is honest, where defaulting it to a category would not be.
        features = json.loads(r["features"]) if r["features"] else {}
        label = str(features.get(ANALYTICS_BREAKDOWN_COLUMN) or "Not supplied")
        mutation_types[label] = mutation_types.get(label, 0) + 1

    return {
        "total_patients": patients,
        "total_predictions": len(rows),
        "mean_probability": (
            round(sum(probabilities) / len(probabilities), 6) if probabilities else None
        ),
        "risk_distribution": distribution,
        "mutation_type_distribution": mutation_types,
        "feature_set_distribution": feature_sets,
    }


def write_audit_log(user_id: int | None, action: str, entity: str | None = None,
                    entity_id: int | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, action, entity, entity_id) VALUES (?,?,?,?)",
            (user_id, action, entity, entity_id),
        )
