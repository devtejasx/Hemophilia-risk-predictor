"""The audit trail: which actions are recorded, and what must never land in it.

The trail exists so a reviewer can reconstruct who did what. These tests pin
both halves of that: the events that must appear, and the fact that the rows
carry references rather than content — no password, no token, no feature values.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import db
from tests.conftest import requires_model

pytestmark = requires_model

PASSWORD = "correct-horse-battery-staple"


def _audit_rows(action: str | None = None) -> list[dict]:
    with db.get_connection() as conn:
        sql = "SELECT * FROM audit_logs"
        params: tuple = ()
        if action is not None:
            sql += " WHERE action = ?"
            params = (action,)
        return [dict(r) for r in conn.execute(sql + " ORDER BY id", params)]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_db_path", str(tmp_path / "audit.db"))
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth(client):
    response = client.post(
        "/api/auth/register",
        json={
            "email": "auditor@example.org",
            "full_name": "Test Clinician",
            "password": PASSWORD,
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _case(client: TestClient, auth: dict, feature_set: str = "merged") -> dict:
    schema = client.get(
        f"/api/predictions/schema?feature_set={feature_set}", headers=auth
    ).json()
    features = {
        column: (
            schema["categorical"][column][0]
            if column in schema["categorical"]
            else 1000.0
        )
        for column in schema["required"]
    }
    return {"feature_set": feature_set, "features": features}


def test_registration_is_recorded(client, auth):
    assert len(_audit_rows("user.register")) == 1


def test_login_is_recorded(client, auth):
    client.post(
        "/api/auth/login",
        json={"email": "auditor@example.org", "password": PASSWORD},
    )
    assert len(_audit_rows("user.login")) == 1


def test_a_failed_login_records_nothing(client, auth):
    before = len(_audit_rows())
    response = client.post(
        "/api/auth/login",
        json={"email": "auditor@example.org", "password": "wrong-password-entirely"},
    )
    assert response.status_code == 401
    assert len(_audit_rows()) == before


def test_patient_creation_is_recorded(client, auth):
    response = client.post(
        "/api/patients",
        json={"identifier": "MRN-AUDIT", "display_name": "Audit Case"},
        headers=auth,
    )
    assert response.status_code == 201
    rows = _audit_rows("patient.create")
    assert len(rows) == 1
    assert rows[0]["entity_id"] == response.json()["id"]


def test_prediction_and_explanation_are_recorded(client, auth):
    patient_id = client.post(
        "/api/patients",
        json={"identifier": "MRN-AUDIT-2", "display_name": "Audit Case 2"},
        headers=auth,
    ).json()["id"]

    created = client.post(
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth),
        headers=auth,
    )
    assert created.status_code == 201, created.text
    prediction_id = created.json()["id"]
    assert len(_audit_rows("prediction.create")) == 1

    # First read computes and stores the explanation.
    assert client.get(
        f"/api/predictions/{prediction_id}/explanation", headers=auth
    ).status_code == 200
    assert len(_audit_rows("explanation.compute")) == 1

    # Second read serves the stored copy, and is still a read worth recording.
    assert client.get(
        f"/api/predictions/{prediction_id}/explanation", headers=auth
    ).status_code == 200
    assert len(_audit_rows("explanation.access")) == 1


def test_the_trail_stores_references_not_content(client, auth):
    """Every column is an id, an action name or a timestamp.

    A row must never carry a password, a token or the submitted feature values;
    the schema is what enforces that, so this asserts the schema.
    """
    patient_id = client.post(
        "/api/patients",
        json={"identifier": "MRN-AUDIT-3", "display_name": "Audit Case 3"},
        headers=auth,
    ).json()["id"]
    client.post(
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth),
        headers=auth,
    )

    rows = _audit_rows()
    assert rows, "expected some audit activity"
    assert set(rows[0]) == {"id", "user_id", "action", "entity", "entity_id", "created_at"}

    haystack = " ".join(
        str(value) for row in rows for value in row.values()
    ).lower()
    for secret in (PASSWORD, "bearer", "eyj", "password"):
        assert secret not in haystack, f"audit trail leaked {secret!r}"
