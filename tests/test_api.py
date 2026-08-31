"""API contract, authentication, authorisation and failure handling.

Each test gets an isolated temporary database, so nothing here touches a real
one and the suite is order-independent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import db
from tests.conftest import requires_model

pytestmark = requires_model

PASSWORD = "correct-horse-battery-staple"

VALID_GENOMIC = {
    "Variant Type": "Missense",
    "Mechanism": "Substitution",
    "Domain": "A2",
    "Subtype": "Heavy chain",
    "In Poly A": "N",
    "Reported Clinical Severity": "Severe",
    "exon_number": 14,
    "codon_number": 1200,
    "is_intron": False,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_db_path", str(tmp_path / "test.db"))
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def _register(client: TestClient, email: str = "clinician@example.org") -> dict:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "full_name": "Test Clinician", "password": PASSWORD},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def auth(client):
    return _register(client)


@pytest.fixture()
def patient_id(client, auth):
    response = client.post(
        "/api/patients",
        json={"identifier": "MRN-001", "display_name": "Patient A"},
        headers=auth,
    )
    assert response.status_code == 201
    return response.json()["id"]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


def test_health_reports_database_and_model(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["database"] == "ok"
    assert body["model"] == "loaded"
    assert body["model_version"] == "champ-v1"


def test_root_carries_the_research_disclaimer(client):
    assert "not clinically validated" in client.get("/").json()["disclaimer"].lower()


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def test_register_and_login(client):
    _register(client)
    response = client.post(
        "/api/auth/login",
        json={"email": "clinician@example.org", "password": PASSWORD},
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_weak_password_is_rejected(client):
    response = client.post(
        "/api/auth/register",
        json={"email": "a@b.org", "full_name": "A", "password": "password123"},
    )
    assert response.status_code == 422


def test_duplicate_registration_conflicts(client):
    _register(client)
    response = client.post(
        "/api/auth/register",
        json={"email": "clinician@example.org", "full_name": "X", "password": PASSWORD},
    )
    assert response.status_code == 409


def test_wrong_password_is_rejected_without_revealing_the_account(client):
    _register(client)
    response = client.post(
        "/api/auth/login",
        json={"email": "clinician@example.org", "password": "wrong-password-here"},
    )
    unknown = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.org", "password": "wrong-password-here"},
    )
    assert response.status_code == unknown.status_code == 401
    assert response.json()["detail"] == unknown.json()["detail"]


def test_password_is_not_stored_in_plaintext_and_is_salted(client):
    _register(client, "one@example.org")
    _register(client, "two@example.org")
    first = db.get_user_by_email("one@example.org")["password_hash"]
    second = db.get_user_by_email("two@example.org")["password_hash"]
    assert PASSWORD not in first
    assert first.startswith("$2")
    assert first != second, "identical passwords must not produce identical hashes"


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/patients"),
        ("post", "/api/patients"),
        ("get", "/api/analytics"),
        ("get", "/api/predictions/schema"),
        ("get", "/api/predictions/1"),
    ],
)
def test_protected_endpoints_require_a_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_garbage_token_is_rejected(client):
    response = client.get(
        "/api/patients", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


# --------------------------------------------------------------------------
# Patients + authorisation
# --------------------------------------------------------------------------


def test_create_and_list_patients(client, auth):
    client.post(
        "/api/patients",
        json={"identifier": "MRN-9", "display_name": "Nine"},
        headers=auth,
    )
    listed = client.get("/api/patients", headers=auth).json()
    assert [p["identifier"] for p in listed] == ["MRN-9"]


def test_duplicate_identifier_conflicts(client, auth, patient_id):
    response = client.post(
        "/api/patients",
        json={"identifier": "MRN-001", "display_name": "Duplicate"},
        headers=auth,
    )
    assert response.status_code == 409


def test_a_user_cannot_read_another_users_patient(client, auth, patient_id):
    other = _register(client, "other@example.org")
    assert client.get(f"/api/patients/{patient_id}", headers=other).status_code == 404
    assert client.get("/api/patients", headers=other).json() == []


def test_a_user_cannot_predict_against_another_users_patient(client, auth, patient_id):
    other = _register(client, "other@example.org")
    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=VALID_GENOMIC, headers=other
    )
    assert response.status_code == 404


def test_missing_patient_returns_404(client, auth):
    assert client.get("/api/patients/99999", headers=auth).status_code == 404


# --------------------------------------------------------------------------
# Prediction schema and validation
# --------------------------------------------------------------------------


def test_schema_endpoint_lists_only_trained_categories(client, auth):
    schema = client.get("/api/predictions/schema", headers=auth).json()
    assert schema["model_version"] == "champ-v1"
    assert "Missense" in schema["categorical"]["Variant Type"]
    assert "Unknown" not in schema["categorical"]["Variant Type"]
    # a1/a2/a3 must remain selectable and distinct from A1/A2/A3.
    assert {"a1", "A1"} <= set(schema["categorical"]["Domain"])


def test_unknown_category_returns_422_with_allowed_values(client, auth, patient_id):
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={**VALID_GENOMIC, "Variant Type": "Invented Variant"},
        headers=auth,
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["field"] == "Variant Type"
    assert "Missense" in detail["allowed_values"]


def test_missing_required_field_returns_422(client, auth, patient_id):
    payload = {k: v for k, v in VALID_GENOMIC.items() if k != "Domain"}
    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    )
    assert response.status_code == 422


def test_out_of_range_number_returns_422(client, auth, patient_id):
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={**VALID_GENOMIC, "exon_number": 99999},
        headers=auth,
    )
    assert response.status_code == 422


def test_no_prediction_is_stored_when_input_is_invalid(client, auth, patient_id):
    client.post(
        f"/api/patients/{patient_id}/predictions",
        json={**VALID_GENOMIC, "Variant Type": "Nope"},
        headers=auth,
    )
    assert client.get(f"/api/patients/{patient_id}/history", headers=auth).json() == []


# --------------------------------------------------------------------------
# Medical-claims guardrails
# --------------------------------------------------------------------------


def test_prediction_language_is_probabilistic_and_never_recommends_treatment(
    client, auth, patient_id
):
    body = client.post(
        f"/api/patients/{patient_id}/predictions", json=VALID_GENOMIC, headers=auth
    ).json()

    assert "estimated risk" in body["risk_category"].lower()
    assert "research decision-support prototype" in body["disclaimer"].lower()

    text = f"{body['interpretation']} {body['risk_category']}".lower()
    assert "probability" in body["interpretation"].lower()
    for forbidden in (
        "will develop",
        "diagnos",
        "prescrib",
        "immune tolerance induction",
        "hospitalization",
        "recommend",
        "should switch",
    ):
        assert forbidden not in text, f"clinical claim leaked into output: {forbidden}"


def test_interpretation_states_the_estimate_is_variant_level(client, auth, patient_id):
    body = client.post(
        f"/api/patients/{patient_id}/predictions", json=VALID_GENOMIC, headers=auth
    ).json()
    assert "not a prediction about an individual patient" in body["interpretation"]


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------


def test_analytics_counts_only_real_stored_predictions(client, auth, patient_id):
    empty = client.get("/api/analytics", headers=auth).json()
    assert empty["total_predictions"] == 0
    assert empty["mean_probability"] is None

    client.post(
        f"/api/patients/{patient_id}/predictions", json=VALID_GENOMIC, headers=auth
    )
    after = client.get("/api/analytics", headers=auth).json()
    assert after["total_patients"] == 1
    assert after["total_predictions"] == 1
    assert after["mean_probability"] is not None
    assert sum(after["risk_distribution"].values()) == 1


def test_analytics_is_scoped_per_user(client, auth, patient_id):
    client.post(
        f"/api/patients/{patient_id}/predictions", json=VALID_GENOMIC, headers=auth
    )
    other = _register(client, "other@example.org")
    assert client.get("/api/analytics", headers=other).json()["total_predictions"] == 0


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_model_outage_returns_503_not_a_fabricated_number(
    client, auth, patient_id, monkeypatch
):
    """The archived pipeline silently fell back to a hand-written formula."""
    from backend.services import ml

    monkeypatch.setattr(ml, "_prediction", None)
    monkeypatch.setattr(ml, "_load_error", "artifact missing")

    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=VALID_GENOMIC, headers=auth
    )
    assert response.status_code == 503
    assert "artifact missing" in response.json()["detail"]
    assert client.get("/health").json()["status"] == "degraded"


def test_internal_errors_do_not_leak_details(tmp_path, monkeypatch):
    """An unexpected failure must return a generic 500, not a stack trace.

    raise_server_exceptions=False makes TestClient behave like a real server:
    by default it re-raises instead of letting the handler produce a response.
    """
    monkeypatch.setattr(db, "_db_path", str(tmp_path / "leak.db"))
    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        auth = _register(client)

        def boom(*args, **kwargs):
            raise RuntimeError("secret path C:/private/db.sqlite")

        monkeypatch.setattr(db, "list_patients", boom)
        response = client.get("/api/patients", headers=auth)

    assert response.status_code == 500
    assert "secret path" not in response.text
    assert "RuntimeError" not in response.text
    assert response.json()["error"] == "internal_error"
