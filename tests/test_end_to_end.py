"""The acceptance test.

    input -> API -> validation -> preprocessing -> model -> prediction
          -> explanation -> database -> API response

One test walks the whole path and asserts at each stage. If this passes, the
system's stated workflow genuinely works.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import db
from tests.conftest import requires_model

pytestmark = requires_model

PASSWORD = "correct-horse-battery-staple"

GENOMIC = {
    "Variant Type": "Large structural change (>50 bp)",
    "Mechanism": "Deletion",
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
    monkeypatch.setattr(db, "_db_path", str(tmp_path / "e2e.db"))
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_full_clinical_workflow(client):
    # 1. The service is up, with both a database and a model.
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["model"] == "loaded"

    # 2. Register and sign in.
    token = client.post(
        "/api/auth/register",
        json={"email": "dr@example.org", "full_name": "Dr E2E", "password": PASSWORD},
    ).json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=auth).json()["email"] == "dr@example.org"

    # 3. Create a patient.
    patient = client.post(
        "/api/patients",
        json={"identifier": "MRN-E2E", "display_name": "E2E Patient"},
        headers=auth,
    ).json()
    patient_id = patient["id"]
    assert patient["prediction_count"] == 0

    # 4. The form vocabulary comes from the model itself.
    schema = client.get("/api/predictions/schema", headers=auth).json()
    for field, value in GENOMIC.items():
        if field in schema["categorical"]:
            assert value in schema["categorical"][field], f"{field}={value} not offered"

    # 5. Submit valid CHAMP-compatible genomic input.
    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=GENOMIC, headers=auth
    )
    assert response.status_code == 201, response.text
    prediction = response.json()
    prediction_id = prediction["id"]

    # 6. A calibrated probability, banded by the model's own threshold.
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["model_version"] == "champ-v1"
    assert prediction["preprocessing_version"] == "champ-preprocessing-1"
    expected = (
        "Elevated estimated risk"
        if prediction["probability"] >= prediction["threshold"]
        else "Lower estimated risk"
    )
    assert prediction["risk_category"] == expected

    # 7. Preprocessing genuinely ran: the stored profile round-trips.
    stored = db.get_prediction(prediction_id, user_id=1)
    assert stored is not None
    assert stored["variant_type"] == GENOMIC["Variant Type"]
    assert stored["domain"] == "A2"
    assert stored["probability"] == pytest.approx(prediction["probability"])

    # 8. Explanations: SHAP and LIME, over supplied CHAMP columns only.
    explanation = client.get(
        f"/api/predictions/{prediction_id}/explanation", headers=auth
    ).json()
    assert explanation["prediction_id"] == prediction_id
    assert "variant" in explanation["unit_of_explanation"].lower()

    for method in ("shap", "lime"):
        block = explanation[method]
        assert block["available"], f"{method}: {block.get('reason')}"
        assert block["contributions"], f"{method} returned no contributions"
        for item in block["contributions"]:
            assert item["feature"] in GENOMIC, f"{method} named an unsupplied feature"
            assert item["direction"] in {"increases", "decreases", "no effect"}

    # 9. Explanations are persisted alongside the prediction.
    persisted = db.get_explanations(prediction_id)
    assert set(persisted) == {"shap", "lime"}

    # 10. A second read is served from storage and matches.
    again = client.get(
        f"/api/predictions/{prediction_id}/explanation", headers=auth
    ).json()
    assert again["shap"]["contributions"] == explanation["shap"]["contributions"]

    # 11. History shows the prediction.
    history = client.get(f"/api/patients/{patient_id}/history", headers=auth).json()
    assert [h["id"] for h in history] == [prediction_id]

    # 12. The dashboard reflects it.
    analytics = client.get("/api/analytics", headers=auth).json()
    assert analytics["total_patients"] == 1
    assert analytics["total_predictions"] == 1
    assert analytics["variant_type_distribution"]["Large structural change (>50 bp)"] == 1

    # 13. The patient list carries the latest estimate.
    listed = client.get("/api/patients", headers=auth).json()[0]
    assert listed["prediction_count"] == 1
    assert listed["latest_probability"] == pytest.approx(prediction["probability"])

    # 14. Invalid input produces a clear error and stores nothing.
    bad = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={**GENOMIC, "Domain": "Not A Domain"},
        headers=auth,
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["field"] == "Domain"
    assert (
        len(client.get(f"/api/patients/{patient_id}/history", headers=auth).json()) == 1
    )


def test_repeated_identical_input_is_reproducible(client):
    token = client.post(
        "/api/auth/register",
        json={"email": "d2@example.org", "full_name": "D2", "password": PASSWORD},
    ).json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    patient_id = client.post(
        "/api/patients",
        json={"identifier": "MRN-R", "display_name": "R"},
        headers=auth,
    ).json()["id"]

    first = client.post(
        f"/api/patients/{patient_id}/predictions", json=GENOMIC, headers=auth
    ).json()
    second = client.post(
        f"/api/patients/{patient_id}/predictions", json=GENOMIC, headers=auth
    ).json()

    assert first["probability"] == second["probability"]
    assert first["id"] != second["id"]
    assert len(client.get(f"/api/patients/{patient_id}/history", headers=auth).json()) == 2
