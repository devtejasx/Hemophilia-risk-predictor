"""The acceptance test.

    input -> API -> validation -> preprocessing -> model -> prediction
          -> explanation -> database -> API response

One test walks the whole path and asserts at each stage. If this passes, the
system's stated workflow genuinely works on the MMC2 + MMC3 dataset.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import db
from ml.preprocessing import hemophilia_a as ha
from tests.conftest import MODEL_VERSION, requires_model

pytestmark = requires_model

PASSWORD = "correct-horse-battery-staple"

#: A merged-block record, written out rather than generated, so the test reads
#: as an example of a real submission. Every value here occurs in the dataset.
CASE = {
    "mut_type": "Point",
    "mut_effect": "Missense",
    "location": "Exon",
    "e_i_numb": "14",
    "locnumb": "14",
    "n_bp": "1",
    "nuc_numb": "1834",
    "mut_syn": "c.1834C>T",
    "cli_phe": "Severe",
    "aa_numb_old": 593.0,
    "aa_numb": 612.0,
    # optional, supplied here to exercise the full path
    "ntchange": "C>T",
    "aa_first": "Arg",
    "clotting": "<1",
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_db_path", str(tmp_path / "e2e.db"))
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_full_clinical_workflow(client):
    # 1. The service is up, with a database and every prediction mode loaded.
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["model"] == "loaded"
    assert set(health["model_versions"]) == {"genomic", "clinical", "merged"}

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

    # 4. The form vocabulary comes from the model itself, and every required
    #    field of the merged block is one the caller can actually fill.
    schema = client.get("/api/predictions/schema", headers=auth).json()
    assert set(schema["required"]) <= set(CASE)
    for field, value in CASE.items():
        if field in schema["categorical"] and field not in schema["open_vocabulary"]:
            assert value in schema["categorical"][field], f"{field}={value} not offered"

    # 5. Submit the record.
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={"feature_set": "merged", "features": CASE, "mutation_label": "c.1834C>T"},
        headers=auth,
    )
    assert response.status_code == 201, response.text
    prediction = response.json()
    prediction_id = prediction["id"]

    # 6. A calibrated probability, banded by the model's own threshold.
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["model_version"] == MODEL_VERSION
    assert prediction["preprocessing_version"] == "mmc-preprocessing-1"
    assert prediction["feature_set"] == "merged"
    elevated = prediction["probability"] >= prediction["threshold"]
    assert prediction["prediction"] == int(elevated)
    assert prediction["risk"] == ("High" if elevated else "Low")
    assert prediction["risk_category"] == (
        "Elevated estimated risk" if elevated else "Lower estimated risk"
    )

    # 7. Preprocessing genuinely ran, and the stored record round-trips exactly.
    stored = db.get_prediction(prediction_id, user_id=1)
    assert stored is not None
    assert stored["features"] == CASE
    assert stored["mutation_label"] == "c.1834C>T"
    assert stored["probability"] == pytest.approx(prediction["probability"])

    # 8. Explanations: SHAP and LIME, over MMC2/MMC3 columns only.
    explanation = client.get(
        f"/api/predictions/{prediction_id}/explanation", headers=auth
    ).json()
    assert explanation["prediction_id"] == prediction_id
    assert explanation["feature_set"] == "merged"
    assert "clinical record" in explanation["unit_of_explanation"]

    known = set(ha.GENOMIC_CANDIDATES) | set(ha.CLINICAL_CANDIDATES)
    for method in ("shap", "lime"):
        block = explanation[method]
        assert block["available"], f"{method}: {block.get('reason')}"
        assert block["contributions"], f"{method} returned no contributions"
        for item in block["contributions"]:
            assert item["feature"] in known, f"{method} named an unknown feature"
            assert item["label"]
            assert item["supplied"] == (item["feature"] in CASE)
            assert item["direction"] in {"increases", "decreases", "no effect"}

    # 9. Explanations are persisted alongside the prediction.
    assert set(db.get_explanations(prediction_id)) == {"shap", "lime"}

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
    assert analytics["mutation_type_distribution"]["Point"] == 1
    assert analytics["feature_set_distribution"]["merged"] == 1

    # 13. The patient list carries the latest estimate.
    listed = client.get("/api/patients", headers=auth).json()[0]
    assert listed["prediction_count"] == 1
    assert listed["latest_probability"] == pytest.approx(prediction["probability"])
    assert listed["latest_risk"] == prediction["risk"]

    # 14. Invalid input produces a clear error and stores nothing.
    bad = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={"feature_set": "merged", "features": {**CASE, "mut_type": "Not A Type"}},
        headers=auth,
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["field"] == "mut_type"
    assert (
        len(client.get(f"/api/patients/{patient_id}/history", headers=auth).json()) == 1
    )


def test_all_three_prediction_modes_work_end_to_end(client):
    """Genomic, clinical and merged are the notebook's three blocks. Each is a
    separately trained artifact and each must serve its own feature space."""
    token = client.post(
        "/api/auth/register",
        json={"email": "d3@example.org", "full_name": "D3", "password": PASSWORD},
    ).json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    patient_id = client.post(
        "/api/patients",
        json={"identifier": "MRN-M", "display_name": "M"},
        headers=auth,
    ).json()["id"]

    seen = {}
    for feature_set in ("genomic", "clinical", "merged"):
        schema = client.get(
            f"/api/predictions/schema?feature_set={feature_set}", headers=auth
        ).json()
        features = {k: v for k, v in CASE.items() if k in schema["required"]}
        # Anything required by this mode but missing from the shared CASE dict
        # is filled from the model's own vocabulary.
        for column in schema["required"]:
            features.setdefault(
                column,
                schema["categorical"][column][0]
                if column in schema["categorical"]
                else 1.0,
            )

        body = client.post(
            f"/api/patients/{patient_id}/predictions",
            json={"feature_set": feature_set, "features": features},
            headers=auth,
        )
        assert body.status_code == 201, body.text
        result = body.json()
        assert result["feature_set"] == feature_set
        assert result["model_version"] == f"mmc-{feature_set}-v1"
        seen[feature_set] = result

        explanation = client.get(
            f"/api/predictions/{result['id']}/explanation", headers=auth
        ).json()
        for item in explanation["shap"]["contributions"]:
            assert item["feature"] in features or not item["supplied"]

    # A genomic model must never have been shown a clinical field.
    genomic_features = set(
        client.get("/api/predictions/schema?feature_set=genomic", headers=auth)
        .json()["groups"]
    )
    assert not genomic_features & set(ha.CLINICAL_CANDIDATES)

    analytics = client.get("/api/analytics", headers=auth).json()
    assert analytics["feature_set_distribution"] == {
        "genomic": 1,
        "clinical": 1,
        "merged": 1,
    }


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

    payload = {"feature_set": "merged", "features": CASE}
    first = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    ).json()
    second = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    ).json()

    assert first["probability"] == second["probability"]
    assert first["id"] != second["id"]
    assert len(client.get(f"/api/patients/{patient_id}/history", headers=auth).json()) == 2


def test_a_champ_era_database_migrates_without_losing_data(tmp_path):
    """Upgrading an existing installation must not destroy its history."""
    import sqlite3

    path = tmp_path / "champ.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);
        CREATE TABLE patients (id INTEGER PRIMARY KEY, user_id INTEGER);
        CREATE TABLE genomic_profiles (
            id INTEGER PRIMARY KEY, patient_id INTEGER, variant_type TEXT);
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY, patient_id INTEGER,
            genomic_profile_id INTEGER, probability REAL);
        CREATE TABLE explanations (id INTEGER PRIMARY KEY, prediction_id INTEGER);
        INSERT INTO users VALUES (1, 'old@example.org');
        INSERT INTO patients VALUES (1, 1);
        INSERT INTO genomic_profiles VALUES (1, 1, 'Missense');
        INSERT INTO predictions VALUES (1, 1, 1, 0.42);
        """
    )
    conn.commit()

    renamed = db.migrate_legacy_champ_tables(conn)
    conn.commit()
    assert set(renamed) == {"genomic_profiles", "predictions", "explanations"}

    conn.executescript(db.SCHEMA)
    conn.commit()

    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"case_records", "predictions", "legacy_champ_predictions"} <= tables

    # Nothing was destroyed: users, patients and the old rows are all still there.
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 1
    assert (
        conn.execute("SELECT probability FROM legacy_champ_predictions").fetchone()[0]
        == 0.42
    )
    assert (
        conn.execute("SELECT variant_type FROM legacy_champ_genomic_profiles")
        .fetchone()[0]
        == "Missense"
    )
    # The new table is empty and CHAMP-free.
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    assert "genomic_profile_id" not in columns
    assert {"case_record_id", "feature_set", "risk"} <= columns

    # Running it again changes nothing.
    assert db.migrate_legacy_champ_tables(conn) == []
    conn.close()
