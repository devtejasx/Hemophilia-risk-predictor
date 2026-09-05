"""API contract, authentication, authorisation and failure handling.

Each test gets an isolated temporary database, so nothing here touches a real
one and the suite is order-independent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import db
from ml.inference import FEATURE_SET_VERSIONS
from tests.conftest import MODEL_VERSION, requires_model

pytestmark = requires_model

PASSWORD = "correct-horse-battery-staple"


def _case(client: TestClient, auth: dict, feature_set: str = "merged", **overrides):
    """A request body built from the served schema, so it cannot go stale.

    Every required field is filled with a value the model was actually fitted
    on; optional fields are left out, which is the normal case for this dataset
    (several clinical columns are more than 99% missing).
    """
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
    features.update(overrides)
    return {"feature_set": feature_set, "features": features}


def _ml():
    from backend.services import ml

    return ml


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
    assert body["model_version"] == MODEL_VERSION
    assert set(body["model_versions"]) == {"genomic", "clinical", "merged"}


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
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth),
        headers=other,
    )
    assert response.status_code == 404


def test_missing_patient_returns_404(client, auth):
    assert client.get("/api/patients/99999", headers=auth).status_code == 404


# --------------------------------------------------------------------------
# Prediction schema and validation
# --------------------------------------------------------------------------


def test_schema_endpoint_describes_the_merged_mode_by_default(client, auth):
    schema = client.get("/api/predictions/schema", headers=auth).json()
    assert schema["model_version"] == MODEL_VERSION
    assert schema["feature_set"] == "merged"
    assert schema["available_feature_sets"] == ["genomic", "clinical", "merged"]
    assert "Point" in schema["categorical"]["mut_type"]
    assert "Unknown" not in schema["categorical"]["mut_type"]
    # Every field is offered with a human label and a form group.
    assert schema["groups"]["mut_type"] == "genomic"
    assert schema["groups"]["cli_phe"] == "clinical"
    assert schema["labels"]["cli_phe"] == "Clinical severity"


@pytest.mark.parametrize("feature_set", ["genomic", "clinical", "merged"])
def test_each_prediction_mode_has_its_own_schema(client, auth, feature_set):
    schema = client.get(
        f"/api/predictions/schema?feature_set={feature_set}", headers=auth
    ).json()
    assert schema["feature_set"] == feature_set
    groups = set(schema["groups"].values())
    if feature_set == "genomic":
        assert groups == {"genomic"}
    elif feature_set == "clinical":
        assert groups == {"clinical"}
    else:
        assert groups == {"genomic", "clinical"}


def test_unknown_category_returns_422_with_allowed_values(client, auth, patient_id):
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth, mut_type="Invented Mutation Type"),
        headers=auth,
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["field"] == "mut_type"
    assert "Point" in detail["allowed_values"]


def test_missing_required_field_returns_422(client, auth, patient_id):
    payload = _case(client, auth)
    payload["features"].pop("cli_phe")
    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    )
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "cli_phe"


def test_unknown_feature_set_returns_422(client, auth, patient_id):
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={"feature_set": "astrology", "features": {"mut_type": "Point"}},
        headers=auth,
    )
    assert response.status_code == 422


def test_empty_feature_object_returns_422(client, auth, patient_id):
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={"feature_set": "merged", "features": {}},
        headers=auth,
    )
    assert response.status_code == 422


def test_non_numeric_number_returns_422(client, auth, patient_id):
    response = client.post(
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth, aa_numb="not a number"),
        headers=auth,
    )
    assert response.status_code == 422


def test_a_censored_measurement_is_accepted_and_stored_verbatim(
    client, auth, patient_id
):
    """'<1' is a real FVIII reading, not a typo, and the API must take it.

    The source tables report assays as bounds and ranges below or above the
    limit of detection, so a caller may write them the same way. The service
    parses the value and its censoring with the same parser training used, so
    the request succeeds — and the stored record keeps the string that was
    actually sent, rather than a silently coerced number that would erase the
    fact that the reading was censored.
    """
    schema = client.get("/api/predictions/schema", headers=auth).json()
    assert schema["numeric"]["clotting"]["accepts_censored"] is True

    payload = _case(client, auth, clotting="<1")
    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    )
    assert response.status_code == 201, response.text

    body = response.json()
    assert body["features"]["clotting"] == "<1"
    assert 0.0 <= body["probability"] <= 1.0

    # And it survives a round trip: the stored row still transforms, so an
    # explanation of it does not 409 on its own censored input.
    explanation = client.get(
        f"/api/predictions/{body['id']}/explanation", headers=auth
    )
    assert explanation.status_code == 200, explanation.text


def test_no_prediction_is_stored_when_input_is_invalid(client, auth, patient_id):
    client.post(
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth, mut_type="Nope"),
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
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
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


def test_interpretation_states_the_estimate_is_record_level(client, auth, patient_id):
    body = client.post(
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
    ).json()
    assert "not a prediction about an individual patient" in body["interpretation"]


# --------------------------------------------------------------------------
# Prediction contract
# --------------------------------------------------------------------------


def test_prediction_response_carries_the_documented_fields(client, auth, patient_id):
    payload = _case(client, auth)
    body = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    ).json()
    assert body["prediction"] in (0, 1)
    assert body["risk"] in ("Low", "High")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == MODEL_VERSION
    assert body["feature_set"] == "merged"
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    # The stored record echoes exactly what was sent, nothing added.
    assert body["features"] == payload["features"]


@pytest.mark.parametrize("feature_set", ["genomic", "clinical", "merged"])
def test_every_prediction_mode_answers_from_its_own_model(
    client, auth, patient_id, feature_set
):
    """Each mode must answer from its own artifact, not from the default one.

    The expected version is read from the inference registry rather than
    written out here, so a retrain that renames an artifact updates the
    expectation in one place — but the mapping mode -> artifact is still
    asserted, which is the guarantee: a genomic request must not be quietly
    served by the fused model.
    """
    body = client.post(
        f"/api/patients/{patient_id}/predictions",
        json=_case(client, auth, feature_set=feature_set),
        headers=auth,
    ).json()
    assert body["feature_set"] == feature_set
    assert body["model_version"] == FEATURE_SET_VERSIONS[feature_set]
    # Distinct modes are distinct artifacts, so no two share a version.
    assert len(set(FEATURE_SET_VERSIONS.values())) == len(FEATURE_SET_VERSIONS)


def test_a_stored_prediction_can_be_read_back_and_explained(client, auth, patient_id):
    created = client.post(
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
    ).json()

    fetched = client.get(f"/api/predictions/{created['id']}", headers=auth).json()
    assert fetched["probability"] == created["probability"]

    explanation = client.get(
        f"/api/predictions/{created['id']}/explanation", headers=auth
    ).json()
    assert explanation["feature_set"] == "merged"
    assert explanation["shap"]["available"], explanation["shap"].get("reason")
    assert explanation["lime"]["available"], explanation["lime"].get("reason")
    for item in explanation["shap"]["contributions"]:
        assert item["label"]
        assert item["supplied"] == (item["feature"] in created["features"])


def test_an_explanation_only_names_fields_the_served_model_accepts(
    client, auth, patient_id
):
    """An explanation may name a field only if the served schema offers it.

    This is the guarantee that a blocklist of retired column names used to
    approximate: an explanation must never attribute the probability to
    something the caller cannot supply — a column from a superseded schema, a
    derived aggregate such as ``clotting_mean``, or an encoded name such as
    ``cat__mut_type_Point``. Taking the accepted set from the served schema
    means the check follows whatever the model is actually fitted on instead of
    a hand-written list that goes stale on the next retrain.
    """
    schema = client.get("/api/predictions/schema", headers=auth).json()
    accepted = set(schema["labels"])
    assert accepted == set(schema["required"]) | set(schema["optional"])
    # The model consumes aggregates of those fields, so naming a model feature
    # would be a real failure rather than a vacuous one.
    assert set(schema["model_features"]) - accepted

    created = client.post(
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
    ).json()
    explanation = client.get(
        f"/api/predictions/{created['id']}/explanation", headers=auth
    ).json()
    for method in ("shap", "lime"):
        contributions = explanation[method]["contributions"]
        assert contributions, f"{method} explained nothing: {explanation[method]}"
        for item in contributions:
            assert item["feature"] in accepted, (
                f"{method} named '{item['feature']}', which is not a field of "
                f"the '{schema['feature_set']}' input schema"
            )


def test_a_user_cannot_explain_another_users_prediction(client, auth, patient_id):
    created = client.post(
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
    ).json()
    other = _register(client, "other@example.org")
    response = client.get(
        f"/api/predictions/{created['id']}/explanation", headers=other
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------


def test_analytics_counts_only_real_stored_predictions(client, auth, patient_id):
    empty = client.get("/api/analytics", headers=auth).json()
    assert empty["total_predictions"] == 0
    assert empty["mean_probability"] is None

    client.post(
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
    )
    after = client.get("/api/analytics", headers=auth).json()
    assert after["total_patients"] == 1
    assert after["total_predictions"] == 1
    assert after["mean_probability"] is not None
    assert sum(after["risk_distribution"].values()) == 1
    assert after["feature_set_distribution"] == {"merged": 1}
    assert sum(after["mutation_type_distribution"].values()) == 1


def test_analytics_is_scoped_per_user(client, auth, patient_id):
    client.post(
        f"/api/patients/{patient_id}/predictions", json=_case(client, auth), headers=auth
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
    ml = _ml()
    payload = _case(client, auth)
    monkeypatch.setitem(ml._errors, "merged", "artifact missing")
    monkeypatch.delitem(ml._predictions, "merged")

    response = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    )
    assert response.status_code == 503
    assert "artifact missing" in response.json()["detail"]
    assert client.get("/health").json()["status"] == "degraded"


def test_one_unavailable_mode_does_not_take_down_the_others(
    client, auth, patient_id, monkeypatch
):
    """Each prediction mode is a separate artifact; losing one must not stop the
    others from serving."""
    ml = _ml()
    payload = _case(client, auth, feature_set="genomic")
    monkeypatch.setitem(ml._errors, "clinical", "artifact missing")
    monkeypatch.delitem(ml._predictions, "clinical")

    ok = client.post(
        f"/api/patients/{patient_id}/predictions", json=payload, headers=auth
    )
    assert ok.status_code == 201

    gone = client.post(
        f"/api/patients/{patient_id}/predictions",
        json={"feature_set": "clinical", "features": {"cli_phe": "Severe"}},
        headers=auth,
    )
    assert gone.status_code == 503


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
