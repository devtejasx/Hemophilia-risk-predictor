"""Artifact loading, inference contract, thresholding and input validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml import artifacts
from ml.inference import (
    RISK_ELEVATED,
    RISK_LOWER,
    InputValidationError,
    PredictionService,
)
from ml.preprocessing import champ
from tests.conftest import MODEL_VERSION, requires_model

pytestmark = requires_model


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def test_champ_model_is_discoverable():
    assert MODEL_VERSION in artifacts.available_versions()


def test_metadata_declares_champ_as_the_dataset():
    bundle = artifacts.load_bundle(MODEL_VERSION)
    assert bundle.is_trained_on_champ
    dataset = bundle.metadata["dataset"]
    assert dataset["name"] == "CHAMP"
    assert dataset["n_labelled"] == 2296
    assert dataset["unit_of_observation"].startswith("F8 variant")


def test_metadata_records_calibration_and_threshold():
    bundle = artifacts.load_bundle(MODEL_VERSION)
    assert bundle.metadata["calibration"]["method"] == "isotonic"
    assert 0.0 < bundle.threshold < 1.0
    assert bundle.metadata["threshold"]["selected_on"] == "validation split"


def test_metadata_feature_count_matches_the_estimator():
    """The exact mismatch that let the old pipeline feed a model 9 zero columns."""
    bundle = artifacts.load_bundle(MODEL_VERSION)
    assert len(bundle.feature_names) == bundle.model.n_features_in_


def test_bundle_is_cached_and_not_reloaded_per_call():
    a = artifacts.load_bundle(MODEL_VERSION)
    b = artifacts.load_bundle(MODEL_VERSION)
    assert a is b


def test_unknown_version_raises_with_available_versions_listed():
    with pytest.raises(artifacts.ArtifactError, match="Available"):
        artifacts.load_bundle("does-not-exist")


def test_legacy_synthetic_artifacts_are_preserved_but_refuse_to_serve():
    assert "legacy-synthetic-v0" in artifacts.available_versions()
    meta = artifacts.read_metadata("legacy-synthetic-v0")
    assert meta["status"] == "preserved-not-servable"
    assert "fabricated" in meta["provenance_warning"]
    with pytest.raises(artifacts.ArtifactError, match="preserved for provenance"):
        artifacts.load_bundle("legacy-synthetic-v0")


def test_metrics_file_records_a_real_evaluation():
    bundle = artifacts.load_bundle(MODEL_VERSION)
    metrics = json.loads((bundle.path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["generated_by"] == "scripts/train_champ.py"
    held_out = metrics["held_out_test"]
    # A real, modest model: better than chance, nowhere near the fabricated
    # near-perfect scores of legacy-synthetic-v0.
    assert 0.5 < held_out["roc_auc"] < 0.95
    assert held_out["average_precision"] > 0.20  # beats the 20% base rate


# --------------------------------------------------------------------------
# Prediction contract
# --------------------------------------------------------------------------


def test_predict_returns_the_documented_contract(service, valid_payload):
    result = service.predict(valid_payload).as_dict()
    assert set(result) == {
        "probability",
        "risk_category",
        "threshold",
        "model_version",
        "preprocessing_version",
        "features_used",
        "provenance_warning",
    }
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"] == MODEL_VERSION
    assert result["provenance_warning"] is None


def test_risk_category_boundary_is_the_recorded_threshold(service, valid_payload):
    threshold = service.bundle.threshold
    result = service.predict(valid_payload)
    expected = RISK_ELEVATED if result.probability >= threshold else RISK_LOWER
    assert result.risk_category == expected
    assert result.threshold == threshold


def test_risk_wording_is_probabilistic_not_a_verdict():
    for label in (RISK_LOWER, RISK_ELEVATED):
        assert "estimated risk" in label
        assert "will develop" not in label.lower()


def test_prediction_is_deterministic(service, valid_payload):
    first = service.predict(valid_payload)
    second = service.predict(valid_payload)
    assert first.probability == second.probability


def test_higher_risk_variant_scores_above_a_benign_one(service, valid_payload):
    """Directional sanity: a large structural deletion should not score below a
    mild missense variant. Both crosstabs and the literature agree on ordering."""
    benign = {**valid_payload, "Variant Type": "Missense",
              "Mechanism": "Substitution", "Reported Clinical Severity": "Mild"}
    severe = {**valid_payload, "Variant Type": "Large structural change (>50 bp)",
              "Mechanism": "Deletion", "Reported Clinical Severity": "Severe"}
    assert service.predict(severe).probability > service.predict(benign).probability


def test_transform_produces_the_exact_model_input_width(service, valid_payload):
    matrix = service.transform(valid_payload)
    assert matrix.shape == (1, service.bundle.model.n_features_in_)
    assert not np.isnan(matrix).any()


def test_no_feature_is_silently_zero_filled(service, valid_payload):
    """Regression guard for audit finding B-04: the old serving path filled 9 of
    20 features with zeros. Every source column must reach the matrix."""
    matrix = service.transform(valid_payload)
    assert np.count_nonzero(matrix) > len(champ.CATEGORICAL_FEATURES)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_missing_required_field_is_rejected(service, valid_payload):
    payload = {k: v for k, v in valid_payload.items() if k != "Domain"}
    with pytest.raises(InputValidationError) as excinfo:
        service.predict(payload)
    assert excinfo.value.field == "Domain"
    assert excinfo.value.allowed


def test_unknown_category_is_rejected_rather_than_bucketed(service, valid_payload):
    with pytest.raises(InputValidationError) as excinfo:
        service.predict({**valid_payload, "Variant Type": "Not A Real Variant"})
    assert excinfo.value.field == "Variant Type"
    assert "Missense" in excinfo.value.allowed


def test_non_numeric_number_is_rejected(service, valid_payload):
    with pytest.raises(InputValidationError, match="must be a number"):
        service.predict({**valid_payload, "exon_number": "abc"})


def test_non_dict_input_is_rejected(service):
    with pytest.raises(InputValidationError):
        service.predict("not an object")  # type: ignore[arg-type]


def test_optional_numeric_fields_may_be_omitted(service, valid_payload):
    payload = {k: v for k, v in valid_payload.items()
               if k not in {"exon_number", "codon_number"}}
    result = service.predict(payload)
    assert 0.0 <= result.probability <= 1.0


def test_snake_case_keys_are_accepted(service, valid_payload):
    snake = {k.lower().replace(" ", "_"): v for k, v in valid_payload.items()}
    assert service.predict(snake).probability == service.predict(valid_payload).probability


def test_case_typos_in_input_are_normalised(service, valid_payload):
    result = service.predict({**valid_payload, "In Poly A": "n"})
    assert 0.0 <= result.probability <= 1.0


def test_input_schema_only_offers_categories_the_model_knows(service):
    schema = service.input_schema()
    categories = champ.fitted_categories(service.bundle.preprocessor)
    for column, allowed in schema["categorical"].items():
        assert set(allowed) <= set(categories[column])
        assert champ.MISSING_CATEGORY not in allowed
    assert set(schema["required"]) == set(champ.CATEGORICAL_FEATURES)


def test_service_loads_the_model_once(valid_payload):
    service = PredictionService(MODEL_VERSION)
    model_id = id(service.bundle.model)
    service.predict(valid_payload)
    service.predict(valid_payload)
    assert id(service.bundle.model) == model_id
