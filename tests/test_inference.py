"""Artifact loading, inference contract, thresholding and input validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml import artifacts
from ml.inference import (
    FEATURE_SET_VERSIONS,
    RISK_ELEVATED,
    RISK_LOWER,
    InputValidationError,
    PredictionService,
    service_for_feature_set,
)
from ml.preprocessing import hemophilia_a as ha
from tests.conftest import MODEL_VERSION, requires_all_models, requires_model

pytestmark = requires_model


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def test_the_served_model_is_discoverable():
    assert MODEL_VERSION in artifacts.available_versions()


@requires_all_models
def test_one_model_version_exists_per_feature_set():
    available = set(artifacts.available_versions())
    assert set(FEATURE_SET_VERSIONS) == set(ha.FEATURE_SET_NAMES)
    assert set(FEATURE_SET_VERSIONS.values()) <= available


def test_metadata_declares_the_mmc_dataset():
    bundle = artifacts.load_bundle(MODEL_VERSION)
    assert bundle.dataset_name == "MMC2+MMC3"
    dataset = bundle.metadata["dataset"]
    assert dataset["join"]["key"] == "mut_id"
    assert dataset["labels"]["n_labelled"] == 4966
    assert dataset["labels"]["n_positive"] == 836
    assert dataset["merge"]["n_merged_rows"] == 4962
    assert dataset["group_key"] == "mut_id"
    assert dataset["label_column"] == "Inhibitors"


def test_metadata_records_both_source_files_by_hash():
    files = artifacts.load_bundle(MODEL_VERSION).metadata["dataset"]["files"]
    assert set(files) == {"mmc2", "mmc3"}
    for entry in files.values():
        assert "mmc" in entry["path"].lower()
        assert len(entry["sha256"]) == 64


def test_metadata_records_the_grouped_split():
    split = artifacts.load_bundle(MODEL_VERSION).metadata["split"]
    assert "mut_id" in split["strategy"]
    assert split["train"] + split["validation"] + split["test"] == 4962
    assert (
        split["train_mutations"] + split["validation_mutations"] + split["test_mutations"]
        == split["unique_mutations"]
    )
    assert split["overlaps"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }


def test_metadata_records_calibration_and_threshold():
    bundle = artifacts.load_bundle(MODEL_VERSION)
    assert bundle.metadata["calibration"]["method"] == "isotonic"
    assert "mut_id" in bundle.metadata["calibration"]["cv"]
    assert 0.0 < bundle.threshold < 1.0
    assert bundle.metadata["threshold"]["selected_on"] == "validation split"


def test_metadata_feature_count_matches_the_estimator():
    """The exact mismatch that let an earlier pipeline feed a model zero columns."""
    bundle = artifacts.load_bundle(MODEL_VERSION)
    assert len(bundle.feature_names) == bundle.model.n_features_in_


def test_metadata_carries_a_reconstructible_feature_spec():
    bundle = artifacts.load_bundle(MODEL_VERSION)
    spec = bundle.feature_spec
    assert spec.name == bundle.feature_set == "merged"
    assert len(spec.columns) == 29
    assert ha.LABEL_COLUMN not in spec.columns
    assert ha.GROUP_COLUMN not in spec.columns


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
    assert metrics["generated_by"] == "scripts/train_inhibitor_model.py"
    held_out = metrics["held_out_test"]
    # A real, modest model: better than chance, nowhere near the fabricated
    # near-perfect scores of legacy-synthetic-v0.
    assert 0.5 < held_out["roc_auc"] < 0.95
    assert held_out["pr_auc"] > held_out["positive_rate"]  # beats the base rate
    at = held_out["at_threshold"]
    cm = at["confusion_matrix"]
    assert sum(cm.values()) == metrics["split"]["test"]
    for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1"):
        assert 0.0 <= at[key] <= 1.0


def test_test_metrics_are_not_training_metrics():
    """Validation and test are separate evaluations of separate rows."""
    metrics = json.loads(
        (artifacts.load_bundle(MODEL_VERSION).path / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["validation"] != metrics["held_out_test"]
    assert metrics["split"]["train"] > metrics["split"]["test"] > 0


# --------------------------------------------------------------------------
# Prediction contract
# --------------------------------------------------------------------------


def test_predict_returns_the_documented_contract(service, valid_payload):
    result = service.predict(valid_payload).as_dict()
    assert set(result) == {
        "prediction",
        "risk",
        "probability",
        "risk_category",
        "threshold",
        "model_version",
        "feature_set",
        "preprocessing_version",
        "features_used",
        "provenance_warning",
    }
    assert result["prediction"] in (0, 1)
    assert result["risk"] in ("Low", "High")
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"] == MODEL_VERSION
    assert result["feature_set"] == "merged"
    assert result["provenance_warning"] is None


def test_prediction_and_risk_agree_with_the_threshold(service, valid_payload):
    threshold = service.bundle.threshold
    result = service.predict(valid_payload)
    elevated = result.probability >= threshold
    assert result.prediction == int(elevated)
    assert result.risk == ("High" if elevated else "Low")
    assert result.risk_category == (RISK_ELEVATED if elevated else RISK_LOWER)
    assert result.threshold == threshold


def test_risk_wording_is_probabilistic_not_a_verdict():
    for label in (RISK_LOWER, RISK_ELEVATED):
        assert "estimated risk" in label
        assert "will develop" not in label.lower()


def test_prediction_is_deterministic(service, valid_payload):
    assert (
        service.predict(valid_payload).probability
        == service.predict(valid_payload).probability
    )


def test_severity_moves_the_estimate_in_the_expected_direction(service, valid_payload):
    """Directional sanity: severe disease carries a higher reported inhibitor
    rate than mild disease in this dataset, and the model should reflect that."""
    mild = service.predict({**valid_payload, "cli_phe": "Mild"}).probability
    severe = service.predict({**valid_payload, "cli_phe": "Severe"}).probability
    assert severe > mild


def test_transform_produces_the_exact_model_input_width(service, valid_payload):
    matrix = service.transform(valid_payload)
    assert matrix.shape == (1, service.bundle.model.n_features_in_)
    assert not np.isnan(matrix).any()


def test_no_feature_is_silently_zero_filled(service, valid_payload):
    """Every supplied column must reach the matrix as a non-zero signal."""
    matrix = service.transform(valid_payload)
    assert np.count_nonzero(matrix) >= len(valid_payload)


# --------------------------------------------------------------------------
# Feature-set routing
# --------------------------------------------------------------------------


@requires_all_models
def test_each_feature_set_serves_its_own_model():
    for name, version in FEATURE_SET_VERSIONS.items():
        svc = service_for_feature_set(name)
        assert svc.version == version
        assert svc.feature_set == name


@requires_all_models
def test_the_genomic_model_never_accepts_a_clinical_column():
    genomic = service_for_feature_set("genomic")
    assert not set(genomic.spec.columns) & set(ha.CLINICAL_CANDIDATES)
    clinical = service_for_feature_set("clinical")
    assert not set(clinical.spec.columns) & set(ha.GENOMIC_CANDIDATES)


@requires_all_models
def test_a_clinical_only_payload_is_enough_for_the_clinical_model():
    svc = service_for_feature_set("clinical")
    schema = svc.input_schema()
    payload = {
        column: (
            schema["categorical"][column][0]
            if column in schema["categorical"]
            else 1.0
        )
        for column in schema["required"]
    }
    result = svc.predict(payload)
    assert 0.0 <= result.probability <= 1.0
    assert result.feature_set == "clinical"


def test_unknown_feature_set_is_rejected():
    with pytest.raises(InputValidationError) as excinfo:
        service_for_feature_set("astrology")
    assert excinfo.value.allowed == ["clinical", "genomic", "merged"]


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_missing_required_field_is_rejected(service, valid_payload):
    required = service.input_schema()["required"][0]
    payload = {k: v for k, v in valid_payload.items() if k != required}
    with pytest.raises(InputValidationError) as excinfo:
        service.predict(payload)
    assert excinfo.value.field == required


def test_unknown_value_in_a_closed_vocabulary_is_rejected(service, valid_payload):
    with pytest.raises(InputValidationError) as excinfo:
        service.predict({**valid_payload, "mut_type": "Not A Real Mutation Type"})
    assert excinfo.value.field == "mut_type"
    assert "Point" in excinfo.value.allowed


def test_unseen_value_in_an_open_vocabulary_is_accepted(service, valid_payload):
    """mut_syn is HGVS notation, not a category: a new mutation has a notation
    the model has never seen, and the encoder has an infrequent bucket for it."""
    assert "mut_syn" in service.input_schema()["open_vocabulary"]
    result = service.predict({**valid_payload, "mut_syn": "c.9999999A>T"})
    assert 0.0 <= result.probability <= 1.0


def test_case_only_typos_are_normalised_rather_than_rejected(service, valid_payload):
    canonical = service.predict(valid_payload).probability
    typo = {k: (v.lower() if isinstance(v, str) else v) for k, v in valid_payload.items()}
    assert service.predict(typo).probability == canonical


def test_non_numeric_number_is_rejected(service, valid_payload):
    numeric = service.spec.numeric[0]
    with pytest.raises(InputValidationError, match="must be a number"):
        service.predict({**valid_payload, numeric: "abc"})


def test_non_dict_input_is_rejected(service):
    with pytest.raises(InputValidationError):
        service.predict("not an object")  # type: ignore[arg-type]


def test_optional_fields_may_be_omitted(service, valid_payload):
    optional = service.input_schema()["optional"]
    assert optional  # the merged set has plenty
    result = service.predict(valid_payload)  # valid_payload supplies none of them
    assert 0.0 <= result.probability <= 1.0


def test_omitted_optional_fields_are_marked_missing_not_invented(service, valid_payload):
    """A blank assay must stay blank. Filling it with the training mode would
    report a measurement that was never taken."""
    frame = service.validate(valid_payload)
    for column in service.input_schema()["optional"]:
        assert frame[column].isna().all(), column


def test_snake_case_keys_are_accepted(service, valid_payload):
    snake = {
        k.lower().replace(" ", "_").replace("/", "_"): v
        for k, v in valid_payload.items()
    }
    assert (
        service.predict(snake).probability == service.predict(valid_payload).probability
    )


def test_input_schema_only_offers_categories_the_model_knows(service):
    schema = service.input_schema()
    categories = ha.fitted_categories(service.bundle.preprocessor, service.spec)
    for column, allowed in schema["categorical"].items():
        assert set(allowed) <= set(categories[column])
        assert ha.MISSING_CATEGORY not in allowed
    assert set(schema["required"]) <= set(service.spec.columns)
    assert set(schema["required"]) | set(schema["optional"]) == set(service.spec.columns)


def test_input_schema_labels_every_field_for_a_human(service):
    schema = service.input_schema()
    for column in service.spec.columns:
        assert schema["labels"][column]
        assert schema["groups"][column] in {"genomic", "clinical"}


def test_service_loads_the_model_once(valid_payload):
    service = PredictionService(MODEL_VERSION)
    model_id = id(service.bundle.model)
    service.predict(valid_payload)
    service.predict(valid_payload)
    assert id(service.bundle.model) == model_id
