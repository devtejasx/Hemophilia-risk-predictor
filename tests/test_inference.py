"""Artifact loading, inference contract, thresholding and input validation."""

from __future__ import annotations

import json
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    """The split is over *mutations*, not clinical records.

    A mutation contributes one row now, so the split rows and the split groups
    are the same 2,515 objects - and one mutation's clinical records can no
    longer straddle train and test, because they were collapsed before the
    split ever happened.
    """
    bundle = artifacts.load_bundle(MODEL_VERSION)
    split = bundle.metadata["split"]
    population = bundle.metadata["dataset"]["mutation_level"]

    assert "mut_id" in split["strategy"]
    assert split["unit"] == "one F8 mutation"
    # Every modelled mutation lands in exactly one split, and the three splits
    # account for the whole modelling population - nothing quietly dropped.
    assert (
        split["train"] + split["validation"] + split["test"]
        == split["unique_mutations"]
        == population["n_final_modelling_population"]
        == 2515
    )
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
    """The spec now carries two column lists, and both have to survive the round
    trip through metadata: ``inputs`` (the raw MMC2/MMC3 fields a caller sends)
    and ``columns`` (the aggregates the estimator consumes)."""
    bundle = artifacts.load_bundle(MODEL_VERSION)
    spec = bundle.feature_spec
    assert spec.name == bundle.feature_set == "merged"

    assert spec.columns == [*spec.categorical, *spec.numeric]
    assert spec.inputs
    # Each measurement fans out into mean/median/min/max/censored_rate, so the
    # model always consumes more columns than a caller supplies fields.
    assert len(spec.columns) > len(spec.inputs)
    # required is a promise about what the *caller* must send, not about the
    # derived features.
    assert set(spec.required) <= set(spec.inputs)

    for reserved in (ha.LABEL_COLUMN, ha.GROUP_COLUMN, ha.TARGET_COLUMN):
        assert reserved not in spec.columns
        assert reserved not in spec.inputs

    # Every model feature is traceable back to a field the caller filled in, so
    # an explanation can name "FVIII clotting activity" rather than
    # "num__clotting_median".
    assert {spec.source_column_for(c) for c in spec.columns} <= set(spec.inputs)
    assert spec.source_column_for("num__clotting_median") == "clotting"
    assert spec.source_column_for("num__severity_prop_severe") == "cli_phe"
    assert spec.source_column_for("cat__cli_phe_mode_Severe") == "cli_phe"

    # Nothing excluded as leakage or as an identifier came back in through
    # either list.
    assert not set(spec.columns) & set(ha.EXCLUDED_COLUMNS)
    assert not set(spec.inputs) & set(ha.EXCLUDED_COLUMNS)


def test_bundle_is_cached_and_not_reloaded_per_call():
    a = artifacts.load_bundle(MODEL_VERSION)
    b = artifacts.load_bundle(MODEL_VERSION)
    assert a is b


def test_unknown_version_raises_with_available_versions_listed():
    with pytest.raises(artifacts.ArtifactError, match="Available"):
        artifacts.load_bundle("does-not-exist")


def test_only_mmc_artifacts_can_be_served():
    """A model that never saw MMC2/MMC3 must not be reachable from the serving
    path at all.

    ``legacy-synthetic-v0`` was fitted on fabricated rows. It used to sit in
    ml/artifacts/ behind a "preserved-not-servable" flag - one metadata edit
    away from being served. It now lives in archive/legacy-artifacts/, outside
    the directory the loader scans, so this asserts on the directory contents
    rather than on the flag: whatever is in there is servable by definition, so
    only the three MMC2/MMC3 versions may be in there.
    """
    root = artifacts.artifacts_dir()
    allowed = set(FEATURE_SET_VERSIONS.values()) | {"training_summary.json"}
    present = {p.name for p in root.iterdir()}
    assert present <= allowed, f"unexpected entries in {root}: {sorted(present - allowed)}"
    assert MODEL_VERSION in present

    assert "legacy-synthetic-v0" not in artifacts.available_versions()
    with pytest.raises(artifacts.ArtifactError, match="Available"):
        artifacts.load_bundle("legacy-synthetic-v0")

    # Archived, not deleted: the fabricated-data provenance record is still
    # readable, it is just no longer loadable.
    archived = REPO_ROOT / "archive" / "legacy-artifacts" / "legacy-synthetic-v0"
    assert archived.is_dir()
    meta = json.loads((archived / "metadata.json").read_text(encoding="utf-8"))
    assert "fabricated" in meta["provenance_warning"]


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
    """Asserted on ``inputs``, the fields a caller may send. ``columns`` would
    pass vacuously now that the clinical block reaches the model as aggregates
    (``clotting_mean``) whose names no longer match the raw candidates."""
    genomic = service_for_feature_set("genomic")
    assert not set(genomic.spec.inputs) & set(ha.CLINICAL_CANDIDATES)
    assert not {
        genomic.spec.source_column_for(c) for c in genomic.spec.columns
    } & set(ha.CLINICAL_CANDIDATES)

    clinical = service_for_feature_set("clinical")
    assert not set(clinical.spec.inputs) & set(ha.GENOMIC_CANDIDATES)
    assert not {
        clinical.spec.source_column_for(c) for c in clinical.spec.columns
    } & set(ha.GENOMIC_CANDIDATES)


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
    """A codon or a variant residue is notation, not a closed category.

    The HGVS strings that used to carry this guarantee (mut_syn, aa_syn) are
    excluded now - they were 95% and 81% unique, so they named the mutation
    instead of describing it. The wide-vocabulary fields that remain keep the
    same property: a newly reported mutation carries a spelling the encoder has
    never seen, and the infrequent bucket absorbs it rather than the request
    being rejected.
    """
    open_fields = service.input_schema()["open_vocabulary"]
    assert open_fields
    categories = ha.fitted_categories(service.bundle.preprocessor, service.spec)
    vocab_column = {ha.aggregate_source_column(c): c for c in service.spec.categorical}

    for field_name in open_fields:
        unseen = "ZZZ-never-observed"
        assert unseen not in categories[vocab_column[field_name]]
        result = service.predict({**valid_payload, field_name: unseen})
        assert 0.0 <= result.probability <= 1.0


def test_case_only_typos_are_normalised_rather_than_rejected(service, valid_payload):
    canonical = service.predict(valid_payload).probability
    typo = {k: (v.lower() if isinstance(v, str) else v) for k, v in valid_payload.items()}
    assert service.predict(typo).probability == canonical


def test_non_numeric_number_is_rejected(service, valid_payload):
    """Unparseable is rejected; censored is not.

    The source files write measurements as free text - "<1" means below the
    assay's detection limit and "1 to 5" is a reported range - so those are
    readings, not typos, and the parser turns them into a value plus a
    censoring flag. "abc" is not a reading at all and must still be refused
    rather than silently imputed.
    """
    schema = service.input_schema()
    numeric = next(f for f in schema["numeric"] if f in schema["required"])

    with pytest.raises(InputValidationError, match="must be a measurement") as excinfo:
        service.predict({**valid_payload, numeric: "abc"})
    assert excinfo.value.field == numeric

    for field_name, description in schema["numeric"].items():
        assert description["accepts_censored"] is True
        for censored in ("<1", ">5", "1 to 5"):
            result = service.predict({**valid_payload, field_name: censored})
            assert 0.0 <= result.probability <= 1.0


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
    report a measurement that was never taken.

    One omitted field now fans out into several model columns - an absent
    ``antigen`` has to leave mean, median, min, max *and* the censoring rate
    missing - so the check follows each optional field to every column derived
    from it. A censored_rate of 0.0 for an assay that was never run would be a
    fabricated measurement dressed up as a real one.
    """
    frame = service.validate(valid_payload)
    omitted = set(service.input_schema()["optional"])
    assert omitted  # the merged set has plenty
    assert not omitted & set(valid_payload)

    derived = [
        c for c in service.spec.columns if service.spec.source_column_for(c) in omitted
    ]
    assert derived
    for column in derived:
        assert frame[column].isna().all(), column

    # The mirror image: what the caller did supply must actually reach the row,
    # otherwise the assertion above would pass on an all-missing frame.
    supplied = [
        c
        for c in service.spec.columns
        if service.spec.source_column_for(c) in valid_payload
    ]
    assert supplied
    for column in supplied:
        assert frame[column].notna().all(), column


def test_snake_case_keys_are_accepted(service, valid_payload):
    snake = {
        k.lower().replace(" ", "_").replace("/", "_"): v
        for k, v in valid_payload.items()
    }
    assert (
        service.predict(snake).probability == service.predict(valid_payload).probability
    )


def test_input_schema_only_offers_categories_the_model_knows(service):
    """The schema is keyed on the raw field a caller sends (``cli_phe``); the
    fitted vocabulary lives on the aggregate the model consumes
    (``cli_phe_mode``). The UI must still only ever offer values the encoder
    actually saw."""
    schema = service.input_schema()
    categories = ha.fitted_categories(service.bundle.preprocessor, service.spec)
    vocab_column = {ha.aggregate_source_column(c): c for c in service.spec.categorical}

    assert schema["categorical"]
    for field_name, allowed in schema["categorical"].items():
        assert allowed
        assert set(allowed) <= set(categories[vocab_column[field_name]])
        assert ha.MISSING_CATEGORY not in allowed

    # A caller supplies raw fields, never the derived model features.
    assert set(schema["required"]) <= set(service.spec.inputs)
    assert set(schema["required"]) | set(schema["optional"]) == set(service.spec.inputs)
    assert not set(schema["required"]) & set(schema["optional"])
    assert not set(schema["categorical"]) & set(schema["numeric"])
    assert set(schema["categorical"]) | set(schema["numeric"]) == set(service.spec.inputs)
    # The aggregates are reported for transparency, not asked for. Genomic
    # columns pass through under their own names, so the two lists overlap; it
    # is the *derived* columns that must never be requested from a caller.
    assert set(schema["model_features"]) == set(service.spec.columns)
    derived = {
        c for c in service.spec.columns if ha.aggregate_source_column(c) != c
    }
    assert derived
    assert not derived & set(service.spec.inputs)


def test_input_schema_labels_every_field_for_a_human(service):
    schema = service.input_schema()
    for field_name in service.spec.inputs:
        assert schema["labels"][field_name]
        # A real label, not the database column name echoed back.
        assert schema["labels"][field_name] != field_name
        assert schema["groups"][field_name] in {"genomic", "clinical"}

    # Columns dropped as leakage (they exist only because inhibitor testing
    # happened) or as near-unique identifiers are not offered to a caller under
    # any name, so no form can ask for them.
    for dropped in (
        "type",
        "utype",
        "assay",
        "pa_race",
        "mut_syn",
        "aa_syn",
        "aa_change",
        "codon_change",
        "aa_numb_old",
    ):
        assert dropped in ha.EXCLUDED_COLUMNS
        assert dropped not in schema["labels"]
        assert dropped not in service.spec.inputs
        assert dropped not in service.spec.columns


def test_service_loads_the_model_once(valid_payload):
    service = PredictionService(MODEL_VERSION)
    model_id = id(service.bundle.model)
    service.predict(valid_payload)
    service.predict(valid_payload)
    assert id(service.bundle.model) == model_id
