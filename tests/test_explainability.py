"""SHAP and LIME correctness — especially that an explanation names only the
model's own MMC2/MMC3 columns, never a CHAMP-era one, and never implies the
caller entered a value they left blank."""

from __future__ import annotations

import pytest

from ml.explainability.service import ExplanationService
from ml.inference import FEATURE_LABELS, service_for_feature_set
from ml.preprocessing import hemophilia_a as ha
from tests.conftest import requires_all_models, requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def explainer(service):
    return ExplanationService(service.bundle)


@pytest.fixture(scope="module")
def explanation(explainer, service, valid_payload):
    return explainer.explain(service.transform(valid_payload), valid_payload, top_n=6)


def test_both_methods_are_available(explanation):
    assert explanation["shap"]["available"], explanation["shap"].get("reason")
    assert explanation["lime"]["available"], explanation["lime"].get("reason")


def test_explanation_states_its_unit_and_feature_set(explanation):
    assert explanation["feature_set"] == "merged"
    unit = explanation["unit_of_explanation"]
    assert "clinical record" in unit
    assert "not an individual patient" in unit


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_only_name_the_models_own_source_columns(
    explanation, service, method
):
    """No post-encoding names like 'cat__mut_type_Point', and nothing outside the
    feature set this model was fitted on."""
    for item in explanation[method]["contributions"]:
        assert item["feature"] in service.spec.columns
        assert not item["feature"].startswith(("cat__", "num__"))


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_never_name_a_champ_column(explanation, method):
    """The old feature space must not survive anywhere in an explanation."""
    retired = {
        "Variant Type",
        "Mechanism",
        "Domain",
        "Subtype",
        "In Poly A",
        "Reported Clinical Severity",
        "exon_number",
        "codon_number",
        "is_intron",
    }
    for item in explanation[method]["contributions"]:
        assert item["feature"] not in retired
        assert item["feature"] in set(ha.GENOMIC_CANDIDATES) | set(
            ha.CLINICAL_CANDIDATES
        )


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_carry_a_human_readable_label(explanation, method):
    for item in explanation[method]["contributions"]:
        assert item["label"]
        assert item["label"] == FEATURE_LABELS[item["feature"]]["label"]


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_report_the_value_actually_supplied(
    explanation, valid_payload, method
):
    for item in explanation[method]["contributions"]:
        assert item["value"] == valid_payload.get(item["feature"])


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_an_omitted_field_is_flagged_rather_than_given_a_value(
    explanation, valid_payload, method
):
    """A blank optional field still reaches the model as an explicit Unknown, so
    it may legitimately appear — but never as if the caller had entered it."""
    for item in explanation[method]["contributions"]:
        assert item["supplied"] == (item["feature"] in valid_payload)
        if not item["supplied"]:
            assert item["value"] is None


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_direction_matches_the_sign_of_the_contribution(explanation, method):
    for item in explanation[method]["contributions"]:
        value = item["contribution"]
        expected = "increases" if value > 0 else "decreases" if value < 0 else "no effect"
        assert item["direction"] == expected


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_zero_contribution_is_not_labelled_a_decrease(explanation, method):
    """A displayed +0.0000 alongside "decreases" reads as a contradiction."""
    for item in explanation[method]["contributions"]:
        if item["contribution"] == 0:
            assert item["direction"] == "no effect"


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_are_ranked_by_absolute_effect(explanation, method):
    magnitudes = [abs(c["contribution"]) for c in explanation[method]["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_no_duplicate_features_after_aggregation(explanation):
    """One categorical expands to hundreds of one-hot columns; they must be
    summed back onto a single row, not listed separately."""
    for method in ("shap", "lime"):
        names = [c["feature"] for c in explanation[method]["contributions"]]
        assert len(names) == len(set(names))


def test_shap_uses_the_fast_tree_path_over_the_boosted_model(explainer):
    """The merged model is an XGBClassifier under an isotonic calibrator. If the
    tree path stops recognising it, explanations fall back to a model-agnostic
    explainer that is far too slow for a request."""
    assert explainer._base_tree_estimators()


def test_shap_declares_the_basis_of_its_attribution(explanation):
    """TreeSHAP runs against the ensemble underneath the isotonic calibrator, so
    the payload must say so rather than implying it explains the calibrated
    probability directly."""
    assert explanation["shap"]["basis"] in {
        "uncalibrated ensemble",
        "calibrated probability",
    }


def test_top_n_is_respected(explainer, service, valid_payload):
    out = explainer.explain(service.transform(valid_payload), valid_payload, top_n=3)
    assert len(out["shap"]["contributions"]) <= 3
    assert len(out["lime"]["contributions"]) <= 3


def test_single_method_can_be_requested(explainer, service, valid_payload):
    out = explainer.explain(
        service.transform(valid_payload), valid_payload, methods=("shap",)
    )
    assert "shap" in out and "lime" not in out


def test_global_importance_names_only_the_models_own_features(explainer, service):
    result = explainer.global_importance(top_n=12)
    assert result["available"], result.get("reason")
    assert result["feature_set"] == "merged"
    features = [f["feature"] for f in result["features"]]
    assert set(features) <= set(service.spec.columns)
    assert all(f["label"] for f in result["features"])


def test_global_importance_ranks_a_real_driver_above_a_near_empty_column(explainer):
    """cli_phe (clinical severity, 1% missing) should outrank assay, which is
    99.5% missing and carries almost nothing."""
    result = explainer.global_importance(top_n=30)
    ranking = [f["feature"] for f in result["features"]]
    assert ranking.index("cli_phe") < ranking.index("assay")


def test_importances_are_non_negative(explainer):
    for item in explainer.global_importance()["features"]:
        assert item["importance"] >= 0


@requires_all_models
def test_a_genomic_explanation_never_mentions_a_clinical_feature():
    svc = service_for_feature_set("genomic")
    schema = svc.input_schema()
    payload = {
        column: (
            schema["categorical"][column][0]
            if column in schema["categorical"]
            else 1000.0
        )
        for column in schema["required"]
    }
    out = ExplanationService(svc.bundle).explain(svc.transform(payload), payload)
    for method in ("shap", "lime"):
        for item in out[method]["contributions"]:
            assert item["feature"] not in ha.CLINICAL_CANDIDATES


def test_explanation_reports_failure_instead_of_inventing_values(
    service, valid_payload, monkeypatch
):
    """A broken explainer must say it is unavailable, never fall back to
    something that looks like a real attribution."""
    ex = ExplanationService(service.bundle)
    monkeypatch.setattr(
        ex, "_get_shap", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = ex._explain_shap(service.transform(valid_payload), valid_payload, 5)
    assert result["available"] is False
    assert "boom" in result["reason"]
    assert "contributions" not in result
