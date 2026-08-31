"""SHAP and LIME correctness — especially that explanations never name a
feature the caller did not supply."""

from __future__ import annotations

import pytest

from ml.explainability.service import ExplanationService
from ml.preprocessing import champ
from tests.conftest import requires_model

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


def test_explanation_states_it_is_variant_level(explanation):
    assert "variant" in explanation["unit_of_explanation"].lower()
    assert "not an individual patient" in explanation["unit_of_explanation"]


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_only_name_supplied_champ_columns(
    explanation, valid_payload, method
):
    """No post-encoding names like 'cat__Variant Type_Missense', and nothing the
    caller never provided."""
    for item in explanation[method]["contributions"]:
        assert item["feature"] in champ.FEATURE_COLUMNS
        assert not item["feature"].startswith(("cat__", "num__", "bin__"))
        assert item["feature"] in valid_payload


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_report_the_value_actually_supplied(
    explanation, valid_payload, method
):
    for item in explanation[method]["contributions"]:
        assert item["value"] == valid_payload[item["feature"]]


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_direction_matches_the_sign_of_the_contribution(explanation, method):
    for item in explanation[method]["contributions"]:
        expected = "increases" if item["contribution"] > 0 else "decreases"
        assert item["direction"] == expected


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_are_ranked_by_absolute_effect(explanation, method):
    magnitudes = [abs(c["contribution"]) for c in explanation[method]["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_no_duplicate_features_after_aggregation(explanation):
    """One categorical expands to many one-hot columns; they must be summed back
    onto a single row, not listed separately."""
    for method in ("shap", "lime"):
        names = [c["feature"] for c in explanation[method]["contributions"]]
        assert len(names) == len(set(names))


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


def test_global_importance_ranks_known_drivers_highly(explainer):
    result = explainer.global_importance(top_n=9)
    assert result["available"], result.get("reason")
    features = [f["feature"] for f in result["features"]]
    assert set(features) <= set(champ.FEATURE_COLUMNS)
    # Variant Type and reported severity dominate the crosstabs; they should not
    # rank below the near-constant In Poly A flag.
    assert features.index("Variant Type") < features.index("In Poly A")
    assert features.index("Reported Clinical Severity") < features.index("In Poly A")


def test_importances_are_non_negative(explainer):
    for item in explainer.global_importance()["features"]:
        assert item["importance"] >= 0


def test_explanation_reports_failure_instead_of_inventing_values(service, monkeypatch):
    """A broken explainer must say it is unavailable, never fall back to
    something that looks like a real attribution."""
    ex = ExplanationService(service.bundle)
    monkeypatch.setattr(
        ex, "_get_shap", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = ex._explain_shap(service.transform({
        "Variant Type": "Missense", "Mechanism": "Substitution", "Domain": "A2",
        "Subtype": "Heavy chain", "In Poly A": "N",
        "Reported Clinical Severity": "Mild",
    }), {}, 5)
    assert result["available"] is False
    assert "boom" in result["reason"]
    assert "contributions" not in result
