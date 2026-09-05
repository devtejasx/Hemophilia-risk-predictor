"""SHAP and LIME correctness — especially that an explanation names only the
raw MMC2/MMC3 fields this model actually takes as input, never a column that was
retired or excluded as leakage, and never implies the caller entered a value they
left blank."""

from __future__ import annotations

import pytest

from ml.explainability.service import ExplanationService, FeatureContribution
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
    """No post-encoding names like 'cat__mut_type_Point', no derived statistic
    like 'clotting_mean', and nothing outside the raw input fields this model
    was fitted on.

    The model consumes aggregates (``clotting_mean``, ``severity_prop_severe``);
    ``FeatureSpec.source_column_for`` maps each one back to the field the caller
    filled in, so an explanation talks about ``clotting`` and ``cli_phe``.
    """
    for item in explanation[method]["contributions"]:
        assert item["feature"] in service.spec.inputs
        assert not item["feature"].startswith(("cat__", "num__"))
        # Already a raw field, not something aggregate_source_column still has
        # a layer to strip off.
        assert ha.aggregate_source_column(item["feature"]) == item["feature"]


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_contributions_never_name_a_column_outside_the_models_input_set(
    explanation, method
):
    """An explanation must never reach outside the model's own input set.

    Superseded feature spaces, columns that leak the outcome and near-unique
    identifiers are all recorded in ``EXCLUDED_COLUMNS`` with their reason; none
    of them may surface in an explanation, and every field named must still be a
    live MMC2 genomic or MMC3 clinical candidate.
    """
    live = set(ha.GENOMIC_CANDIDATES) | set(ha.CLINICAL_CANDIDATES)
    for item in explanation[method]["contributions"]:
        assert item["feature"] not in ha.EXCLUDED_COLUMNS
        assert item["feature"] in live


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
    """The word and the number must never point opposite ways.

    ``contribution`` is rounded to six decimals for display, so a real effect
    smaller than that shows as 0.000000 while still carrying its true sign —
    only a *visible* sign can be checked from the payload. The exact
    zero -> "no effect" rule is pinned on the contribution object itself, in
    ``test_an_exactly_zero_contribution_is_not_labelled_a_direction``.
    """
    for item in explanation[method]["contributions"]:
        value = item["contribution"]
        assert item["direction"] in {"increases", "decreases", "no effect"}
        if value > 0:
            assert item["direction"] == "increases"
        elif value < 0:
            assert item["direction"] == "decreases"


def test_an_exactly_zero_contribution_is_not_labelled_a_direction():
    """A displayed +0.0000 alongside "decreases" reads as a contradiction, so a
    contribution of exactly zero is reported as "no effect".

    Asserted against ``FeatureContribution`` rather than a served payload
    because the payload rounds the value it shows, and a rounded zero is not
    necessarily an exact one.
    """

    def direction(value: float) -> str:
        return FeatureContribution(
            feature="clotting",
            label="FVIII clotting activity (%)",
            value=None,
            supplied=False,
            contribution=value,
        ).direction

    assert direction(0.0) == "no effect"
    assert direction(-0.0) == "no effect"
    assert direction(1e-12) == "increases"
    assert direction(-1e-12) == "decreases"


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


def test_shap_takes_the_fast_tree_path_whenever_the_estimator_allows_it(explainer):
    """Exact TreeSHAP over a calibrated ensemble's base estimators runs in
    milliseconds; the model-agnostic fallback takes tens of seconds per row. So
    whenever a base estimator exposes an interface ``shap.TreeExplainer`` reads
    directly, the tree path must be taken — and the declared basis must say
    which path was actually taken, so a permutation attribution over the
    calibrated probability is never presented as an attribution of the
    uncalibrated ensemble.

    Asserted against the interface rather than a named estimator class, so
    swapping the selected model does not by itself break the suite.
    """
    calibrated = getattr(explainer.bundle.model, "calibrated_classifiers_", None)
    assert calibrated, "the served model is expected to be a calibrated ensemble"

    bases = [
        ExplanationService._unwrap(getattr(entry, "estimator", None))
        for entry in calibrated
    ]
    # scikit-learn forests expose estimators_; XGBoost exposes get_booster.
    readable = [
        b for b in bases if hasattr(b, "estimators_") or hasattr(b, "get_booster")
    ]
    trees = explainer._base_tree_estimators()
    assert len(trees) == len(readable)

    _, basis = explainer._get_shap()
    assert basis == ("uncalibrated ensemble" if trees else "calibrated probability")


def test_shap_declares_the_basis_of_its_attribution(explanation):
    """The tree path attributes the ensemble underneath the isotonic calibrator
    while the fallback attributes the calibrated probability itself, so the
    payload must name which of the two it is rather than leaving a reader to
    assume."""
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
    assert set(features) <= set(service.spec.inputs)
    assert all(f["label"] for f in result["features"])


def test_global_importance_ranks_a_real_driver_above_a_near_empty_column(explainer):
    """cli_phe (clinical severity, reported for almost every mutation) should
    outrank discrep, which is over 99% null and carries almost nothing."""
    result = explainer.global_importance(top_n=30)
    ranking = [f["feature"] for f in result["features"]]
    assert ranking.index("cli_phe") < ranking.index("discrep")


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
