"""The seam between the API and whichever model answers it.

These tests are the reason the protocols in `backend.services.contracts` are
worth having: they fail the moment a served implementation stops satisfying the
interface the routers are written against, which is exactly the breakage a
future model swap would otherwise introduce silently.
"""

from __future__ import annotations

import pytest

from backend.services.contracts import (
    ExplanationProvider,
    PredictionModel,
    PredictionOutcome,
)
from ml.explainability.service import ExplanationService
from ml.inference import PredictionService
from tests.conftest import requires_model

pytestmark = requires_model

#: Every column `predictions` stores. A model whose result cannot fill these
#: cannot be persisted, so the contract names them explicitly.
REQUIRED_RESULT_KEYS = {
    "probability",
    "prediction",
    "risk",
    "risk_category",
    "threshold",
    "model_version",
    "feature_set",
    "preprocessing_version",
}


def test_the_served_model_satisfies_the_prediction_interface(service):
    assert isinstance(service, PredictionModel)


def test_the_explainer_satisfies_the_explanation_interface(service):
    assert isinstance(ExplanationService(service.bundle), ExplanationProvider)


def test_a_prediction_result_satisfies_the_outcome_interface(service, valid_payload):
    result = service.predict(valid_payload)
    assert isinstance(result, PredictionOutcome)


def test_a_result_carries_every_column_a_prediction_row_stores(
    service, valid_payload
):
    payload = service.predict(valid_payload).as_dict()
    missing = REQUIRED_RESULT_KEYS - set(payload)
    assert not missing, f"result cannot fill stored columns: {sorted(missing)}"


def test_the_registry_hands_out_interface_satisfying_objects():
    """What the routers actually receive, not just what the classes declare."""
    from backend.services import ml

    ml.startup()
    try:
        assert ml.available_feature_sets(), "no model loaded to check"
        for feature_set in ml.available_feature_sets():
            assert isinstance(ml.prediction_service(feature_set), PredictionModel)
            assert isinstance(ml.explanation_service(feature_set), ExplanationProvider)
    finally:
        ml.shutdown()


@pytest.mark.parametrize(
    "method", ["version", "feature_set", "input_schema", "transform", "predict"]
)
def test_the_interface_names_only_members_the_implementation_has(method):
    assert hasattr(PredictionService, method)
