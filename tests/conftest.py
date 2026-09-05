"""Shared fixtures. Adds the repo root to sys.path so `ml` imports without install."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.artifacts import available_versions  # noqa: E402
from ml.inference import (  # noqa: E402
    DEFAULT_MODEL_VERSION,
    FEATURE_SET_VERSIONS,
    PredictionService,
)

#: Taken from the inference module rather than written out, so renaming the
#: served artifact cannot silently skip the whole suite.
MODEL_VERSION = DEFAULT_MODEL_VERSION

requires_model = pytest.mark.skipif(
    MODEL_VERSION not in available_versions(),
    reason=f"{MODEL_VERSION} not built; run scripts/train_inhibitor_model.py",
)

requires_all_models = pytest.mark.skipif(
    not set(FEATURE_SET_VERSIONS.values()) <= set(available_versions()),
    reason="not all feature-set models are built; run scripts/train_inhibitor_model.py",
)


@pytest.fixture(scope="session")
def service() -> PredictionService:
    return PredictionService(MODEL_VERSION)


@pytest.fixture(scope="session")
def valid_payload(service: PredictionService) -> dict:
    """A merged-block input using only values the model was fitted on.

    Built from the served model's own vocabulary rather than hand-written, so it
    cannot drift away from what the artifact accepts. Every required column is
    filled; optional ones are deliberately left out to exercise the
    explicit-missing path.
    """
    payload: dict = {}
    schema = service.input_schema()
    for column in schema["required"]:
        if column in schema["categorical"]:
            payload[column] = schema["categorical"][column][0]
        else:
            payload[column] = 1000.0
    return payload
