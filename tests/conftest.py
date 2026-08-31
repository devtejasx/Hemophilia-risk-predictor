"""Shared fixtures. Adds the repo root to sys.path so `ml` imports without install."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.artifacts import available_versions  # noqa: E402
from ml.inference import PredictionService  # noqa: E402

MODEL_VERSION = "champ-v1"

requires_model = pytest.mark.skipif(
    MODEL_VERSION not in available_versions(),
    reason=f"{MODEL_VERSION} not built; run scripts/train_champ.py",
)


@pytest.fixture(scope="session")
def service() -> PredictionService:
    return PredictionService(MODEL_VERSION)


@pytest.fixture(scope="session")
def valid_payload() -> dict:
    """A CHAMP-compatible input using only categories present in the registry."""
    return {
        "Variant Type": "Missense",
        "Mechanism": "Substitution",
        "Domain": "A2",
        "Subtype": "Heavy chain",
        "In Poly A": "N",
        "Reported Clinical Severity": "Severe",
        "exon_number": 14,
        "codon_number": 1200,
        "is_intron": 0,
    }
