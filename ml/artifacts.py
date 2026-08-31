"""Versioned model artifact loading.

One artifact directory holds everything needed to reproduce a prediction:
the estimator, the fitted preprocessor, and metadata describing exactly what
they were fitted on. Nothing about a model is inferred from a filename.

Bundles are cached per (version, directory), so a long-running API process
loads each model once at startup rather than re-reading pickles on every
request — the behaviour of the previous ``load_models()``-inside-``predict()``
implementation.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_FILE = "model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"
METADATA_FILE = "metadata.json"

#: Artifacts that must never be served by default. Loading one requires asking
#: for it by name, and callers are expected to surface the warning.
NON_DEFAULT_VERSIONS = {"legacy-synthetic-v0"}


class ArtifactError(RuntimeError):
    """An artifact directory is missing, incomplete, or internally inconsistent."""


@dataclass(frozen=True)
class ArtifactBundle:
    """A loaded model version."""

    version: str
    model: Any
    preprocessor: Any
    metadata: dict[str, Any]
    path: Path

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata.get("features", {}).get("names", []))

    @property
    def threshold(self) -> float:
        value = self.metadata.get("threshold", {}).get("value")
        if value is None:
            raise ArtifactError(
                f"{self.version} metadata does not record a decision threshold"
            )
        return float(value)

    @property
    def is_trained_on_champ(self) -> bool:
        return self.metadata.get("dataset", {}).get("name") == "CHAMP"

    @property
    def provenance_warning(self) -> str | None:
        """Text the API must attach to any prediction from this bundle."""
        return self.metadata.get("provenance_warning")


_cache: dict[tuple[str, str], ArtifactBundle] = {}
_lock = threading.Lock()


def artifacts_dir(base_dir: str | Path | None = None) -> Path:
    return Path(base_dir) if base_dir else DEFAULT_ARTIFACTS_DIR


def available_versions(base_dir: str | Path | None = None) -> list[str]:
    """Version names that have a metadata file, newest-looking last."""
    root = artifacts_dir(base_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / METADATA_FILE).is_file())


def read_metadata(version: str, base_dir: str | Path | None = None) -> dict[str, Any]:
    path = artifacts_dir(base_dir) / version / METADATA_FILE
    if not path.is_file():
        raise ArtifactError(f"No metadata for model version '{version}' at {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle(version: str, base_dir: str | Path | None = None) -> ArtifactBundle:
    """Load a model version, caching the result.

    Raises ArtifactError with an actionable message rather than returning a
    partially-loaded bundle — a caller must never be able to mistake a load
    failure for a prediction.
    """
    root = artifacts_dir(base_dir)
    key = (str(root.resolve()), version)

    cached = _cache.get(key)
    if cached is not None:
        return cached

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached

        path = root / version
        if not path.is_dir():
            known = available_versions(base_dir) or ["<none>"]
            raise ArtifactError(
                f"Model version '{version}' not found in {root}. Available: {', '.join(known)}"
            )

        metadata = read_metadata(version, base_dir)

        if metadata.get("status") == "preserved-not-servable":
            raise ArtifactError(
                f"Model version '{version}' is preserved for provenance and cannot "
                f"be served. {metadata.get('status_reason', '')}".strip()
            )

        model_path = path / MODEL_FILE
        pre_path = path / PREPROCESSOR_FILE
        for required in (model_path, pre_path):
            if not required.is_file():
                raise ArtifactError(f"{version} is incomplete: missing {required.name}")

        model = joblib.load(model_path)
        preprocessor = joblib.load(pre_path)

        bundle = ArtifactBundle(
            version=version,
            model=model,
            preprocessor=preprocessor,
            metadata=metadata,
            path=path,
        )
        _verify_consistency(bundle)

        _cache[key] = bundle
        logger.info(
            "Loaded model version %s (%s features, threshold %.4f)",
            version,
            len(bundle.feature_names),
            bundle.threshold,
        )
        if bundle.provenance_warning:
            logger.warning("%s: %s", version, bundle.provenance_warning)
        return bundle


def _verify_consistency(bundle: ArtifactBundle) -> None:
    """Fail loudly if metadata and estimator disagree about the feature space.

    This is the specific check that would have caught the previous pipeline's
    train/inference mismatch, where the serving code sent 20 columns of which
    9 were zeros the model had never seen.
    """
    declared = bundle.feature_names
    if not declared:
        raise ArtifactError(f"{bundle.version} metadata declares no feature names")

    actual = getattr(bundle.model, "n_features_in_", None)
    if actual is not None and actual != len(declared):
        raise ArtifactError(
            f"{bundle.version} is inconsistent: metadata declares {len(declared)} "
            f"features but the estimator expects {actual}"
        )


def clear_cache() -> None:
    """Drop cached bundles. Used by tests; not called by the application."""
    with _lock:
        _cache.clear()
