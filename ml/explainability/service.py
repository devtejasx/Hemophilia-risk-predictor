"""The canonical explanation service: SHAP and LIME over the CHAMP model.

Two rules govern everything here:

1. **Never name a feature the caller did not supply.** Explanations are reported
   against the original CHAMP columns (``Variant Type``, ``Domain``, ...), not
   against post-encoding names like ``cat__Variant Type_Missense`` which mean
   nothing to a reader and imply inputs that were never given.
2. **Never invent a contribution.** If SHAP or LIME is unavailable or fails, the
   response says so. It does not fall back to feature importances dressed up as
   a local explanation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ml.artifacts import ArtifactBundle
from ml.preprocessing import champ

logger = logging.getLogger(__name__)

#: Explanations are expensive relative to a tree prediction, so the number of
#: features returned is capped rather than dumping all 47 encoded columns.
DEFAULT_TOP_N = 8


@dataclass
class FeatureContribution:
    feature: str
    """The original CHAMP column, e.g. 'Variant Type'."""
    value: Any
    """The value the caller actually supplied for it."""
    contribution: float
    """Signed effect on the predicted probability. Positive = raises risk."""
    direction: str = field(init=False)

    def __post_init__(self) -> None:
        self.direction = "increases" if self.contribution > 0 else "decreases"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "contribution": round(float(self.contribution), 6),
            "direction": self.direction,
        }


class ExplanationService:
    """SHAP (global + local) and LIME (local) for one loaded model version."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self.encoded_names = champ.encoded_feature_names(bundle.preprocessor)
        self._background = self._load_background()
        self._shap_explainer: Any = None
        self._lime_explainer: Any = None

    # -- setup ------------------------------------------------------------

    def _load_background(self) -> np.ndarray | None:
        path: Path = self.bundle.path / "background.npy"
        if not path.is_file():
            logger.warning(
                "No background sample for %s; LIME will be unavailable",
                self.bundle.version,
            )
            return None
        return np.load(path)

    def _base_tree_estimators(self) -> list[Any]:
        """The tree models underneath a CalibratedClassifierCV, if any.

        Calibration fits one clone of the base estimator per CV fold. Exact
        TreeSHAP over those clones runs in milliseconds, where a model-agnostic
        permutation explainer over the calibrated wrapper takes ~18s per row —
        far too slow for an API request.

        The attribution describes the underlying ensemble rather than the
        isotonic output. Isotonic calibration is a monotone map, so it rescales
        probabilities without reordering feature effects; the ranking and sign
        of contributions are unchanged. Reported in the payload as
        ``basis: "uncalibrated ensemble"`` so this is never implied to be
        an attribution of the calibrated probability itself.
        """
        model = self.bundle.model
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if not calibrated:
            return []
        estimators = []
        for entry in calibrated:
            base = getattr(entry, "estimator", None)
            if base is not None and hasattr(base, "estimators_"):
                estimators.append(base)
        return estimators

    def _get_shap(self):
        """(explainer_or_list, basis) — TreeSHAP when possible, else permutation."""
        if self._shap_explainer is None:
            import shap

            if self._background is None:
                raise RuntimeError("SHAP requires a background sample")

            trees = self._base_tree_estimators()
            if trees:
                self._shap_explainer = (
                    [shap.TreeExplainer(t) for t in trees],
                    "uncalibrated ensemble",
                )
            else:
                self._shap_explainer = (
                    shap.Explainer(
                        lambda data: self.bundle.model.predict_proba(data)[:, 1],
                        shap.maskers.Independent(self._background, max_samples=64),
                    ),
                    "calibrated probability",
                )
        return self._shap_explainer

    @staticmethod
    def _positive_class_values(raw: Any) -> np.ndarray:
        """Normalise SHAP output to a (n_samples, n_features) positive-class array."""
        values = np.asarray(raw.values if hasattr(raw, "values") else raw)
        if values.ndim == 3:  # (samples, features, classes)
            values = values[:, :, -1]
        return values

    def _get_lime(self):
        if self._lime_explainer is None:
            import lime.lime_tabular

            if self._background is None:
                raise RuntimeError("LIME requires a background sample")
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self._background,
                feature_names=self.encoded_names,
                class_names=["No inhibitor reported", "Inhibitor reported"],
                mode="classification",
                random_state=42,
            )
        return self._lime_explainer

    # -- name mapping -----------------------------------------------------

    def _aggregate_to_source_columns(
        self, weights: np.ndarray, supplied: dict[str, Any]
    ) -> list[FeatureContribution]:
        """Sum encoded-column effects back onto the CHAMP column they came from.

        A single categorical input expands to many one-hot columns; reporting
        them separately would overstate the number of factors and name
        categories the caller did not choose.
        """
        totals: dict[str, float] = defaultdict(float)
        for name, weight in zip(self.encoded_names, weights):
            totals[champ.source_column_for(name)] += float(weight)

        contributions = [
            FeatureContribution(
                feature=col,
                value=supplied.get(col),
                contribution=value,
            )
            for col, value in totals.items()
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return contributions

    # -- public API -------------------------------------------------------

    def explain(
        self,
        matrix: np.ndarray,
        supplied: dict[str, Any],
        top_n: int = DEFAULT_TOP_N,
        methods: tuple[str, ...] = ("shap", "lime"),
    ) -> dict[str, Any]:
        """Local explanation for one already-transformed input row.

        ``matrix`` must come from ``PredictionService.transform`` so the columns
        are guaranteed to match what the model was fitted on.
        """
        result: dict[str, Any] = {
            "model_version": self.bundle.version,
            "unit_of_explanation": (
                "F8 variant. Contributions describe the model's use of this "
                "variant's registry features, not an individual patient."
            ),
        }

        if "shap" in methods:
            result["shap"] = self._explain_shap(matrix, supplied, top_n)
        if "lime" in methods:
            result["lime"] = self._explain_lime(matrix, supplied, top_n)
        return result

    def _explain_shap(
        self, matrix: np.ndarray, supplied: dict[str, Any], top_n: int
    ) -> dict[str, Any]:
        try:
            explainer, basis = self._get_shap()
            if isinstance(explainer, list):
                rows, bases = [], []
                for tree_explainer in explainer:
                    out = tree_explainer(matrix)
                    rows.append(self._positive_class_values(out)[0])
                    base_vals = np.asarray(out.base_values).ravel()
                    bases.append(float(base_vals[-1]))
                row = np.mean(rows, axis=0)
                base = float(np.mean(bases))
            else:
                out = explainer(matrix)
                row = self._positive_class_values(out)[0]
                base = float(np.asarray(out.base_values).ravel()[0])

            contributions = self._aggregate_to_source_columns(row, supplied)
            return {
                "available": True,
                "basis": basis,
                "base_value": round(base, 6),
                "contributions": [c.as_dict() for c in contributions[:top_n]],
            }
        except Exception as exc:
            logger.exception("SHAP explanation failed")
            return {"available": False, "reason": f"SHAP unavailable: {exc}"}

    def _explain_lime(
        self, matrix: np.ndarray, supplied: dict[str, Any], top_n: int
    ) -> dict[str, Any]:
        try:
            explainer = self._get_lime()
            explanation = explainer.explain_instance(
                matrix[0],
                self.bundle.model.predict_proba,
                num_features=min(len(self.encoded_names), 20),
                labels=(1,),
            )
            weights = np.zeros(len(self.encoded_names))
            for idx, weight in explanation.as_map()[1]:
                weights[idx] = weight
            contributions = self._aggregate_to_source_columns(weights, supplied)
            return {
                "available": True,
                "local_prediction": round(
                    float(explanation.local_pred[0])
                    if explanation.local_pred is not None
                    else float("nan"),
                    6,
                ),
                "contributions": [c.as_dict() for c in contributions[:top_n]],
            }
        except Exception as exc:
            logger.exception("LIME explanation failed")
            return {"available": False, "reason": f"LIME unavailable: {exc}"}

    def global_importance(self, top_n: int = 15) -> dict[str, Any]:
        """Mean absolute SHAP value per CHAMP column over the background sample.

        This is the model's overall behaviour, not a statement about any
        individual variant.
        """
        if self._background is None:
            return {"available": False, "reason": "No background sample stored"}
        try:
            explainer, basis = self._get_shap()
            sample = self._background[: min(100, len(self._background))]
            if isinstance(explainer, list):
                stacked = [
                    np.abs(self._positive_class_values(e(sample))) for e in explainer
                ]
                mean_abs = np.mean(stacked, axis=0).mean(axis=0)
            else:
                mean_abs = np.abs(
                    self._positive_class_values(explainer(sample))
                ).mean(axis=0)
            totals: dict[str, float] = defaultdict(float)
            for name, value in zip(self.encoded_names, mean_abs):
                totals[champ.source_column_for(name)] += float(value)
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            return {
                "available": True,
                "method": "mean |SHAP value| over the training background sample",
                "basis": basis,
                "n_background": int(len(sample)),
                "features": [
                    {"feature": k, "importance": round(v, 6)} for k, v in ranked[:top_n]
                ],
            }
        except Exception as exc:
            logger.exception("Global SHAP importance failed")
            return {"available": False, "reason": str(exc)}


_service: ExplanationService | None = None


def get_explanation_service(bundle: ArtifactBundle) -> ExplanationService:
    """Process-wide singleton so explainers are constructed once."""
    global _service
    if _service is None or _service.bundle.version != bundle.version:
        _service = ExplanationService(bundle)
    return _service
