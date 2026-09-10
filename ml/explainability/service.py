"""The canonical explanation service: SHAP and LIME over the served model.

Two rules govern everything here:

1. **Never imply an input the caller did not give.** Explanations are reported
   against the original MMC2/MMC3 columns (``mut_type``, ``cli_phe``, ...), not
   against post-encoding names like ``cat__mut_type_Point`` which mean nothing to
   a reader. The column list comes from the artifact's own ``FeatureSpec``, so a
   genomic-only model can never name a clinical field and vice versa. A field the
   caller left blank may still appear — its absence is itself an input to the
   model — but it carries ``supplied: false`` and a null value, never a value the
   caller never typed.
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
from ml.preprocessing import hemophilia_a as ha

logger = logging.getLogger(__name__)

#: Explanations are expensive relative to a tree prediction, so the number of
#: features returned is capped rather than dumping every encoded column — 320
#: of them for the merged model.
DEFAULT_TOP_N = 8

#: LIME perturbs the encoded vector. Over a few hundred encoded columns the
#: default neighbourhood size is slow enough to matter on a request path, so it
#: is reduced here; the local model is still fitted on a thousand samples.
LIME_NUM_SAMPLES = 1000


@dataclass
class FeatureContribution:
    feature: str
    """The original source column, e.g. 'mut_type'."""

    label: str
    """A human-readable name for it, e.g. 'Mutation type'."""

    value: Any
    """The value the caller supplied, or None when the field was left blank."""

    supplied: bool
    """False when the caller omitted this field.

    An omitted optional field still reaches the model — as the explicit
    ``Unknown`` category, or as the training median with its missing indicator
    set — so its effect is real and worth reporting. The flag exists so a
    contribution is never read as a value the caller entered.
    """

    contribution: float
    """Signed effect on the predicted probability. Positive = raises risk."""

    direction: str = field(init=False)

    def __post_init__(self) -> None:
        # A contribution of exactly zero is neither. Reporting it as
        # "decreases" alongside a displayed +0.0000 reads as a contradiction.
        if self.contribution > 0:
            self.direction = "increases"
        elif self.contribution < 0:
            self.direction = "decreases"
        else:
            self.direction = "no effect"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": self.value,
            "supplied": self.supplied,
            "contribution": round(float(self.contribution), 6),
            "direction": self.direction,
        }


class ExplanationService:
    """SHAP (global + local) and LIME (local) for one loaded model version."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self.spec = bundle.feature_spec
        self.encoded_names = ha.encoded_feature_names(bundle.preprocessor)
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

    @staticmethod
    def _unwrap(estimator: Any) -> Any:
        """The estimator itself, or the final step of a pipeline wrapping it."""
        steps = getattr(estimator, "steps", None)
        return steps[-1][1] if steps else estimator

    @staticmethod
    def _is_tree_model(estimator: Any) -> bool:
        """Whether shap.TreeExplainer can read this estimator directly.

        ``estimators_`` covers the scikit-learn forests; ``get_booster`` covers
        XGBoost, which the merged feature set selects.
        """
        return hasattr(estimator, "estimators_") or hasattr(estimator, "get_booster")

    def _base_tree_estimators(self) -> list[Any]:
        """The tree models underneath a CalibratedClassifierCV, if any.

        Calibration fits one clone of the base estimator per CV fold. Exact
        TreeSHAP over those clones runs in milliseconds, where a model-agnostic
        permutation explainer over the calibrated wrapper takes tens of seconds
        per row — far too slow for an API request.

        The attribution describes the underlying ensemble rather than the
        isotonic output. Isotonic calibration is a monotone map, so it rescales
        probabilities without reordering feature effects; the ranking and sign of
        contributions are unchanged. Reported in the payload as
        ``basis: "uncalibrated ensemble"`` so this is never implied to be an
        attribution of the calibrated probability itself.
        """
        model = self.bundle.model
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if not calibrated:
            return []
        estimators = []
        for entry in calibrated:
            # The selected estimator may be a RandomForest, an XGBClassifier, or
            # an imblearn Pipeline wrapping one. All three are readable by
            # TreeExplainer once unwrapped; anything else falls through to the
            # model-agnostic path.
            base = self._unwrap(getattr(entry, "estimator", None))
            if base is not None and self._is_tree_model(base):
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
                # The matrix is one-hot and standardised, not raw continuous
                # measurements; quartile discretisation of a 0/1 column produces
                # degenerate bins.
                discretize_continuous=False,
                random_state=42,
            )
        return self._lime_explainer

    # -- name mapping -----------------------------------------------------

    def _aggregate_to_source_columns(
        self, weights: np.ndarray, supplied: dict[str, Any]
    ) -> list[FeatureContribution]:
        """Sum encoded-column effects back onto the source column they came from.

        A single categorical input expands to many one-hot columns; reporting
        them separately would overstate the number of factors and name
        categories the caller did not choose.
        """
        totals: dict[str, float] = defaultdict(float)
        for name, weight in zip(self.encoded_names, weights):
            totals[self.spec.source_column_for(name)] += float(weight)

        contributions = [
            FeatureContribution(
                feature=col,
                label=_label_for(col),
                value=supplied.get(col),
                supplied=supplied.get(col) is not None,
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
            "feature_set": self.spec.name,
            "unit_of_explanation": (
                "One F8 mutation: its MMC2 genomic description together with the "
                "aggregate of the MMC3 clinical records reporting it. "
                "Contributions describe the model's use of that mutation's "
                f"{self.spec.name} features, not an individual patient's future."
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
                num_features=min(len(self.encoded_names), 40),
                num_samples=LIME_NUM_SAMPLES,
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
        """Mean absolute SHAP value per source column over the background sample.

        This is the model's overall behaviour, not a statement about any
        individual record.
        """
        if self._background is None:
            return {"available": False, "reason": "No background sample stored"}
        try:
            explainer, basis = self._get_shap()
            sample = self._background[: min(64, len(self._background))]
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
                totals[self.spec.source_column_for(name)] += float(value)
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            return {
                "available": True,
                "method": "mean |SHAP value| over the training background sample",
                "basis": basis,
                "feature_set": self.spec.name,
                "n_background": int(len(sample)),
                "features": [
                    {"feature": k, "label": _label_for(k), "importance": round(v, 6)}
                    for k, v in ranked[:top_n]
                ],
            }
        except Exception as exc:
            logger.exception("Global SHAP importance failed")
            return {"available": False, "reason": str(exc)}


def _label_for(column: str) -> str:
    """The human-readable name for a source column, or the column itself."""
    from ml.inference import FEATURE_LABELS

    return FEATURE_LABELS.get(column, {}).get("label", column)

