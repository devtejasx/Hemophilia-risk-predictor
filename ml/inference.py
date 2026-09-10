"""The single inference entrypoint for the whole system.

Backend, scripts and tests all call ``PredictionService.predict``. There is no
second copy of the feature-building logic anywhere — that duplication is what
allowed training and serving to drift apart in the previous implementation.

The service is driven entirely by the artifact's own ``FeatureSpec``, so a
version trained on the genomic block accepts exactly the genomic columns, and a
version trained on the merged block accepts exactly the merged columns. Nothing
here names a column literally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.artifacts import ArtifactBundle, ArtifactError, load_bundle
from ml.preprocessing import hemophilia_a as ha

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "mmc2-mmc3-v1"

#: One artifact version per prediction mode. The names say which source table
#: the mode draws on: ``mmc2-genomic-v1`` sees only the MMC2 mutation
#: description, ``mmc3-clinical-v1`` only the aggregated MMC3 clinical record,
#: and ``mmc2-mmc3-v1`` - the default - is the fused model that sees both.
FEATURE_SET_VERSIONS: dict[str, str] = {
    "genomic": "mmc2-genomic-v1",
    "clinical": "mmc3-clinical-v1",
    "merged": "mmc2-mmc3-v1",
}

#: Risk bands. The boundary between "lower" and "elevated" is the model's own
#: calibrated threshold, not a hardcoded 0.5. The wording is deliberately
#: probabilistic: this is decision support, not a verdict.
RISK_LOWER = "Lower estimated risk"
RISK_ELEVATED = "Elevated estimated risk"


class InputValidationError(ValueError):
    """Input cannot be mapped onto the feature space the model was fitted on."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        allowed: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.allowed = allowed or []


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    prediction: int
    risk: str
    risk_category: str
    threshold: float
    model_version: str
    feature_set: str
    preprocessing_version: str
    features_used: list[str]
    provenance_warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "risk": self.risk,
            "probability": self.probability,
            "risk_category": self.risk_category,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "feature_set": self.feature_set,
            "preprocessing_version": self.preprocessing_version,
            "features_used": self.features_used,
            "provenance_warning": self.provenance_warning,
        }


class PredictionService:
    """Loads one model version once and serves predictions from it."""

    def __init__(
        self,
        version: str = DEFAULT_MODEL_VERSION,
        artifacts_dir: str | None = None,
    ) -> None:
        self.bundle: ArtifactBundle = load_bundle(version, artifacts_dir)
        self.spec: ha.FeatureSpec = self.bundle.feature_spec
        # Everything the encoder saw, used for strict validation…
        self._categories = ha.fitted_categories(self.bundle.preprocessor, self.spec)
        # …the subset with its own encoded column, which is what the UI offers…
        self._offered = ha.frequent_categories(self.bundle.preprocessor, self.spec)
        # …and the columns where an unseen value is accepted rather than rejected.
        self._open = ha.open_vocabulary_columns(self.bundle.preprocessor, self.spec)

        # The model consumes aggregates (clotting_mean, cli_phe_mode); a caller
        # supplies the raw field (clotting, cli_phe). Map each raw field onto
        # the categorical aggregate that carries its fitted vocabulary, and
        # treat everything else as a measurement.
        self._vocab_column: dict[str, str] = {}
        for column in self.spec.categorical:
            self._vocab_column.setdefault(ha.aggregate_source_column(column), column)
        self._categorical_inputs = {
            f for f in self.spec.inputs if f in self._vocab_column
        }
        self._measurement_inputs = {
            f for f in self.spec.inputs if f not in self._categorical_inputs
        }
        self._open_inputs = {
            f for f, column in self._vocab_column.items() if column in self._open
        }

    # -- introspection ----------------------------------------------------

    @property
    def version(self) -> str:
        return self.bundle.version

    @property
    def feature_set(self) -> str:
        return self.spec.name

    def input_schema(self) -> dict[str, Any]:
        """What a caller may supply, and which values are accepted.

        Keyed on the **raw MMC2/MMC3 fields**, not on the aggregates the model
        consumes: a caller describes one mutation and one clinical record, and
        the service derives the mean/median/min/max/censoring features from it
        exactly as training did. The frontend builds its form from this, so the
        UI cannot offer a value the model was never fitted on.

        ``required`` holds the fields present in almost every training row; the
        rest are optional and are imputed explicitly, so an unmeasured assay is
        recorded as unmeasured rather than invented.
        """
        return {
            "feature_set": self.spec.name,
            "categorical": {
                field_name: sorted(
                    v
                    for v in self._offered.get(self._vocab_column[field_name], [])
                    if v != ha.MISSING_CATEGORY
                )
                for field_name in self.spec.inputs
                if field_name in self._categorical_inputs
            },
            # Fields where the listed values are suggestions rather than the only
            # accepted ones. See hemophilia_a.OPEN_VOCABULARY_THRESHOLD.
            "open_vocabulary": sorted(self._open_inputs),
            "numeric": {
                field_name: {
                    "description": FEATURE_LABELS.get(field_name, {}).get(
                        "description", field_name
                    ),
                    "required": field_name in self.spec.required,
                    # The source files write these as bounds and ranges, and so
                    # may a caller: "<1" is a real reading, not a typo.
                    "accepts_censored": True,
                }
                for field_name in self.spec.inputs
                if field_name in self._measurement_inputs
            },
            "labels": {
                field_name: FEATURE_LABELS.get(field_name, {}).get("label", field_name)
                for field_name in self.spec.inputs
            },
            "groups": {
                field_name: (
                    "genomic"
                    if field_name in ha.GENOMIC_CANDIDATES
                    else "clinical"
                    if field_name in ha.CLINICAL_CANDIDATES
                    else "other"
                )
                for field_name in self.spec.inputs
            },
            "required": list(self.spec.required),
            "optional": [f for f in self.spec.inputs if f not in self.spec.required],
            # What the model actually consumes, for transparency. Callers do not
            # send these; the service derives them.
            "model_features": list(self.spec.columns),
        }

    # -- validation -------------------------------------------------------

    def validate(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Raw input dict -> the model's feature row, or a named error.

        Required fields must be supplied. Optional ones may be omitted or left
        blank; a blank categorical becomes the explicit ``Unknown`` level the
        preprocessor was fitted with, and a blank measurement is imputed with
        the training median.

        A categorical value outside the fitted vocabulary raises with the
        allowed list attached, so the API returns a useful 422 rather than a
        confident prediction about a row the model has never seen. A
        measurement is accepted in any form the source files use - a number,
        a bound like ``"<1"``, or a range like ``"1 to 5"`` - and is rejected
        only if it cannot be parsed at all.
        """
        if not isinstance(payload, dict):
            raise InputValidationError("Input must be an object of feature values.")

        record: dict[str, Any] = {}

        for field_name in self.spec.inputs:
            raw = payload.get(field_name, payload.get(_snake(field_name)))
            blank = raw is None or (isinstance(raw, str) and not raw.strip())
            if blank:
                if field_name in self.spec.required:
                    raise InputValidationError(
                        f"'{field_name}' is required.",
                        field=field_name,
                        allowed=self._allowed(field_name),
                    )
                continue  # absent -> NaN -> the fitted imputer handles it

            if field_name in self._categorical_inputs:
                record[field_name] = self._validated_category(field_name, raw)
            else:
                record[field_name] = self._validated_measurement(field_name, raw)

        return ha.build_input_row(self.spec, record)

    def _validated_category(self, field_name: str, raw: Any) -> str:
        value = str(raw).strip()
        known = self._categories.get(self._vocab_column[field_name], [])
        if value not in known:
            # Training folds case-only spelling variants onto the dominant form
            # (hemophilia_a.collapse_case_variants), so accept the same
            # spellings here rather than rejecting "missense" for "Missense".
            folded = {v.casefold(): v for v in known}
            value = folded.get(value.casefold(), value)
        if value not in known and field_name not in self._open_inputs:
            raise InputValidationError(
                f"'{value}' is not a value of '{field_name}' that this model was "
                f"trained on.",
                field=field_name,
                allowed=self._allowed(field_name),
            )
        # For an open-vocabulary field an unseen value is passed through: the
        # encoder maps it to the same "infrequent" bucket it learned from the
        # rare values in training, rather than to an all-zeros row.
        return value

    def _validated_measurement(self, field_name: str, raw: Any) -> Any:
        """Check the value parses, then pass the **original** through.

        The raw string is what reaches ``build_input_row``, so the censoring
        indicator is derived by the same parser training used. Validating here
        and parsing there would be two chances to disagree; parsing once, in
        the shared function, is one.
        """
        _, censoring = ha.parse_measurement(raw)
        if censoring is None:
            raise InputValidationError(
                f"'{field_name}' must be a measurement - a number, a bound like "
                f"'<1', or a range like '1 to 5'. Got {raw!r}.",
                field=field_name,
            )
        return raw

    def _allowed(self, field_name: str) -> list[str]:
        column = self._vocab_column.get(field_name)
        if column is None:
            return []
        return sorted(
            v for v in self._offered.get(column, []) if v != ha.MISSING_CATEGORY
        )

    # -- prediction -------------------------------------------------------

    def transform(self, payload: dict[str, Any]) -> np.ndarray:
        """Validated input -> the exact matrix the estimator was fitted on."""
        frame = self.validate(payload)
        return self.bundle.preprocessor.transform(frame)

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Estimate the probability that this mutation is reported with an inhibitor.

        The unit is one F8 **mutation**, described by its MMC2 genomic block and
        the aggregate of the MMC3 clinical records reporting it. The model was
        fitted on mutations, so an estimate is attributable to the
        mutation-and-assay description supplied here, never to an individual
        patient's future.

        Raises InputValidationError for bad input and ArtifactError for a broken
        model. It never falls back to a hand-written formula: a caller must be
        able to distinguish a prediction from a failure.
        """
        matrix = self.transform(payload)

        try:
            probability = float(self.bundle.model.predict_proba(matrix)[0][1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ArtifactError(
                f"Model '{self.version}' failed to produce a probability: {exc}"
            ) from exc

        threshold = self.bundle.threshold
        elevated = probability >= threshold

        return PredictionResult(
            probability=round(probability, 6),
            prediction=int(elevated),
            risk="High" if elevated else "Low",
            risk_category=RISK_ELEVATED if elevated else RISK_LOWER,
            threshold=threshold,
            model_version=self.version,
            feature_set=self.spec.name,
            preprocessing_version=str(
                self.bundle.metadata.get("preprocessing_version", "unknown")
            ),
            features_used=list(self.spec.inputs),
            provenance_warning=self.bundle.provenance_warning,
        )


#: Human-readable labels for the *raw* MMC2/MMC3 fields a caller supplies, so
#: the UI does not have to show database identifiers. The model's own features
#: are aggregates of these; ``FeatureSpec.source_column_for`` maps a model
#: feature back to the field named here, which is what explanations report. A
#: field absent from this map falls back to its own name.
FEATURE_LABELS: dict[str, dict[str, str]] = {
    # --- genomic (MMC2) ---
    "mut_type": {"label": "Mutation type", "description": "Point, deletion, insertion, duplication, …"},
    "mut_effect": {"label": "Mutation effect", "description": "Missense, nonsense, frameshift, splice, …"},
    "location": {"label": "Location in the gene", "description": "Exon, intron, promoter, UTR"},
    "e_i_numb": {"label": "Exon / intron number", "description": "Which exon or intron carries the variant"},
    "locnumb": {"label": "Location number", "description": "Numbering of the affected region"},
    "aa_numb": {"label": "Amino-acid position", "description": "Residue position of the variant"},
    "codon_first": {"label": "Reference codon", "description": "The codon before the change"},
    "codon_last": {"label": "Variant codon", "description": "The codon after the change"},
    "n_bp": {"label": "Base pairs affected", "description": "How many bases the variant spans"},
    "nuc_numb": {"label": "Nucleotide position", "description": "cDNA position of the variant"},
    "ntchange": {"label": "Nucleotide change", "description": "Base substitution, e.g. C>T"},
    "aa_first": {"label": "Reference amino acid", "description": "The residue before the change"},
    "aa_last": {"label": "Variant amino acid", "description": "The residue after the change"},
    "CpG": {"label": "CpG dinucleotide", "description": "Whether the variant sits at a CpG site"},
    # --- clinical (MMC3) ---
    "clotting": {"label": "FVIII clotting activity (%)", "description": "Reported one-stage clotting activity. Bounds and ranges are accepted: '<1', '>5', '1 to 5'"},
    "discrep": {"label": "Assay discrepancy", "description": "Reported discrepancy between assays"},
    "ratio": {"label": "Activity ratio", "description": "Ratio between the reported assays"},
    "antigen": {"label": "FVIII antigen (%)", "description": "Reported FVIII antigen level. Bounds such as '<1' are accepted"},
    "act/ant": {"label": "Activity / antigen ratio", "description": "Clotting activity divided by antigen"},
    "cli_phe": {"label": "Clinical severity", "description": "Severe, moderate or mild phenotype"},
}


def _snake(column: str) -> str:
    """'act/ant' -> 'act_ant', so JSON clients may use either form."""
    return column.lower().replace(" ", "_").replace("/", "_")


_services: dict[str, PredictionService] = {}


def get_prediction_service(
    version: str | None = None, artifacts_dir: str | None = None
) -> PredictionService:
    """Process-wide cache, so each model version is loaded once per process."""
    key = version or DEFAULT_MODEL_VERSION
    service = _services.get(key)
    if service is None:
        service = PredictionService(key, artifacts_dir)
        _services[key] = service
    return service


def service_for_feature_set(
    feature_set: str = "merged", artifacts_dir: str | None = None
) -> PredictionService:
    """The model version that serves one prediction mode."""
    if feature_set not in FEATURE_SET_VERSIONS:
        raise InputValidationError(
            f"Unknown feature set '{feature_set}'.",
            field="feature_set",
            allowed=sorted(FEATURE_SET_VERSIONS),
        )
    return get_prediction_service(FEATURE_SET_VERSIONS[feature_set], artifacts_dir)


def predict(payload: dict[str, Any], feature_set: str = "merged") -> dict[str, Any]:
    """Convenience wrapper around the cached service for one feature set."""
    return service_for_feature_set(feature_set).predict(payload).as_dict()
