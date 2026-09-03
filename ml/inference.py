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

DEFAULT_MODEL_VERSION = "mmc-merged-v1"

#: One artifact version per feature set from final(1).ipynb. The API exposes
#: these as prediction modes; ``merged`` is the default because it is the only
#: one that sees both the mutation and the clinical record.
FEATURE_SET_VERSIONS: dict[str, str] = {
    "genomic": "mmc-genomic-v1",
    "clinical": "mmc-clinical-v1",
    "merged": "mmc-merged-v1",
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

    # -- introspection ----------------------------------------------------

    @property
    def version(self) -> str:
        return self.bundle.version

    @property
    def feature_set(self) -> str:
        return self.spec.name

    def input_schema(self) -> dict[str, Any]:
        """What a caller may supply, and which values are accepted.

        The frontend builds its form from this, so the UI cannot offer a value
        the model has never been fitted on. ``required`` holds the columns that
        were present in almost every training row; the rest are optional and are
        imputed explicitly (an unmeasured assay is recorded as unmeasured, not
        invented).
        """
        return {
            "feature_set": self.spec.name,
            "categorical": {
                col: sorted(v for v in values if v != ha.MISSING_CATEGORY)
                for col, values in self._offered.items()
            },
            # Columns where the listed values are suggestions rather than the
            # only accepted ones, because the column is an identifier or a
            # measurement written as text. See hemophilia_a.OPEN_VOCABULARY_THRESHOLD.
            "open_vocabulary": sorted(self._open),
            "numeric": {
                col: {
                    "description": FEATURE_LABELS.get(col, {}).get("description", col),
                    "required": col in self.spec.required,
                }
                for col in self.spec.numeric
            },
            "labels": {
                col: FEATURE_LABELS.get(col, {}).get("label", col)
                for col in self.spec.columns
            },
            "groups": {
                col: (
                    "genomic"
                    if col in ha.GENOMIC_CANDIDATES
                    else "clinical"
                    if col in ha.CLINICAL_CANDIDATES
                    else "other"
                )
                for col in self.spec.columns
            },
            "required": list(self.spec.required),
            "optional": [c for c in self.spec.columns if c not in self.spec.required],
        }

    # -- validation -------------------------------------------------------

    def validate(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Turn a raw input dict into a single-row frame in training column order.

        Required columns must be supplied. Optional ones may be omitted or left
        blank; a blank categorical becomes the explicit ``Unknown`` level the
        preprocessor was fitted with, and a blank number is imputed with the
        training median. A categorical value that is not in the fitted
        vocabulary raises with the allowed list attached, so the API can return
        a useful 422 rather than a confident prediction about a row the model
        has never seen.
        """
        if not isinstance(payload, dict):
            raise InputValidationError("Input must be an object of feature values.")

        row: dict[str, Any] = {}

        for col in self.spec.categorical:
            raw = payload.get(col, payload.get(_snake(col)))
            blank = raw is None or (isinstance(raw, str) and not raw.strip())
            if blank:
                if col in self.spec.required:
                    raise InputValidationError(
                        f"'{col}' is required.",
                        field=col,
                        allowed=self._allowed(col),
                    )
                row[col] = np.nan  # -> MISSING_CATEGORY in the fitted imputer
                continue

            value = str(raw).strip()
            known = self._categories.get(col, [])
            if value not in known:
                # Training folds case-only spelling variants onto the dominant
                # form (hemophilia_a.collapse_case_variants), so accept the same
                # spellings here rather than rejecting "missense" for "Missense".
                folded = {v.casefold(): v for v in known}
                value = folded.get(value.casefold(), value)
            if value not in known and col not in self._open:
                raise InputValidationError(
                    f"'{value}' is not a value of '{col}' that this model was "
                    f"trained on.",
                    field=col,
                    allowed=self._allowed(col),
                )
            # For an open-vocabulary column an unseen value is passed through:
            # the encoder maps it to the same "infrequent" bucket it learned
            # from the rare values in training, rather than to an all-zeros row.
            row[col] = value

        for col in self.spec.numeric:
            raw = payload.get(col, payload.get(_snake(col)))
            if raw is None or raw == "":
                if col in self.spec.required:
                    raise InputValidationError(f"'{col}' is required.", field=col)
                row[col] = np.nan  # imputed with the training median
                continue
            try:
                row[col] = float(raw)
            except (TypeError, ValueError):
                raise InputValidationError(
                    f"'{col}' must be a number, got {raw!r}.", field=col
                ) from None

        return ha.frame_for(self.spec, row)

    def _allowed(self, column: str) -> list[str]:
        return sorted(
            v for v in self._offered.get(column, []) if v != ha.MISSING_CATEGORY
        )

    # -- prediction -------------------------------------------------------

    def transform(self, payload: dict[str, Any]) -> np.ndarray:
        """Validated input -> the exact matrix the estimator was fitted on."""
        frame = self.validate(payload)
        return self.bundle.preprocessor.transform(frame)

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Estimate the probability that this record reports an inhibitor.

        The unit is one clinical record of one F8 mutation, as in MMC3. Several
        records may describe the same mutation, so an estimate is attributable
        to the record's mutation-and-assay description rather than to an
        individual patient's future.

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
            features_used=list(self.spec.columns),
            provenance_warning=self.bundle.provenance_warning,
        )


#: Human-readable labels for the source columns, so the UI does not have to show
#: raw database identifiers. Descriptions are drawn from the meaning of the
#: column in MMC2/MMC3; a column absent from this map falls back to its own name.
FEATURE_LABELS: dict[str, dict[str, str]] = {
    # --- genomic (MMC2) ---
    "mut_type": {"label": "Mutation type", "description": "Point, deletion, insertion, duplication, …"},
    "mut_effect": {"label": "Mutation effect", "description": "Missense, nonsense, frameshift, splice, …"},
    "location": {"label": "Location in the gene", "description": "Exon, intron, promoter, UTR"},
    "e_i_numb": {"label": "Exon / intron number", "description": "Which exon or intron carries the variant"},
    "locnumb": {"label": "Location number", "description": "Numbering of the affected region"},
    "aa_numb_old": {"label": "Amino-acid position (legacy numbering)", "description": "Residue position under the older numbering"},
    "aa_numb": {"label": "Amino-acid position", "description": "Residue position of the variant"},
    "codon_change": {"label": "Codon change", "description": "Reference codon to variant codon, e.g. CGC TGC"},
    "codon_first": {"label": "Reference codon", "description": "The codon before the change"},
    "codon_last": {"label": "Variant codon", "description": "The codon after the change"},
    "n_bp": {"label": "Base pairs affected", "description": "How many bases the variant spans"},
    "nuc_numb": {"label": "Nucleotide position", "description": "cDNA position of the variant"},
    "ntchange": {"label": "Nucleotide change", "description": "Base substitution, e.g. C>T"},
    "mut_syn": {"label": "cDNA notation (HGVS)", "description": "e.g. c.1834C>T"},
    "aa_change": {"label": "Amino-acid change", "description": "Reference residue to variant residue, e.g. Arg Cys"},
    "aa_first": {"label": "Reference amino acid", "description": "The residue before the change"},
    "aa_last": {"label": "Variant amino acid", "description": "The residue after the change"},
    "aa_syn": {"label": "Protein notation (HGVS)", "description": "e.g. p.Arg612Cys"},
    "CpG": {"label": "CpG dinucleotide", "description": "Whether the variant sits at a CpG site"},
    "utype": {"label": "Curated mutation subtype", "description": "Subtype recorded for the mutation"},
    # --- clinical (MMC3) ---
    "clotting": {"label": "FVIII clotting activity (%)", "description": "Reported one-stage clotting activity"},
    "discrep": {"label": "Assay discrepancy", "description": "Reported discrepancy between assays"},
    "ratio": {"label": "Activity ratio", "description": "Ratio between the reported assays"},
    "assay": {"label": "Assay generation", "description": "Which generation of assay was used"},
    "antigen": {"label": "FVIII antigen (%)", "description": "Reported FVIII antigen level"},
    "act/ant": {"label": "Activity / antigen ratio", "description": "Clotting activity divided by antigen"},
    "type": {"label": "Reported case type", "description": "Case classification recorded with the report"},
    "pa_race": {"label": "Reported population / origin", "description": "Population recorded for the case"},
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


def reset_services() -> None:
    """Drop cached services. Used by tests; not called by the application."""
    _services.clear()


def predict(payload: dict[str, Any], feature_set: str = "merged") -> dict[str, Any]:
    """Convenience wrapper around the cached service for one feature set."""
    return service_for_feature_set(feature_set).predict(payload).as_dict()
