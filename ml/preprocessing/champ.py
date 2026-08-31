"""CHAMP dataset loading, validation, normalisation and feature transformation.

This is the ONE place CHAMP is interpreted. Training and inference both import
from here, so the two cannot drift apart — the class of bug that made the
previous pipeline silently send nine zero-filled features to the model.

CHAMP is the CDC Hemophilia A Mutation Project registry. Each row is an F8
*variant*, not a patient, and ``History of Inhibitor`` is a registry aggregation
over the reports for that variant. Predictions built on it are therefore
variant-attributable, never an individual patient's probability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

DEFAULT_CHAMP_PATH = Path(__file__).resolve().parents[1] / "data" / "champ.csv"

LABEL_COLUMN = "History of Inhibitor"
POSITIVE_LABELS = {"yes"}
NEGATIVE_LABELS = {"no"}
#: Rows whose inhibitor history was never reported. Excluded from training, and
#: the exclusion is reported rather than imputed — see ``LabelReport``.
UNLABELLED_VALUES = {"not reported", ""}

CATEGORICAL_FEATURES = [
    "Variant Type",
    "Mechanism",
    "Domain",
    "Subtype",
    "In Poly A",
    "Reported Clinical Severity",
]
NUMERIC_FEATURES = ["exon_number", "codon_number"]
BINARY_FEATURES = ["is_intron"]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

#: Columns deliberately not used, with the reason. Kept in code (not only in
#: docs) so the justification travels with the pipeline.
EXCLUDED_COLUMNS: dict[str, str] = {
    "HGVS cDNA": "near-unique identifier (2296 distinct values over 2296 labelled rows)",
    "hg19 Coordinates": "near-unique genomic coordinate identifier",
    "HGVS Protein": "near-unique identifier",
    "Mature Protein": "near-unique identifier",
    "Comments": "free text, ~96% null",
    "Reference Number": (
        "LEAKAGE: identifies the source publication, not the variant. Grouped by "
        "reference, some studies are 100% inhibitor-positive and others 0%, "
        "because inhibitor-focused papers report inhibitor-positive variants. "
        "Including it would let the model recover the label from the citation."
    ),
    "Year Reported": (
        "reporting artifact rather than biology, and the column contains an "
        "out-of-range 0.0"
    ),
    "Severe (<1 U/dL)": "45-90% null; superseded by Reported Clinical Severity",
    "Moderate (1-5 U/dL)": "45-90% null; superseded by Reported Clinical Severity",
    "Mild (>5 U/dL)": "45-90% null; superseded by Reported Clinical Severity",
    "No FVIII level given": "45-90% null; superseded by Reported Clinical Severity",
    "Newly Added in the Current Version": "registry bookkeeping",
}

#: Columns where a difference in letter case is a data-entry typo and nothing
#: more, so values may be case-folded to the registry's dominant spelling.
#:
#: ``Domain`` is deliberately ABSENT. In FVIII nomenclature the lowercase a1, a2
#: and a3 are the *acidic regions*, which are biologically distinct from the
#: A1/A2/A3 domains. CHAMP contains both spellings (A1 750 rows vs a1 34,
#: A2 727 vs a2 19, A3 655 vs a3 43). Case-folding that column would silently
#: merge 96 acidic-region rows into the A-domains and corrupt the feature.
_CASE_INSENSITIVE_COLUMNS = (
    "Variant Type",
    "Mechanism",
    "Subtype",
    "In Poly A",
    "Reported Clinical Severity",
)

#: The dominant spelling for each case-folded group, taken from the registry
#: itself. Frozen here so normalisation is explicit and stable rather than
#: recomputed from whatever data happens to be loaded.
CASE_CANONICAL_FORMS: dict[str, str] = {
    # Mechanism
    "deletion": "Deletion",
    "duplication": "Duplication",
    "deletion/insertion": "Deletion/Insertion",
    # Subtype
    "heavy chain": "Heavy chain",
    "light chain": "Light chain",
    "single domain": "single domain",
    "multiple domains": "multiple domains",
    # In Poly A
    "n": "N",
    "y": "Y",
    # Reported Clinical Severity
    "severe": "Severe",
    "not reported": "Not reported",
    "mild/moderate": "Mild/Moderate",
}

MISSING_CATEGORY = "Unknown"

#: Categories seen fewer than this many times in the training split are folded
#: into an explicit "infrequent" bucket. CHAMP contains data-entry artifacts
#: such as a single ``Subtype == "165"`` row; without this, such a singleton
#: landing in the test split alone would abort the transform. Genuinely unknown
#: input values are rejected earlier, by PredictionService input validation,
#: which can produce a useful error message.
MIN_CATEGORY_FREQUENCY = 3


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Result of checking a raw CHAMP frame against the expected schema."""

    n_rows: int
    n_columns: int
    missing_columns: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_columns

    def raise_if_invalid(self) -> None:
        if self.missing_columns:
            raise ValueError(
                "CHAMP file is missing required columns: "
                + ", ".join(self.missing_columns)
            )


@dataclass
class LabelReport:
    """How many rows were usable, and how many were dropped and why."""

    n_total: int
    n_labelled: int
    n_positive: int
    n_excluded_unlabelled: int
    excluded_values: dict[str, int] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        return self.n_positive / self.n_labelled if self.n_labelled else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_labelled": self.n_labelled,
            "n_positive": self.n_positive,
            "positive_rate": round(self.positive_rate, 6),
            "n_excluded_unlabelled": self.n_excluded_unlabelled,
            "excluded_values": self.excluded_values,
            "caveat": (
                "Rows without a reported inhibitor history are excluded. That "
                "exclusion is very unlikely to be random, so the positive rate "
                "describes this registry's reporting, not population incidence."
            ),
        }


# --------------------------------------------------------------------------
# Loading and normalisation
# --------------------------------------------------------------------------


def _clean_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\r\n", " ").replace("\n", " ")).strip()


def load_champ(path: str | Path = DEFAULT_CHAMP_PATH) -> pd.DataFrame:
    """Read champ.csv and normalise column names only.

    The file on disk is never modified. Column names in the registry contain
    embedded newlines (e.g. ``"Severe \\r\\n(<1 U/dL)"``); those are collapsed to
    single spaces so downstream code can refer to them readably.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CHAMP dataset not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [_clean_column_name(c) for c in df.columns]
    # Registry exports carry trailing all-empty columns.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
    return df


def validate_champ(df: pd.DataFrame) -> ValidationReport:
    """Check a loaded frame has the columns this pipeline depends on."""
    required = set(CATEGORICAL_FEATURES) | {LABEL_COLUMN, "Exon", "Codon"}
    missing = sorted(required - set(df.columns))
    empty = sorted(c for c in df.columns if df[c].isna().all())
    counts = (
        df[LABEL_COLUMN].astype(str).str.strip().str.lower().value_counts().to_dict()
        if LABEL_COLUMN in df.columns
        else {}
    )
    return ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_columns=missing,
        empty_columns=empty,
        label_counts={str(k): int(v) for k, v in counts.items()},
    )


def _parse_exon(value: Any) -> tuple[float, int]:
    """Split an ``Exon`` cell into (number, is_intron).

    The registry stores both ``"14"`` and ``"Intron 22"`` in this column. The
    previous pipeline tested ``isinstance(x, str) and 'intron' in x`` against an
    integer column, so its ``exon_risk`` feature was constant 1.0 for every row.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (np.nan, 0)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return (np.nan, 0)
    is_intron = 1 if "intron" in text.lower() else 0
    match = re.search(r"\d+", text)
    number = float(match.group()) if match else np.nan
    return (number, is_intron)


def _to_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def normalise_champ(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the registry's own spelling/case inconsistencies to one form and
    derive ``exon_number``, ``is_intron`` and ``codon_number``.

    No rows are added or removed, and no label is altered.
    """
    out = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            continue
        # Object dtype holding np.nan for missing. Two dtype traps here:
        # pandas' StringDtype NA raises "boolean value of NA is ambiguous"
        # inside SimpleImputer, and Python None is NOT detected as missing by
        # SimpleImputer on object arrays (its mask is `X != X`, and
        # `None != None` is False), which would turn nulls into a literal
        # "None" category instead of MISSING_CATEGORY.
        values = [
            np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()
            for v in out[col]
        ]
        if col in _CASE_INSENSITIVE_COLUMNS:
            values = [
                np.nan
                if (not isinstance(v, str) or v == "")
                else CASE_CANONICAL_FORMS.get(v.casefold(), v)
                for v in values
            ]
        else:
            values = [np.nan if (not isinstance(v, str) or v == "") else v for v in values]
        out[col] = pd.Series(values, index=out.index, dtype=object)

    if "Exon" in out.columns:
        parsed = out["Exon"].map(_parse_exon)
        out["exon_number"] = [p[0] for p in parsed]
        out["is_intron"] = [p[1] for p in parsed]
    if "Codon" in out.columns:
        out["codon_number"] = out["Codon"].map(_to_number)

    return out


def make_labelled_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, LabelReport]:
    """Return (X_raw, y, report) for rows with a reported inhibitor history.

    Rows whose label is "Not reported" are dropped, never imputed. The count and
    the reason are returned so they can be recorded in artifact metadata.
    """
    # fillna before astype(str): under pandas 3 astype(str) preserves NaN, and
    # value_counts() then drops it, so blank labels would be excluded from the
    # dataset without appearing in the exclusion report.
    normalised = (
        df[LABEL_COLUMN].fillna("(missing)").astype(str).str.strip().str.lower()
    ).replace("", "(missing)")

    is_pos = normalised.isin(POSITIVE_LABELS)
    is_neg = normalised.isin(NEGATIVE_LABELS)
    keep = is_pos | is_neg

    excluded = normalised[~keep].value_counts(dropna=False).to_dict()

    X = df.loc[keep, FEATURE_COLUMNS].copy()
    y = is_pos[keep].astype(int)
    y.name = "inhibitor"

    report = LabelReport(
        n_total=len(df),
        n_labelled=int(keep.sum()),
        n_positive=int(is_pos.sum()),
        n_excluded_unlabelled=int((~keep).sum()),
        excluded_values={str(k): int(v) for k, v in excluded.items()},
    )
    return X.reset_index(drop=True), y.reset_index(drop=True), report


def load_champ_features(
    path: str | Path = DEFAULT_CHAMP_PATH,
) -> tuple[pd.DataFrame, pd.Series, LabelReport, ValidationReport]:
    """Full read path: load -> validate -> normalise -> label."""
    raw = load_champ(path)
    validation = validate_champ(raw)
    validation.raise_if_invalid()
    normalised = normalise_champ(raw)
    X, y, labels = make_labelled_dataset(normalised)
    return X, y, labels, validation


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """An unfitted transformer for the CHAMP feature space.

    Must be fitted on the training split ONLY. The previous pipeline imputed
    over the whole dataset before splitting, which leaks test-set statistics
    into training.

    Rare and unseen categories map to an explicit "infrequent" column rather
    than to an all-zeros row, so an out-of-vocabulary value is never silently
    indistinguishable from a genuine absence. Callers should still validate
    input against :func:`fitted_categories` first — ``PredictionService`` does —
    so the user gets a named error rather than a bucketed guess.
    """
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
            ),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def encoded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Column names produced by a *fitted* preprocessor, in model input order."""
    return [str(n) for n in preprocessor.get_feature_names_out()]


def source_column_for(encoded_name: str) -> str:
    """Map an encoded feature name back to the CHAMP column it came from.

    ``cat__Variant Type_Missense`` -> ``Variant Type``. Used so explanations
    never name a feature the clinician did not actually supply.
    """
    name = re.sub(r"^(cat|num|bin)__", "", encoded_name)
    for col in CATEGORICAL_FEATURES:
        if name == col or name.startswith(f"{col}_"):
            return col
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_") :]
    return name


def category_value_for(encoded_name: str) -> str | None:
    """The category a one-hot column represents, or None for numeric/binary."""
    name = re.sub(r"^(cat|num|bin)__", "", encoded_name)
    for col in CATEGORICAL_FEATURES:
        if name.startswith(f"{col}_"):
            return name[len(col) + 1 :]
    return None


def fitted_categories(preprocessor: ColumnTransformer) -> dict[str, list[str]]:
    """The category vocabulary each categorical feature was fitted on.

    The API exposes this so the frontend can only ever offer values the model
    actually knows.
    """
    encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
    return {
        col: [str(v) for v in cats]
        for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }
