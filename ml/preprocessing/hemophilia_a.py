"""MMC2 + MMC3 Hemophilia A dataset: loading, validation, fusion, and features.

This is the ONE place the dataset is interpreted. Training, the API and the
tests all import from here, so training-time and inference-time preprocessing
cannot drift apart.

The two supplementary tables of the source publication:

* **MMC2** — one row per *mutation* (``mut_id``). Genomic description of an F8
  (or, for six rows, F9) variant: type, effect, location, codon/amino-acid
  change, position.
* **MMC3** — one row per *clinical record*. Several records may report the same
  mutation, so ``mut_id`` repeats. Carries the assay values and the
  ``Inhibitors`` field that is the prediction target.

Fusion happens on ``mut_id`` and the modelling unit is the **mutation**, not the
clinical record::

    MMC2 (one row per mut_id)  ──┐
                                 ├── join on mut_id ── aggregate per mut_id ──▶ one row per mutation
    MMC3 (many rows per mut_id) ─┘

Aggregating first, rather than modelling raw clinical records, is what makes the
evaluation honest. A mutation reported by 104 clinical records would otherwise
contribute 104 near-identical rows carrying an identical genomic block; the
majority classifier over those rows looks accurate because it has memorised the
mutation, not because it has learned anything about inhibitor risk. After
aggregation each mutation is exactly one row, so a mutation cannot be split
across train and test even in principle — the grouped split asserts it anyway.

A mutation whose records disagree about the inhibitor outcome has no single
supervised label. Those mutations are counted, reported and **excluded** from
training rather than resolved by majority vote, which would invent a certainty
the source data does not contain.

``mut_id`` itself is never a feature.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Paths (environment-overridable; never absolute in code)
# --------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MMC2_FILENAME = "BVTH_VTH-2024-000215-mmc2.csv"
MMC3_FILENAME = "BVTH_VTH-2024-000215-mmc3.csv"


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR") or DEFAULT_DATA_DIR)


def mmc2_path() -> Path:
    return Path(os.getenv("MMC2_PATH") or (data_dir() / MMC2_FILENAME))


def mmc3_path() -> Path:
    return Path(os.getenv("MMC3_PATH") or (data_dir() / MMC3_FILENAME))


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

GROUP_COLUMN = "mut_id"
LABEL_COLUMN = "Inhibitors"
TARGET_COLUMN = "target"

#: ``Inhibitors`` is free text. Only an explicit yes/no is a label; everything
#: else ("Not reported", "Not", blank, a severity accidentally typed into the
#: field) is excluded and reported, never imputed.
POSITIVE_LABELS = {"yes"}
NEGATIVE_LABELS = {"no"}

#: MMC2 covers more than one gene. Hemophilia A is F8.
GENE_COLUMN = "g_name"
GENE_OF_INTEREST = "F8"

#: Genomic description of the mutation, from MMC2. One value per ``mut_id``.
GENOMIC_CANDIDATES: tuple[str, ...] = (
    "mut_type",
    "mut_effect",
    "location",
    "e_i_numb",
    "locnumb",
    "aa_numb",
    "codon_first",
    "codon_last",
    "n_bp",
    "nuc_numb",
    "ntchange",
    "aa_first",
    "aa_last",
    "CpG",
)

#: Genomic columns that are stored as text but are really measurements: a
#: position along the gene or a length. ``n_bp`` carries unit suffixes ("7kb")
#: and open bounds (">50"); ``e_i_numb`` and ``locnumb`` carry a handful of
#: ranges. Parsed by :func:`parse_measurement` rather than one-hot encoded,
#: because exon 14 is genuinely between exon 13 and exon 15.
GENOMIC_MEASUREMENTS: tuple[str, ...] = (
    "e_i_numb",
    "locnumb",
    "aa_numb",
    "n_bp",
    "nuc_numb",
)

#: Clinical measurements from MMC3, aggregated per mutation. Every one is
#: written as free text in the source file — 4,237 of 6,492 non-null
#: ``clotting`` entries are censored ("<1", ">5") or ranges ("1 to 5") — so all
#: of them go through :func:`parse_measurement`.
CLINICAL_MEASUREMENTS: tuple[str, ...] = (
    "clotting",
    "antigen",
    "ratio",
    "act/ant",
    "discrep",
)

#: The clinical severity phenotype. Ordinal, so it gets a score as well as the
#: per-category proportions.
SEVERITY_COLUMN = "cli_phe"

#: Ordinal scoring for ``cli_phe``. Combination phenotypes ("Severe/moderate")
#: score between their parts. "Unclassified" and "Not reported" have no rank and
#: stay missing rather than being pushed to a middle value.
SEVERITY_SCORES: dict[str, float] = {
    "mild": 1.0,
    "mild/moderate": 1.5,
    "moderate/mild": 1.5,
    "moderate": 2.0,
    "moderate/severe": 2.5,
    "severe/moderate": 2.5,
    "severe": 3.0,
}

#: The severity buckets that get a proportion feature.
SEVERITY_BUCKETS: tuple[str, ...] = ("mild", "moderate", "severe")

#: The raw MMC3 columns that make up the clinical block, before aggregation.
CLINICAL_CANDIDATES: tuple[str, ...] = (*CLINICAL_MEASUREMENTS, SEVERITY_COLUMN)

#: Aggregate statistics computed per mutation for each measurement column.
MEASUREMENT_AGGREGATES: tuple[str, ...] = ("mean", "median", "min", "max")

#: Columns that must never become features, with the reason. Kept in code (not
#: only in documentation) so the justification travels with the pipeline.
#:
#: Four groups: the label and its curated mirrors; fields that only exist
#: *because* an inhibitor assay was performed; reporting/administrative
#: bookkeeping; and near-unique identifiers that would let the model memorise a
#: mutation instead of learning from its biology.
EXCLUDED_COLUMNS: dict[str, str] = {
    "mut_id": "grouping key for the split, not a predictor",
    "target": "the encoded label",
    "Inhibitors": "the label itself",
    # -- the label's curated mirrors in MMC2 -----------------------------
    "uinhibitor": (
        "LEAKAGE: MMC2's curated inhibitor status for the mutation. 97.1% of "
        "merged records with uinhibitor='Yes' are positive and 0.2% of "
        "uinhibitor='No' are, so it reproduces the label almost exactly."
    ),
    "ucomments": "free-text curator note attached to the curated inhibitor block",
    "uclotting": "curated mirror of the clinical assay fields",
    "udiscrep": "curated mirror of the clinical assay fields",
    "uratio": "curated mirror of the clinical assay fields",
    "uantigen": "curated mirror of the clinical assay fields",
    "useverity": "curated mirror of the clinical severity field",
    # -- fields that exist only because inhibitor testing happened -------
    "type": (
        "LEAKAGE: inhibitor kinetic type (I / II / NU=Nijmegen units). It is "
        "recorded only when an inhibitor assay was run, so both its value and "
        "its presence are post-outcome information: 3,652 of 4,962 merged "
        "records leave it blank and the non-blank ones carry a different "
        "positive rate (NU 22.6%, II 6.4%, I 3.8%) against a 16.8% base rate."
    ),
    "utype": (
        "LEAKAGE: MMC2's curated copy of the inhibitor kinetic type, same "
        "reasoning as 'type'. It sits in the curated u* inhibitor block."
    ),
    "assay": (
        "LEAKAGE: which inhibitor assay produced the reading ('1st' / '2nd'). "
        "99.5% null, and non-null only for tested patients."
    ),
    # -- reporting and administrative bookkeeping ------------------------
    "comments": "free text",
    "pri_comments": "free text",
    "reference": (
        "source publication; grouping by publication recovers the label because "
        "inhibitor-focused papers report inhibitor-positive cases"
    ),
    "ref_id": "identifier for the source publication (see 'reference')",
    "all_id": "record identifier",
    "pa_id": "patient identifier",
    "g_id": "gene identifier, constant after the F8 filter",
    "d_id": "database identifier",
    "subgroups": "record bookkeeping",
    "pub_lab": "reporting laboratory; a reporting artifact, not biology",
    "pa_race": (
        "the reporting country, not an ancestry: it repeats 'pub_lab' verbatim "
        "in most rows. Its positive rate swings from 8.8% (India) to 48.7% "
        "(Italy), which tracks which cohorts were published, not biology."
    ),
    "date added": "reporting artifact",
    "Date added to HADB": "reporting artifact",
    "Count_mut_id": "how many records mention the mutation; a reporting artifact",
    "n_clinical_records": (
        "how many MMC3 records back a mutation. A reporting artifact like "
        "Count_mut_id, and after conflicting mutations are excluded it is "
        "actively misleading: multi-record mutations that survive are "
        "unanimous, and unanimous-negative far more often than "
        "unanimous-positive, so the count encodes the exclusion rule rather "
        "than biology (22.2% positive at one record, 0% above four)."
    ),
    "bleed_tool": "2 non-null values in 10,064 records",
    "bleed_score": "0 non-null values in 10,064 records",
    "mutations": "constant ('Natural') for every merged record",
    # -- near-unique identifiers and redundant composites ----------------
    "mut_syn": (
        "IDENTIFIER: the HGVS coding string. 2,509 distinct values across 2,639 "
        "mutations (95.1% unique), so it names the mutation rather than "
        "describing it. Its components are kept separately (nuc_numb, ntchange)."
    ),
    "aa_syn": (
        "IDENTIFIER: the HGVS protein string, 81.4% unique. Same reasoning as "
        "mut_syn; aa_first, aa_numb and aa_last carry the same content."
    ),
    "aa_change": (
        "redundant composite of aa_first + aa_numb + aa_last (850 distinct "
        "values, 32.2% unique), all three of which are kept separately"
    ),
    "codon_change": (
        "redundant composite of codon_first + codon_last, both of which are kept"
    ),
    "aa_numb_old": (
        "the pre-2001 amino-acid numbering of aa_numb; correlation 0.999 with "
        "it, differing only by the 19-residue signal-peptide offset"
    ),
}

#: A categorical feature whose distinct-value count exceeds this fraction of the
#: modelling rows is an identifier rather than a category. Nothing may reach the
#: model above it; :func:`identifier_like_columns` enforces the rule and the
#: test suite asserts on it, so a future column added to the candidate lists
#: cannot silently reintroduce ``mut_syn``-style memorisation.
MAX_CATEGORY_UNIQUENESS_RATIO = 0.5

#: A categorical value seen fewer than this many times in the *training* split is
#: folded into an explicit "infrequent" bucket rather than getting its own
#: column. Several source columns are near-identifiers of the mutation
#: (``mut_syn`` has 2,510 distinct values), and without this the encoded matrix
#: would be mostly singleton columns.
MIN_CATEGORY_FREQUENCY = 3

#: Missing categorical values become this explicit level. See
#: ``PIPELINE_DECISIONS`` — several clinical columns are 75-99% null, and
#: imputing them with the mode would erase the fact that they were never
#: measured.
MISSING_CATEGORY = "Unknown"

#: A categorical column with more distinct training values than this is treated
#: as an *open* vocabulary: an identifier or a measurement written as text
#: (``mut_syn`` holds 1,610 HGVS strings, ``clotting`` 113 activity readings)
#: rather than a fixed set of categories. A value the model has not seen is
#: accepted for such a column and lands in the encoder's "infrequent" bucket —
#: exactly what training did with the rare values it *did* see. Columns with a
#: small closed vocabulary stay strict, so a wrong ``mut_type`` is still a 422
#: with the accepted values attached rather than a confident guess.
OPEN_VOCABULARY_THRESHOLD = 30

#: A source column whose training-split missing rate is at or below this is
#: marked *required* in the API input schema; the rest are optional and are
#: imputed explicitly. Computed at training time and stored in metadata, so the
#: form the frontend renders follows the data rather than a hand-written list.
REQUIRED_MAX_MISSING_RATE = 0.05

FEATURE_SET_NAMES = ("genomic", "clinical", "merged")

PIPELINE_DECISIONS: tuple[str, ...] = (
    "The modelling unit is the mutation, not the clinical record. MMC3's "
    "records are aggregated per mut_id before any split, so a mutation "
    "reported 104 times contributes one row rather than 104 rows sharing an "
    "identical genomic block.",
    "A mutation whose clinical records disagree about the inhibitor outcome is "
    "excluded from supervised training and counted separately. Majority vote "
    "would manufacture a label the source data does not support.",
    "Censored and range measurements are parsed rather than treated as text. "
    "'<1' becomes 1.0 with a left-censoring indicator, '1 to 5' becomes its "
    "midpoint; 4,237 of 6,492 non-null clotting values are written this way, so "
    "one-hot encoding them would discard the single strongest clinical signal.",
    "Categorical missing values are imputed with an explicit 'Unknown' level "
    "rather than the most-frequent value. Columns such as discrep (99.3% null) "
    "are missing because the measurement was never taken; filling them with the "
    "mode invents a measurement and destroys the only signal they carry.",
    "String values are whitespace-stripped and case-only spelling variants are "
    "folded onto the dominant spelling before encoding. The raw files contain "
    "both 'Point' and 'Point ', 'Missense' and 'missense', 'Severe' and "
    "'Severe '; those are the same value typed twice.",
    "Numeric imputation adds a missing-indicator column, so an imputed median "
    "is distinguishable from an observed value.",
    "Model selection uses StratifiedGroupKFold on mut_id, so cross-validation "
    "obeys the same grouping constraint as the outer split.",
    "The served model is fitted on the training split only, and its threshold "
    "is chosen on the validation split. The held-out test split is touched "
    "exactly once, for the numbers reported in metrics.json.",
)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass
class SourceReport:
    """What one raw file contains, before anything is joined."""

    name: str
    path: str
    n_rows: int
    n_columns: int
    n_unique_mut_id: int
    n_duplicate_rows: int
    n_duplicate_mut_id: int
    missing_columns: list[str] = field(default_factory=list)
    missing_values_total: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing_columns

    def raise_if_invalid(self) -> None:
        if self.missing_columns:
            raise ValueError(
                f"{self.name} ({self.path}) is missing required columns: "
                + ", ".join(self.missing_columns)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_unique_mut_id": self.n_unique_mut_id,
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_duplicate_mut_id": self.n_duplicate_mut_id,
            "missing_values_total": self.missing_values_total,
        }


@dataclass
class LabelReport:
    """How many MMC3 records carried a usable ``Inhibitors`` value."""

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
            "encoding": "Inhibitors: Yes -> 1, No -> 0 (case- and space-insensitive)",
            "caveat": (
                "Records without an explicit Yes/No inhibitor value are excluded, "
                "never imputed. That exclusion is unlikely to be random, so the "
                "positive rate describes the reported subset of this dataset, not "
                "population incidence."
            ),
        }


@dataclass
class MergeReport:
    """What the join on ``mut_id`` produced."""

    n_mmc2_rows: int
    n_mmc2_f8_rows: int
    n_mmc3_rows: int
    n_mmc3_labelled: int
    n_merged_rows: int
    n_unique_mut_id: int
    n_duplicate_rows: int
    n_unmatched_clinical: int
    n_conflicting_label_groups: int
    records_per_mutation_max: int
    target_distribution: dict[str, int] = field(default_factory=dict)
    missing_by_feature: dict[str, float] = field(default_factory=dict)
    case_collapses: dict[str, dict[str, str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_mmc2_rows": self.n_mmc2_rows,
            "n_mmc2_f8_rows": self.n_mmc2_f8_rows,
            "n_mmc3_rows": self.n_mmc3_rows,
            "n_mmc3_labelled": self.n_mmc3_labelled,
            "n_merged_rows": self.n_merged_rows,
            "n_unique_mut_id": self.n_unique_mut_id,
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_unmatched_clinical": self.n_unmatched_clinical,
            "n_conflicting_label_groups": self.n_conflicting_label_groups,
            "records_per_mutation_max": self.records_per_mutation_max,
            "target_distribution": self.target_distribution,
            "case_collapses": self.case_collapses,
        }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _clean_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\r\n", " ").replace("\n", " ")).strip()


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. Set MMC2_PATH/MMC3_PATH or DATA_DIR, "
            f"or place the file in {data_dir()}."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [_clean_column_name(c) for c in df.columns]
    return df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore"
    )


def load_mmc2(path: str | Path | None = None) -> pd.DataFrame:
    """Read the MMC2 mutation table. The file on disk is never modified."""
    return _read_csv(path or mmc2_path(), "MMC2")


def load_mmc3(path: str | Path | None = None) -> pd.DataFrame:
    """Read the MMC3 clinical-record table. The file on disk is never modified."""
    return _read_csv(path or mmc3_path(), "MMC3")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _source_report(
    df: pd.DataFrame, name: str, path: str | Path, required: Iterable[str]
) -> SourceReport:
    missing = sorted(set(required) - set(df.columns))
    has_group = GROUP_COLUMN in df.columns
    return SourceReport(
        name=name,
        path=str(path),
        n_rows=len(df),
        n_columns=df.shape[1],
        n_unique_mut_id=int(df[GROUP_COLUMN].nunique()) if has_group else 0,
        n_duplicate_rows=int(df.duplicated().sum()),
        n_duplicate_mut_id=int(df[GROUP_COLUMN].duplicated().sum()) if has_group else 0,
        missing_columns=missing,
        missing_values_total=int(df.isna().sum().sum()),
    )


def validate_mmc2(df: pd.DataFrame, path: str | Path = "<frame>") -> SourceReport:
    return _source_report(df, "MMC2", path, {GROUP_COLUMN, GENE_COLUMN})


def validate_mmc3(df: pd.DataFrame, path: str | Path = "<frame>") -> SourceReport:
    return _source_report(df, "MMC3", path, {GROUP_COLUMN, LABEL_COLUMN})


# --------------------------------------------------------------------------
# Normalisation and labelling
# --------------------------------------------------------------------------


def _norm_text(value: Any) -> Any:
    """Strip and null-normalise a single cell, leaving numbers untouched.

    Two dtype traps are handled here. pandas' ``StringDtype`` NA raises
    "boolean value of NA is ambiguous" inside ``SimpleImputer``; and Python
    ``None`` is not detected as missing by ``SimpleImputer`` on object arrays
    (its mask is ``X != X``, and ``None != None`` is False), so nulls would
    become a literal ``"None"`` category.
    """
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        return np.nan if text == "" else text
    return value


def normalise_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Whitespace-strip the given object columns in place on a copy.

    No rows are added or removed and no label is altered. The raw files record
    both ``"Point"`` and ``"Point "``, ``"Exon"`` and ``"Exon "``, ``"Severe"``
    and ``"Severe "``; those are the same value typed twice.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = pd.Series(
            [_norm_text(v) for v in out[col]], index=out.index, dtype=object
        )
    return out


def collapse_case_variants(
    df: pd.DataFrame, columns: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Fold spellings that differ only in letter case onto the dominant one.

    The files record ``Missense`` 2,674 times and ``missense`` and ``MIssense``
    once each; ``Severe`` 2,700 times and ``severe`` twice. Those are the same
    value typed twice, and leaving them apart splits a category and starves the
    minority spelling of the frequency it needs to survive encoding.

    Unlike a blanket ``.str.lower()``, the winner is always a spelling that
    actually occurs, and every collapse is returned so it can be reported rather
    than happening silently. Folding is applied only to the columns named in the
    candidate lists, none of which carries a meaning that depends on case; it
    must never be applied blindly, because in other F8 annotations lowercase
    ``a1``/``a2``/``a3`` are the acidic regions and are biologically distinct
    from the ``A1``/``A2``/``A3`` domains. See docs/ML.md.
    """
    out = df.copy()
    collapses: dict[str, dict[str, str]] = {}
    for col in columns:
        if col not in out.columns or pd.api.types.is_numeric_dtype(out[col]):
            continue
        counts = out[col].dropna().astype(str).value_counts()
        by_fold: dict[str, list[str]] = {}
        for value in counts.index:
            by_fold.setdefault(value.casefold(), []).append(value)
        mapping = {
            variant: spellings[0]  # value_counts is ordered, so [0] is dominant
            for spellings in by_fold.values()
            if len(spellings) > 1
            for variant in spellings[1:]
        }
        if mapping:
            collapses[col] = mapping
            out[col] = out[col].map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    return out, collapses


# --------------------------------------------------------------------------
# Measurement parsing
# --------------------------------------------------------------------------

#: Tokens the source files use for "no value". Compared case-insensitively.
NULL_TOKENS = {
    "",
    "-",
    "?",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "not",
    "nd",
    "not done",
    "not reported",
    "not tested",
    "unknown",
}

#: Unit suffixes that scale a value. ``n_bp`` records large deletions as "7kb".
UNIT_MULTIPLIERS: dict[str, float] = {"kb": 1_000.0, "mb": 1_000_000.0, "bp": 1.0}

_NUMBER = r"[+-]?\d+(?:\.\d+)?"
_RANGE_RE = re.compile(
    rf"^(?P<low>{_NUMBER})\s*(?:--|–|—|-|to)\s*(?P<high>\d+(?:\.\d+)?)$", re.I
)
_TRAILING_NOTE_RE = re.compile(r"\s*\([^)]*\)\s*$")
_UNIT_RE = re.compile(rf"^(?P<value>{_NUMBER})\s*(?P<unit>kb|mb|bp)$", re.I)

#: Censoring codes returned by :func:`parse_measurement`.
OBSERVED, LEFT_CENSORED, RIGHT_CENSORED, RANGE = "observed", "left", "right", "range"


def parse_measurement(value: Any) -> tuple[float, str] | tuple[float, None]:
    """Parse one free-text measurement into ``(value, censoring)``.

    The clinical columns are not numeric in the source file. ``clotting`` — the
    residual factor VIII activity, and the most informative clinical field there
    is — writes 4,237 of its 6,492 non-null entries as a bound or a range::

        "<1"        -> (1.0,  "left")    activity below the assay's detection limit
        ">5"        -> (5.0,  "right")   activity above the reporting ceiling
        "1 to 5"    -> (3.0,  "range")   reported as an interval
        "1--5"      -> (3.0,  "range")   the same interval, typed differently
        "3"         -> (3.0,  "observed")
        "13 (post)" -> (13.0, "observed") trailing annotation dropped
        "7kb"       -> (7000.0, "observed")  n_bp records deletions in kilobases
        "Not reported" -> (nan, None)

    A bound is kept *as* the bound rather than halved, and the direction is
    carried by the returned censoring code, which becomes its own feature. That
    keeps the arithmetic honest: "<1" is not evidence of 0.5, it is evidence of
    "at most 1", and the aggregate features record how much of a mutation's
    evidence is of that kind.

    Returns ``(nan, None)`` for anything unparseable, never a guess.
    """
    if value is None or value is pd.NA:
        return float("nan"), None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return (float("nan"), None) if np.isnan(v) else (v, OBSERVED)

    text = str(value).strip()
    if text.casefold() in NULL_TOKENS:
        return float("nan"), None

    text = _TRAILING_NOTE_RE.sub("", text).strip()
    if not text or text.casefold() in NULL_TOKENS:
        return float("nan"), None

    censoring = OBSERVED
    if text[0] in "<≤":
        censoring, text = LEFT_CENSORED, text[1:].strip()
    elif text[0] in ">≥":
        censoring, text = RIGHT_CENSORED, text[1:].strip()

    unit = _UNIT_RE.match(text)
    if unit:
        scale = UNIT_MULTIPLIERS[unit.group("unit").casefold()]
        return float(unit.group("value")) * scale, censoring

    span = _RANGE_RE.match(text)
    if span:
        low, high = float(span.group("low")), float(span.group("high"))
        # A leading minus belongs to the number, not to the separator: nuc_numb
        # records promoter positions as "-860", which is one value, not a range.
        if high >= low:
            return (low + high) / 2.0, RANGE if censoring == OBSERVED else censoring

    try:
        return float(text), censoring
    except ValueError:
        return float("nan"), None


def parse_measurement_series(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Vectorised :func:`parse_measurement` -> ``(values, censoring)`` series."""
    parsed = [parse_measurement(v) for v in values]
    numbers = pd.Series([p[0] for p in parsed], index=values.index, dtype="float64")
    codes = pd.Series([p[1] for p in parsed], index=values.index, dtype=object)
    return numbers, codes


def severity_score(value: Any) -> float:
    """Map a ``cli_phe`` phenotype onto the 1-3 ordinal scale, or NaN.

    "Unclassified" and "Not reported" are genuinely unranked and stay NaN rather
    than being pushed to the middle of the scale.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return SEVERITY_SCORES.get(str(value).strip().casefold(), float("nan"))


def encode_target(mmc3: pd.DataFrame) -> tuple[pd.DataFrame, LabelReport]:
    """Keep MMC3 records with an explicit Yes/No and add ``target`` (1/0).

    Everything else — "Not reported", "Not", blanks, and the handful of rows
    where a severity was typed into the inhibitor field — is dropped and counted
    in the report. Nothing is imputed.
    """
    if LABEL_COLUMN not in mmc3.columns:
        raise ValueError(f"MMC3 must contain an '{LABEL_COLUMN}' column.")

    normalised = (
        mmc3[LABEL_COLUMN].fillna("(missing)").astype(str).str.strip().str.lower()
    ).replace("", "(missing)")

    is_pos = normalised.isin(POSITIVE_LABELS)
    keep = is_pos | normalised.isin(NEGATIVE_LABELS)

    excluded = normalised[~keep].value_counts(dropna=False).to_dict()

    labelled = mmc3.loc[keep].copy()
    labelled[TARGET_COLUMN] = is_pos.loc[keep].astype(int).to_numpy()

    report = LabelReport(
        n_total=len(mmc3),
        n_labelled=int(keep.sum()),
        n_positive=int(is_pos.sum()),
        n_excluded_unlabelled=int((~keep).sum()),
        excluded_values={str(k): int(v) for k, v in excluded.items()},
    )
    return labelled.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Mutation-level aggregation
# --------------------------------------------------------------------------

#: Number of clinical records behind a mutation's aggregates.
RECORD_COUNT_COLUMN = "n_clinical_records"

#: Suffixes appended to a source column when it is aggregated. Longest first,
#: so ``_censored_rate`` is stripped before ``_rate`` could ever match.
AGGREGATE_SUFFIXES: tuple[str, ...] = (
    "_censored_rate",
    "_median",
    "_mean",
    "_min",
    "_max",
    "_mode",
)


def aggregate_source_column(name: str) -> str:
    """``clotting_mean`` -> ``clotting``; ``severity_prop_severe`` -> ``cli_phe``.

    Explanations name the field the caller filled in, not the derived statistic,
    so SHAP and LIME talk about "clotting activity" rather than
    "num__clotting_median".
    """
    if name == RECORD_COUNT_COLUMN:
        return RECORD_COUNT_COLUMN
    if name.startswith("severity_") or name == f"{SEVERITY_COLUMN}_mode":
        return SEVERITY_COLUMN
    for suffix in AGGREGATE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclass
class GroupLabelReport:
    """How the record-level labels resolved onto mutations."""

    n_records: int
    n_mutations: int
    n_positive: int
    n_negative: int
    n_conflicting: int
    conflicting_mut_ids: list[str] = field(default_factory=list)
    records_per_mutation_mean: float = 0.0
    records_per_mutation_max: int = 0

    @property
    def n_modelled(self) -> int:
        return self.n_positive + self.n_negative

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_records_in": self.n_records,
            "n_mutations": self.n_mutations,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_conflicting_excluded": self.n_conflicting,
            "n_final_modelling_population": self.n_modelled,
            "positive_rate": (
                round(self.n_positive / self.n_modelled, 6) if self.n_modelled else 0.0
            ),
            "records_per_mutation_mean": round(self.records_per_mutation_mean, 4),
            "records_per_mutation_max": self.records_per_mutation_max,
            "conflicting_mut_ids_sample": self.conflicting_mut_ids[:25],
            "rule": (
                "A mutation is positive when every labelled record reports an "
                "inhibitor, negative when none does, and conflicting when its "
                "records disagree. Conflicting mutations are excluded, not "
                "resolved by majority vote."
            ),
        }


def _mode(values: pd.Series) -> Any:
    """Most frequent non-null value, ties broken by first appearance."""
    clean = values.dropna()
    if clean.empty:
        return np.nan
    counts = clean.astype(str).value_counts()
    return counts.index[0]


def aggregate_to_mutations(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, GroupLabelReport]:
    """Collapse the merged record table to one row per ``mut_id``.

    Genomic columns are constant within a mutation (MMC2 contributes one row per
    ``mut_id``), so they are carried over unchanged - after the measurement-like
    ones are parsed out of text.

    Clinical columns are the reason this function exists. Each measurement
    becomes five features - mean, median, min, max and the share of readings
    that were censored or given as a range - and the severity phenotype becomes
    an ordinal score plus the proportion of records in each severity bucket, its
    distinct-value count and its dominant category. A mutation with one clinical
    record gets the same features as one with 104; only the aggregates differ,
    which is what lets inference build an identical row from a single record.

    Mutations whose records disagree about the outcome are dropped here and
    reported, so nothing downstream has to remember to do it.
    """
    if merged.empty:
        raise ValueError("Nothing to aggregate: the merged frame is empty.")

    grouped = merged.groupby(GROUP_COLUMN, sort=True)
    sizes = grouped.size()
    out: dict[str, pd.Series] = {}

    # -- label resolution ------------------------------------------------
    target_min = grouped[TARGET_COLUMN].min()
    target_max = grouped[TARGET_COLUMN].max()
    conflicting = target_min != target_max
    resolved = target_max.where(~conflicting)

    # -- genomic block: one value per mutation ---------------------------
    for col in GENOMIC_CANDIDATES:
        if col not in merged.columns or col in EXCLUDED_COLUMNS:
            continue
        if col in GENOMIC_MEASUREMENTS:
            numbers, _ = parse_measurement_series(merged[col])
            out[col] = numbers.groupby(merged[GROUP_COLUMN]).first()
        else:
            out[col] = grouped[col].first()

    # -- clinical measurements: five statistics each ---------------------
    for col in CLINICAL_MEASUREMENTS:
        if col not in merged.columns or col in EXCLUDED_COLUMNS:
            continue
        numbers, codes = parse_measurement_series(merged[col])
        by_group = numbers.groupby(merged[GROUP_COLUMN])
        for stat in MEASUREMENT_AGGREGATES:
            out[f"{col}_{stat}"] = getattr(by_group, stat)()
        censored = codes.isin({LEFT_CENSORED, RIGHT_CENSORED, RANGE})
        observed = codes.notna()
        n_obs = observed.groupby(merged[GROUP_COLUMN]).sum()
        out[f"{col}_censored_rate"] = (
            censored.groupby(merged[GROUP_COLUMN]).sum() / n_obs.replace(0, np.nan)
        )

    # -- severity: ordinal score, bucket shares, spread, dominant value ---
    if SEVERITY_COLUMN in merged.columns and SEVERITY_COLUMN not in EXCLUDED_COLUMNS:
        raw = merged[SEVERITY_COLUMN]
        scores = pd.Series(
            [severity_score(v) for v in raw], index=merged.index, dtype="float64"
        )
        by_group = scores.groupby(merged[GROUP_COLUMN])
        out["severity_score_mean"] = by_group.mean()
        out["severity_score_min"] = by_group.min()
        out["severity_score_max"] = by_group.max()

        folded = raw.astype(str).str.strip().str.casefold()
        ranked = scores.notna()
        n_ranked = ranked.groupby(merged[GROUP_COLUMN]).sum().replace(0, np.nan)
        for bucket in SEVERITY_BUCKETS:
            hits = folded.eq(bucket) & ranked
            out[f"severity_prop_{bucket}"] = (
                hits.groupby(merged[GROUP_COLUMN]).sum() / n_ranked
            )
        out["severity_n_distinct"] = (
            raw.where(ranked).groupby(merged[GROUP_COLUMN]).nunique().astype("float64")
        )
        out[f"{SEVERITY_COLUMN}_mode"] = grouped[SEVERITY_COLUMN].agg(_mode)

    out[RECORD_COUNT_COLUMN] = sizes.astype("float64")

    table = pd.DataFrame(out)
    table.insert(0, GROUP_COLUMN, table.index)
    table[TARGET_COLUMN] = resolved
    table = table.reset_index(drop=True)

    # Re-normalise after grouping. ``SeriesGroupBy.first`` re-materialises an
    # all-null object group as Python ``None``, and SimpleImputer's missing mask
    # is ``X != X`` -- which ``None`` does not satisfy. Without this, a genomic
    # categorical's missing values would be fitted as a literal "None" level
    # while the clinical block got the intended MISSING_CATEGORY, so the two
    # halves of the same feature space would disagree about what "not reported"
    # looks like. See ``_norm_text`` for the same trap at the record level.
    table = normalise_frame(table, table.columns)

    report = GroupLabelReport(
        n_records=len(merged),
        n_mutations=int(len(table)),
        n_positive=int((resolved == 1).sum()),
        n_negative=int((resolved == 0).sum()),
        n_conflicting=int(conflicting.sum()),
        conflicting_mut_ids=[str(v) for v in conflicting[conflicting].index],
        records_per_mutation_mean=float(sizes.mean()),
        records_per_mutation_max=int(sizes.max()),
    )

    modelled = table.loc[table[TARGET_COLUMN].notna()].reset_index(drop=True)
    modelled[TARGET_COLUMN] = modelled[TARGET_COLUMN].astype(int)
    return modelled, report


def filter_gene(mmc2: pd.DataFrame, gene: str = GENE_OF_INTEREST) -> pd.DataFrame:
    """Restrict MMC2 to one gene. MMC2 carries six F9 rows alongside F8."""
    if GENE_COLUMN not in mmc2.columns:
        return mmc2.copy()
    mask = mmc2[GENE_COLUMN].astype(str).str.strip().str.upper().eq(gene.upper())
    return mmc2.loc[mask].copy()


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def merge_sources(
    mmc2: pd.DataFrame, mmc3: pd.DataFrame
) -> tuple[pd.DataFrame, LabelReport, MergeReport]:
    """Join labelled MMC3 clinical records to their MMC2 mutation description.

    The join is deliberately asymmetric:

    * MMC2 is reduced to **one row per mutation** first. It already holds one
      row per ``mut_id``, but the ``drop_duplicates`` is kept so a future file
      with repeated mutations cannot silently multiply clinical records.
    * MMC3 keeps **every** clinical record. Rows are not deduplicated by
      ``mut_id``: several patients genuinely carry the same mutation, and two
      records of one mutation may disagree about the inhibitor outcome. Those
      groups are counted and reported, not removed — and the grouped split keeps
      them together.

    An inner join drops clinical records whose mutation is not in the F8 MMC2
    table; that count is reported.
    """
    if GROUP_COLUMN not in mmc2.columns or GROUP_COLUMN not in mmc3.columns:
        raise ValueError(f"Both MMC2 and MMC3 must contain '{GROUP_COLUMN}'.")

    n_mmc2_rows = len(mmc2)
    n_mmc3_rows = len(mmc3)

    mmc2_f8 = filter_gene(mmc2)
    labelled, labels = encode_target(mmc3)

    mmc2_one = (
        mmc2_f8.sort_values(GROUP_COLUMN)
        .drop_duplicates(GROUP_COLUMN, keep="first")
        .copy()
    )

    merged = labelled.merge(
        mmc2_one, on=GROUP_COLUMN, how="inner", suffixes=("_clinical", "_genomic")
    )
    if merged.empty:
        raise ValueError(
            "No records matched between MMC3 and the F8 rows of MMC2 on mut_id."
        )

    merged = merged.dropna(subset=[GROUP_COLUMN, TARGET_COLUMN]).reset_index(drop=True)
    merged = normalise_frame(merged, merged.columns)
    merged, case_collapses = collapse_case_variants(
        merged, [*GENOMIC_CANDIDATES, *CLINICAL_CANDIDATES]
    )

    per_group = merged.groupby(GROUP_COLUMN)[TARGET_COLUMN]
    conflicting = int((per_group.nunique() > 1).sum())

    counts = merged[TARGET_COLUMN].value_counts().to_dict()
    report = MergeReport(
        n_mmc2_rows=n_mmc2_rows,
        n_mmc2_f8_rows=len(mmc2_f8),
        n_mmc3_rows=n_mmc3_rows,
        n_mmc3_labelled=labels.n_labelled,
        n_merged_rows=len(merged),
        n_unique_mut_id=int(merged[GROUP_COLUMN].nunique()),
        n_duplicate_rows=int(merged.duplicated().sum()),
        n_unmatched_clinical=int(labels.n_labelled - len(merged)),
        n_conflicting_label_groups=conflicting,
        records_per_mutation_max=int(merged[GROUP_COLUMN].value_counts().max()),
        target_distribution={str(int(k)): int(v) for k, v in counts.items()},
        case_collapses=case_collapses,
    )
    return merged, labels, report


def load_merged(
    mmc2_file: str | Path | None = None, mmc3_file: str | Path | None = None
) -> tuple[pd.DataFrame, LabelReport, MergeReport, list[SourceReport]]:
    """Read path up to the record-level join: load -> validate -> label -> merge.

    The result still has one row per *clinical record*. Modelling uses
    :func:`load_mutation_table`, which takes the next step and collapses it to
    one row per mutation.
    """
    p2 = Path(mmc2_file or mmc2_path())
    p3 = Path(mmc3_file or mmc3_path())
    mmc2 = load_mmc2(p2)
    mmc3 = load_mmc3(p3)

    sources = [validate_mmc2(mmc2, p2), validate_mmc3(mmc3, p3)]
    for report in sources:
        report.raise_if_invalid()

    merged, labels, merge_report = merge_sources(mmc2, mmc3)
    merge_report.missing_by_feature = {
        col: round(float(merged[col].isna().mean()), 6)
        for col in merged.columns
        if col in set(GENOMIC_CANDIDATES) | set(CLINICAL_CANDIDATES)
    }
    return merged, labels, merge_report, sources


@dataclass
class DatasetBundle:
    """Everything one read of MMC2 + MMC3 produced, with its audit trail."""

    mutations: pd.DataFrame
    records: pd.DataFrame
    labels: LabelReport
    merge: MergeReport
    groups: GroupLabelReport
    sources: list[SourceReport]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": [s.as_dict() for s in self.sources],
            "labels": self.labels.as_dict(),
            "merge": self.merge.as_dict(),
            "mutation_level": self.groups.as_dict(),
            "population": self.population(),
        }

    def population(self) -> dict[str, Any]:
        """The mutation census, from every F8 mutation down to what is modelled.

        Reported as one block because each number only means something next to
        the others: a 19.7% positive rate is a rate among *labelled* mutations,
        and most F8 mutations in MMC2 have no labelled clinical record at all.
        """
        unlabelled = max(self.merge.n_mmc2_f8_rows - self.groups.n_mutations, 0)
        return {
            "n_mmc2_rows": self.merge.n_mmc2_rows,
            "n_mmc2_f8_mutations": self.merge.n_mmc2_f8_rows,
            "n_mmc3_records": self.merge.n_mmc3_rows,
            "n_mmc3_records_labelled": self.merge.n_mmc3_labelled,
            "n_merged_records": self.merge.n_merged_rows,
            "n_mutations_with_a_label": self.groups.n_mutations,
            "n_mutations_unknown": unlabelled,
            "n_positive": self.groups.n_positive,
            "n_negative": self.groups.n_negative,
            "n_conflicting_excluded": self.groups.n_conflicting,
            "n_final_modelling_population": self.groups.n_modelled,
            "positive_rate": (
                round(self.groups.n_positive / self.groups.n_modelled, 6)
                if self.groups.n_modelled
                else 0.0
            ),
        }


def load_mutation_table(
    mmc2_file: str | Path | None = None, mmc3_file: str | Path | None = None
) -> DatasetBundle:
    """The full read path used for training: fusion down to one row per mutation.

        MMC2 ─┐
              ├─ join on mut_id ─ aggregate per mut_id ─ drop conflicting ─▶ mutations
        MMC3 ─┘

    Returns both the mutation table that is modelled and the record table it
    came from, so a caller can report on either without re-reading the files.
    """
    records, labels, merge_report, sources = load_merged(mmc2_file, mmc3_file)
    mutations, groups = aggregate_to_mutations(records)
    return DatasetBundle(
        mutations=mutations,
        records=records,
        labels=labels,
        merge=merge_report,
        groups=groups,
        sources=sources,
    )


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """The exact columns one model consumes, and how each is treated.

    Serialised into artifact metadata, so inference reconstructs the same
    column list and ordering the model was fitted on without re-reading the
    dataset.
    """

    name: str
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    required: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    #: The *raw* MMC2/MMC3 fields a caller supplies, before aggregation. The
    #: model consumes ``columns`` (mean/median/min/max/... of these); the API
    #: asks for these. ``build_input_row`` is the only bridge between the two.
    inputs: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        """Model input order: categorical block, then numeric block."""
        return [*self.categorical, *self.numeric]

    def source_column_for(self, encoded_name: str) -> str:
        """``num__clotting_median`` -> ``clotting``; ``cat__mut_type_Point`` -> ``mut_type``.

        Two layers come off: the encoder's prefix and one-hot suffix, then the
        aggregation suffix. An explanation therefore names the field the caller
        filled in - "FVIII clotting activity" - rather than the derived
        statistic the model happened to consume. Longest match first, because
        column names share prefixes (``clotting_min`` and ``clotting_mean``).
        """
        name = re.sub(r"^(cat|num)__", "", encoded_name)
        if name.startswith("missingindicator_"):
            name = name[len("missingindicator_") :]
        for col in sorted(self.columns, key=len, reverse=True):
            if name == col or name.startswith(f"{col}_"):
                return aggregate_source_column(col)
        return aggregate_source_column(name)

    def category_value_for(self, encoded_name: str) -> str | None:
        """The category a one-hot column stands for, or None for a numeric one."""
        name = re.sub(r"^(cat|num)__", "", encoded_name)
        for col in sorted(self.categorical, key=len, reverse=True):
            if name.startswith(f"{col}_"):
                return name[len(col) + 1 :]
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.name,
            "categorical": list(self.categorical),
            "numeric": list(self.numeric),
            "required": list(self.required),
            "dropped": list(self.dropped),
            "inputs": list(self.inputs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSpec":
        categorical = tuple(payload.get("categorical", ()))
        numeric = tuple(payload.get("numeric", ()))
        return cls(
            name=str(payload.get("feature_set", "merged")),
            categorical=categorical,
            numeric=numeric,
            required=tuple(payload.get("required", ())),
            dropped=tuple(payload.get("dropped", ())),
            inputs=tuple(
                payload.get("inputs")
                or dict.fromkeys(
                    aggregate_source_column(c) for c in (*categorical, *numeric)
                )
            ),
        )


def _usable(table: pd.DataFrame, candidates: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split candidates into (usable, dropped).

    A candidate is dropped when the table does not carry it, when it is on the
    exclusion list, or when it holds a single value (including all-null) and so
    carries no information.
    """
    usable, dropped = [], []
    for col in candidates:
        if col in EXCLUDED_COLUMNS:
            dropped.append(col)
        elif col not in table.columns:
            dropped.append(col)
        elif table[col].nunique(dropna=False) <= 1:
            dropped.append(col)
        else:
            usable.append(col)
    return usable, dropped


def clinical_feature_columns(table: pd.DataFrame) -> list[str]:
    """Aggregate columns in ``table`` that came from the MMC3 clinical block."""
    clinical = set(CLINICAL_CANDIDATES)
    return [
        col
        for col in table.columns
        if col not in (GROUP_COLUMN, TARGET_COLUMN)
        and aggregate_source_column(col) in clinical
    ]


def identifier_like_columns(
    table: pd.DataFrame, columns: Iterable[str]
) -> dict[str, float]:
    """Categorical columns distinct enough to be identifying, and their ratios.

    A column with a distinct value for most rows names the row rather than
    describing it, and a model given one can score well by memorising training
    mutations. Anything above :data:`MAX_CATEGORY_UNIQUENESS_RATIO` is reported
    here; :func:`build_feature_specs` refuses to emit it and the test suite
    asserts the resolved feature sets are clean.
    """
    n = len(table)
    if not n:
        return {}
    flagged: dict[str, float] = {}
    for col in columns:
        if col not in table.columns or pd.api.types.is_numeric_dtype(table[col]):
            continue
        ratio = table[col].nunique(dropna=True) / n
        if ratio > MAX_CATEGORY_UNIQUENESS_RATIO:
            flagged[col] = round(float(ratio), 4)
    return flagged


def build_feature_specs(
    table: pd.DataFrame, required_reference: pd.DataFrame | None = None
) -> dict[str, FeatureSpec]:
    """Resolve the genomic / clinical / merged feature sets from real columns.

    ``table`` is the **mutation-level** table from :func:`aggregate_to_mutations`.
    Nothing is assumed to exist: a candidate the files do not carry, or one that
    survives aggregation with a single value, is dropped and recorded in
    ``FeatureSpec.dropped``.

    A categorical column that is really an identifier is refused outright rather
    than dropped quietly, because reintroducing one is the single easiest way to
    make this model look far better than it is.

    ``required_reference`` - normally the training split - decides which columns
    the API marks required: those missing in at most
    ``REQUIRED_MAX_MISSING_RATE`` of its rows.
    """
    genomic, genomic_dropped = _usable(table, GENOMIC_CANDIDATES)
    clinical, clinical_dropped = _usable(table, clinical_feature_columns(table))

    offenders = identifier_like_columns(table, [*genomic, *clinical])
    if offenders:
        raise ValueError(
            "Identifier-like columns cannot be features: "
            + ", ".join(f"{c} ({r:.0%} distinct)" for c, r in offenders.items())
            + f". The limit is {MAX_CATEGORY_UNIQUENESS_RATIO:.0%}; add the column "
            "to EXCLUDED_COLUMNS with a reason."
        )

    reference = table if required_reference is None else required_reference

    def spec(name: str, columns: list[str], dropped: list[str]) -> FeatureSpec:
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(table[c])]
        categorical = [c for c in columns if c not in numeric]
        model_columns = [*categorical, *numeric]
        inputs = tuple(dict.fromkeys(aggregate_source_column(c) for c in model_columns))
        # Required-ness is a property of the raw field the caller fills in, but
        # it is measured on the aggregate that field produces: "clotting" is
        # required if clotting_mean is present in nearly every training row.
        representative = {}
        for col in model_columns:
            representative.setdefault(aggregate_source_column(col), col)
        required = tuple(
            field_name
            for field_name in inputs
            if representative[field_name] in reference.columns
            and float(reference[representative[field_name]].isna().mean())
            <= REQUIRED_MAX_MISSING_RATE
        )
        return FeatureSpec(
            name=name,
            categorical=tuple(categorical),
            numeric=tuple(numeric),
            required=required,
            dropped=tuple(dropped),
            inputs=inputs,
        )

    merged_columns = list(dict.fromkeys([*genomic, *clinical]))
    return {
        "genomic": spec("genomic", genomic, genomic_dropped),
        "clinical": spec("clinical", clinical, clinical_dropped),
        "merged": spec(
            "merged",
            merged_columns,
            list(dict.fromkeys([*genomic_dropped, *clinical_dropped])),
        ),
    }


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """An unfitted transformer for one feature set.

    Must be fitted on the TRAINING split only — fitting on everything leaks the
    test split's medians and category vocabulary into training.

    Categorical: explicit ``Unknown`` fill, then one-hot with a minimum
    frequency so near-identifier columns collapse into an ``infrequent`` bucket
    instead of thousands of singleton columns. Unseen values at inference map to
    that same bucket rather than to an all-zeros row, so an out-of-vocabulary
    value is never indistinguishable from a genuine absence — and
    ``PredictionService`` validates against the fitted vocabulary first anyway,
    so the caller gets a named error instead of a bucketed guess.

    Numeric: median imputation with a missing indicator, then standardisation.
    """
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
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

    transformers = []
    if spec.categorical:
        transformers.append(("cat", categorical, list(spec.categorical)))
    if spec.numeric:
        transformers.append(("num", numeric, list(spec.numeric)))
    if not transformers:
        raise ValueError(f"Feature set '{spec.name}' resolved to no usable columns.")

    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=True
    )


def encoded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Column names produced by a *fitted* preprocessor, in model input order."""
    return [str(n) for n in preprocessor.get_feature_names_out()]


def fitted_categories(
    preprocessor: ColumnTransformer, spec: FeatureSpec
) -> dict[str, list[str]]:
    """The vocabulary each categorical column was actually fitted on.

    The API exposes this so the UI can only ever offer values the model knows.
    """
    if not spec.categorical:
        return {}
    encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
    return {
        col: [str(v) for v in cats]
        for col, cats in zip(spec.categorical, encoder.categories_)
    }


def frequent_categories(
    preprocessor: ColumnTransformer, spec: FeatureSpec
) -> dict[str, list[str]]:
    """Only the categories that received their own encoded column.

    ``fitted_categories`` reports everything the encoder saw, including the
    values it folded into the "infrequent" bucket. Those folded values are not
    separately representable by the model, so the UI offers this shorter list
    instead of a dropdown with 1,610 entries.
    """
    if not spec.categorical:
        return {}
    encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
    infrequent = getattr(encoder, "infrequent_categories_", None) or [None] * len(
        spec.categorical
    )
    out: dict[str, list[str]] = {}
    for col, cats, folded in zip(spec.categorical, encoder.categories_, infrequent):
        dropped = set() if folded is None else {str(v) for v in folded}
        out[col] = [str(v) for v in cats if str(v) not in dropped]
    return out


def open_vocabulary_columns(
    preprocessor: ColumnTransformer, spec: FeatureSpec
) -> set[str]:
    """Categorical columns that accept a value the model has not seen.

    See ``OPEN_VOCABULARY_THRESHOLD``. Derived from the fitted encoder, so it
    describes the model that is actually being served.
    """
    return {
        col
        for col, cats in fitted_categories(preprocessor, spec).items()
        if len(cats) > OPEN_VOCABULARY_THRESHOLD
    }


def build_input_row(spec: FeatureSpec, record: dict[str, Any]) -> pd.DataFrame:
    """One raw clinical record -> the model's feature row.

    This is the bridge between what a caller types and what the model was
    fitted on, and it deliberately takes the same route training did: the
    record is wrapped in a one-row frame and pushed through
    :func:`aggregate_to_mutations`, so a single record produces exactly the
    aggregates that a mutation with one clinical record produced during
    training (``mean = median = min = max = the value``, ``censored_rate`` 0 or
    1, ``n_clinical_records`` 1).

    Reusing the training function rather than reimplementing the arithmetic is
    the point: there is no second copy of the aggregation to drift out of sync.
    """
    row = dict(record)
    row[GROUP_COLUMN] = row.get(GROUP_COLUMN, "__input__")
    row[TARGET_COLUMN] = 0  # placeholder; dropped below, never read by a model

    frame = pd.DataFrame([row])
    frame = normalise_frame(frame, frame.columns)
    aggregated, _ = aggregate_to_mutations(frame)

    return pd.DataFrame(
        [{col: _first_or_nan(aggregated, col) for col in spec.columns}],
        columns=spec.columns,
    )


def _first_or_nan(frame: pd.DataFrame, column: str) -> Any:
    """The single aggregated value for ``column``, or NaN when it is absent.

    A feature set may name a column the caller left out entirely; the fitted
    imputer handles the gap, so an absent field must arrive as a missing value
    rather than raising.
    """
    if column not in frame.columns or frame.empty:
        return np.nan
    return frame[column].iloc[0]


def frame_for(spec: FeatureSpec, row: dict[str, Any]) -> pd.DataFrame:
    """Deprecated alias for :func:`build_input_row`, kept for callers."""
    return build_input_row(spec, row)
