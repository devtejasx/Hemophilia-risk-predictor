"""MMC2 + MMC3 Hemophilia A dataset: loading, validation, merge, and features.

This is the ONE place the dataset is interpreted. Training, the API and the
tests all import from here, so training-time and inference-time preprocessing
cannot drift apart.

The two supplementary tables of the source publication:

* **MMC2** — one row per *mutation* (``mut_id``). Genomic description of an F8
  (or, for six rows, F9) variant: type, effect, location, codon/amino-acid
  change, HGVS notation.
* **MMC3** — one row per *clinical record*. Several records may report the same
  mutation, so ``mut_id`` repeats. Carries the assay values and the
  ``Inhibitors`` field that is the prediction target.

They are joined on ``mut_id``. Because a mutation can appear in many clinical
records, ``mut_id`` is the **grouping key** for every train/validation/test
split: all records of one mutation stay on one side of every split. Splitting
these rows at random would put clinical records of the *same* mutation into both
training and test, and the identical genomic block would then leak.

``mut_id`` itself is never a feature.

Reference implementation: ``final(1).ipynb`` (cells 3-7). Deviations from it are
listed in ``NOTEBOOK_DEVIATIONS`` below and in docs/ML.md.
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

#: Genomic candidates, from ``final(1).ipynb`` cell 4.
GENOMIC_CANDIDATES: tuple[str, ...] = (
    "mut_type",
    "mut_effect",
    "location",
    "e_i_numb",
    "locnumb",
    "aa_numb_old",
    "aa_numb",
    "codon_change",
    "codon_first",
    "codon_last",
    "n_bp",
    "nuc_numb",
    "ntchange",
    "mut_syn",
    "aa_change",
    "aa_first",
    "aa_last",
    "aa_syn",
    "CpG",
    "utype",
    "mutations",
)

#: Clinical candidates, from ``final(1).ipynb`` cell 4.
CLINICAL_CANDIDATES: tuple[str, ...] = (
    "clotting",
    "discrep",
    "ratio",
    "assay",
    "antigen",
    "act/ant",
    "type",
    "pa_race",
    "bleed_tool",
    "bleed_score",
    "cli_phe",
)

#: Columns that must never become features, with the reason. Kept in code (not
#: only in documentation) so the justification travels with the pipeline. This
#: is the notebook's ``bad`` set, annotated.
EXCLUDED_COLUMNS: dict[str, str] = {
    "mut_id": "grouping key for the split, not a predictor",
    "target": "the encoded label",
    "Inhibitors": "the label itself",
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
    "date added": "reporting artifact",
    "Date added to HADB": "reporting artifact",
    "Count_mut_id": "how many records mention the mutation; a reporting artifact",
}

#: A categorical value seen fewer than this many times in the *training* split is
#: folded into an explicit "infrequent" bucket rather than getting its own
#: column. Several source columns are near-identifiers of the mutation
#: (``mut_syn`` has 2,510 distinct values), and without this the encoded matrix
#: would be mostly singleton columns.
MIN_CATEGORY_FREQUENCY = 3

#: Missing categorical values become this explicit level. See
#: ``NOTEBOOK_DEVIATIONS`` — several clinical columns are 75-99% null, and
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

NOTEBOOK_DEVIATIONS: tuple[str, ...] = (
    "Categorical missing values are imputed with an explicit 'Unknown' level "
    "instead of the notebook's most-frequent strategy. Columns such as "
    "assay (99.5% null) and discrep (99.3% null) are missing because the "
    "measurement was never taken; filling them with the mode invents a "
    "measurement and destroys the only signal they carry.",
    "String values are whitespace-stripped before encoding. The raw files "
    "contain both 'Point' and 'Point ', 'Exon' and 'Exon ', 'Severe' and "
    "'Severe '; the notebook treats those as different categories.",
    "Numeric imputation adds a missing-indicator column, so an imputed median "
    "is distinguishable from an observed one.",
    "Model selection uses StratifiedGroupKFold on mut_id rather than a plain "
    "validation refit, so cross-validation obeys the same grouping constraint "
    "as the outer split.",
    "The served model is fitted on the training split only. The notebook refits "
    "on train+validation after choosing a threshold; keeping the fit on train "
    "alone means the artifact's preprocessor never saw the validation rows that "
    "chose its threshold.",
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
    than happening silently. (The previous CHAMP pipeline could not do this at
    all: there, lowercase ``a1``/``a2``/``a3`` are the acidic regions and are
    biologically distinct from the ``A1``/``A2``/``A3`` domains. No such
    distinction exists in these columns — see docs/ML.md.)
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
    """Full read path: load both files -> validate -> filter -> label -> merge."""
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

    @property
    def columns(self) -> list[str]:
        """Model input order: categorical block, then numeric block."""
        return [*self.categorical, *self.numeric]

    def source_column_for(self, encoded_name: str) -> str:
        """``cat__mut_type_Point`` -> ``mut_type``.

        Used so an explanation names the field the caller filled in, never a
        post-encoding column name. Longest match first, because column names
        share prefixes (``aa_numb`` and ``aa_numb_old``).
        """
        name = re.sub(r"^(cat|num)__", "", encoded_name)
        if name.startswith("missingindicator_"):
            name = name[len("missingindicator_") :]
        for col in sorted(self.columns, key=len, reverse=True):
            if name == col or name.startswith(f"{col}_"):
                return col
        return name

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSpec":
        return cls(
            name=str(payload.get("feature_set", "merged")),
            categorical=tuple(payload.get("categorical", ())),
            numeric=tuple(payload.get("numeric", ())),
            required=tuple(payload.get("required", ())),
            dropped=tuple(payload.get("dropped", ())),
        )


def _usable(merged: pd.DataFrame, candidates: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split candidates into (usable, dropped-with-reason-implied).

    A candidate is dropped when it is not in the merged frame, when it is on the
    exclusion list, or when it holds a single value (including all-null) and so
    carries no information. This is the notebook's ``valid_features`` filter.
    """
    usable, dropped = [], []
    for col in candidates:
        if col in EXCLUDED_COLUMNS:
            dropped.append(col)
        elif col not in merged.columns:
            dropped.append(col)
        elif merged[col].nunique(dropna=False) <= 1:
            dropped.append(col)
        else:
            usable.append(col)
    return usable, dropped


def build_feature_specs(
    merged: pd.DataFrame, required_reference: pd.DataFrame | None = None
) -> dict[str, FeatureSpec]:
    """Resolve the genomic / clinical / merged feature sets from real columns.

    Nothing is assumed to exist. A candidate that the files do not carry, or
    that survives the merge only as a suffixed duplicate (``mutations`` appears
    in both tables and becomes ``mutations_clinical`` / ``mutations_genomic``),
    is dropped and recorded in ``FeatureSpec.dropped``.

    ``required_reference`` — normally the training split — decides which columns
    the API marks required: those missing in at most
    ``REQUIRED_MAX_MISSING_RATE`` of its rows.
    """
    genomic, genomic_dropped = _usable(merged, GENOMIC_CANDIDATES)
    clinical, clinical_dropped = _usable(merged, CLINICAL_CANDIDATES)

    reference = merged if required_reference is None else required_reference

    def spec(name: str, columns: list[str], dropped: list[str]) -> FeatureSpec:
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(merged[c])]
        categorical = [c for c in columns if c not in numeric]
        required = tuple(
            c
            for c in [*categorical, *numeric]
            if c in reference.columns
            and float(reference[c].isna().mean()) <= REQUIRED_MAX_MISSING_RATE
        )
        return FeatureSpec(
            name=name,
            categorical=tuple(categorical),
            numeric=tuple(numeric),
            required=required,
            dropped=tuple(dropped),
        )

    merged_columns = list(dict.fromkeys([*genomic, *clinical]))
    return {
        "genomic": spec("genomic", genomic, genomic_dropped),
        "clinical": spec("clinical", clinical, clinical_dropped),
        "merged": spec(
            "merged", merged_columns, list(dict.fromkeys([*genomic_dropped, *clinical_dropped]))
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


def frame_for(spec: FeatureSpec, row: dict[str, Any]) -> pd.DataFrame:
    """One input dict -> a single-row frame in the model's column order."""
    return pd.DataFrame([{col: row.get(col) for col in spec.columns}], columns=spec.columns)
