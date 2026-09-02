"""Dataset construction, target definition and leakage audit.

Every dataset gets its OWN preprocessing pipeline. Nothing learned on one
dataset is applied to another.

Leakage classes used throughout:
  SAFE            usable predictor, available before inhibitor status is known
  POTENTIAL       plausibly predictive but temporally or structurally suspect
  DEFINITE        derived from the target; must never be a predictor
  TARGET          the outcome itself
  POST_OUTCOME    measured after / affected by inhibitor development
  IDENTIFIER      row or entity id
  METADATA        curation bookkeeping (dates, references, comments)
  UNKNOWN         temporal validity not established
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CHAMP_CSV, EXTERNAL_DATA, REPO

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SAFE, POTENTIAL, DEFINITE = "SAFE", "POTENTIAL LEAKAGE", "DEFINITE LEAKAGE"
TARGET, POST, IDENT, META, UNKNOWN = (
    "TARGET", "POST-OUTCOME", "IDENTIFIER", "METADATA",
    "UNKNOWN - TEMPORAL VALIDITY NOT ESTABLISHED",
)


@dataclass
class DatasetSpec:
    name: str
    source_file: str
    sha256: str
    unit: str                      # patient / mutation / clinical record
    unit_rationale: str
    target_name: str
    target_definition: str
    X: pd.DataFrame                # SAFE predictors only
    y: pd.Series
    groups: pd.Series | None       # for grouped CV, when rows share an entity
    numeric: list[str]
    categorical: list[str]
    audit: pd.DataFrame            # per-column classification
    n_raw_rows: int
    n_raw_cols: int
    notes: list[str] = field(default_factory=list)
    valid_for_supervised: bool = True
    rejection_reason: str | None = None

    def summary(self) -> dict[str, Any]:
        pos = int(self.y.sum())
        return {
            "dataset": self.name,
            "source_file": self.source_file,
            "sha256_16": self.sha256[:16],
            "prediction_unit": self.unit,
            "target": self.target_name,
            "raw_rows": self.n_raw_rows,
            "raw_cols": self.n_raw_cols,
            "usable_samples": len(self.y),
            "unique_prediction_units": int(self.groups.nunique()) if self.groups is not None else len(self.y),
            "positive": pos,
            "negative": int(len(self.y) - pos),
            "positive_pct": round(100 * self.y.mean(), 2),
            "features_before": self.n_raw_cols,
            "features_after_audit": self.X.shape[1],
            "numeric_features": len(self.numeric),
            "categorical_features": len(self.categorical),
            "missingness_pct": round(100 * self.X.isna().to_numpy().mean(), 2),
            "grouped_cv": self.groups is not None,
        }


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _audit_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["column", "classification", "reason"])


def _norm(s: pd.Series) -> pd.Series:
    """Trim and collapse whitespace; leave case alone."""
    return s.astype("object").map(
        lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
        else re.sub(r"\s+", " ", str(v)).strip()
    )


# --------------------------------------------------------------------------
# Shared inhibitor-label vocabulary (HADB / mmc2 / mmc3)
# --------------------------------------------------------------------------
# Verified against the raw value counts, not assumed from the column name.
_POS = {"yes"}
_NEG = {"no", "not affected"}
_UNKNOWN = {"not reported", "not", "nan", ""}          # "Not" is a truncated "Not reported"
_INVALID = {"mild", "severe"}                           # severity values in the wrong column


def _map_inhibitor(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Map a raw HADB inhibitor column to {1,0,NaN} and report what was found."""
    raw = _norm(series).str.lower()
    counts = raw.value_counts(dropna=False).to_dict()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[raw.isin(_POS)] = 1.0
    out[raw.isin(_NEG)] = 0.0
    # everything else (incl. _UNKNOWN and _INVALID) stays NaN and is excluded
    return out, {str(k): int(v) for k, v in counts.items()}


# ==========================================================================
# D1 - CHAMP  (project baseline; reuses the production preprocessing module)
# ==========================================================================
def build_champ() -> DatasetSpec:
    from ml.preprocessing import champ as champ_mod   # production module, unmodified

    X, y, labels, _ = champ_mod.load_champ_features(CHAMP_CSV)
    raw = champ_mod.load_champ(CHAMP_CSV)

    rows = [(c, SAFE, "genomic/registry predictor used by the production pipeline")
            for c in champ_mod.FEATURE_COLUMNS]
    rows.append((champ_mod.LABEL_COLUMN, TARGET, "inhibitor outcome"))
    for c, why in champ_mod.EXCLUDED_COLUMNS.items():
        cls = DEFINITE if "LEAKAGE" in why else (IDENT if "identifier" in why else META)
        rows.append((c, cls, why))

    return DatasetSpec(
        name="D1_CHAMP",
        source_file=str(CHAMP_CSV.relative_to(REPO)),
        sha256=_sha(CHAMP_CSV),
        unit="mutation (F8 variant)",
        unit_rationale=(
            "One row per distinct HGVS cDNA variant; 0 duplicated variants among "
            "labelled rows. 'History of Inhibitor' is a registry aggregation over "
            "the reports for that variant, so the unit is the variant, not a patient."
        ),
        target_name=champ_mod.LABEL_COLUMN,
        target_definition=(
            "Yes -> 1, No -> 0. 'Not reported' (1742) and blank (12) excluded, "
            "never imputed. Verified from raw value counts."
        ),
        X=X, y=y, groups=None,
        numeric=list(champ_mod.NUMERIC_FEATURES) + list(champ_mod.BINARY_FEATURES),
        categorical=list(champ_mod.CATEGORICAL_FEATURES),
        audit=_audit_frame(rows),
        n_raw_rows=len(raw), n_raw_cols=raw.shape[1],
        notes=[
            "Reuses ml/preprocessing/champ.py unchanged (the production pipeline).",
            "Domain is deliberately NOT case-folded: a1/a2/a3 are acidic regions, "
            "distinct from the A1/A2/A3 domains.",
            f"{labels.n_excluded_unlabelled} rows excluded for having no reported "
            "inhibitor history; that exclusion is very unlikely to be random.",
        ],
    )


# ==========================================================================
# D2 - ML-Ready  (mutation-level, genomic + aggregated clinical)
# ==========================================================================
_MLREADY_GENOMIC = [
    "mut_type", "mut_effect", "d_id", "location", "e_i_numb", "CpG",
    "ntchange", "aa_first", "aa_last", "codon_first", "codon_last",
]
_MLREADY_NUMERIC = ["aa_numb", "nuc_numb", "locnumb_num"]


def build_mlready() -> DatasetSpec:
    path = EXTERNAL_DATA / "HemophiliaA_ML_Ready_Inhibitor.csv"
    df = pd.read_csv(path, low_memory=False)
    n_raw_rows, n_raw_cols = df.shape

    y = df["inhibitor_target"].astype(int)

    rows: list[tuple[str, str, str]] = [
        ("inhibitor_target", TARGET, "binary inhibitor outcome derived from mmc3 records"),
    ]

    # --- construction-induced leakage (established empirically, see notes) ---
    for c in ["clinical_record_count", "clotting_count", "ratio_count",
              "antigen_count", "act_ant_count", "Count_mut_id"]:
        rows.append((c, DEFINITE,
                     "record-count column. The label rule excludes mutations whose "
                     "records disagree, so count is mechanically anti-correlated with "
                     "the target (clotting_count single-feature AUC 0.393, p=8.5e-21). "
                     "This is label-construction leakage, not biology."))

    # --- FVIII activity/antigen: inhibitors neutralise FVIII, so a measurement
    #     taken after inhibitor development reflects the outcome ---
    for c in df.columns:
        if c.startswith(("clotting_", "ratio_", "antigen_", "act_ant_")) and not c.endswith("_count"):
            rows.append((c, UNKNOWN,
                         "aggregated FVIII activity/antigen. Inhibitors neutralise "
                         "FVIII, so a post-inhibitor assay reflects the outcome. The "
                         "registry does not record assay timing relative to inhibitor "
                         "development."))
    for c in ["uclotting", "uratio", "uantigen", "udiscrep", "utype"]:
        if c in df.columns:
            rows.append((c, UNKNOWN,
                         "mutation-level FVIII activity/assay summary; timing relative "
                         "to inhibitor development not established"))

    # --- identifiers / free text / near-unique strings ---
    for c in ["mut_syn", "aa_syn", "aa_change", "codon_change"]:
        if c in df.columns:
            rows.append((c, IDENT,
                         f"near-unique variant nomenclature string "
                         f"({df[c].nunique()} distinct over {len(df)} rows)"))
    for c in ["g_id", "g_name"]:
        if c in df.columns:
            rows.append((c, META, "constant/near-constant gene identifier"))
    if "n_bp" in df.columns:
        rows.append(("n_bp", SAFE, "size of the change in base pairs"))

    df = df.copy()
    df["locnumb_num"] = pd.to_numeric(
        df["locnumb"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    rows.append(("locnumb", SAFE, "exon/intron location number (parsed to numeric)"))

    for c in _MLREADY_GENOMIC:
        if c in df.columns:
            rows.append((c, SAFE, "genomic variant descriptor available at genotyping"))
    for c in ["aa_numb", "nuc_numb"]:
        rows.append((c, SAFE, "genomic position, available at genotyping"))
    rows.append(("aa_numb_old", META, "superseded numbering, duplicates aa_numb"))

    cats = [c for c in _MLREADY_GENOMIC if c in df.columns] + ["n_bp"]
    nums = [c for c in _MLREADY_NUMERIC if c in df.columns]
    X = df[cats + nums].copy()
    for c in cats:
        X[c] = _norm(X[c])

    return DatasetSpec(
        name="D2_MLReady",
        source_file="HemophiliaA_ML_Ready_Inhibitor.csv",
        sha256=_sha(path),
        unit="mutation (F8 variant)",
        unit_rationale=(
            "3706 rows, one per mut_id. Built by aggregating mmc3 clinical records up "
            "to the mutation. It is NOT patient-level: mmc3's pa_id is missing for "
            "70.9% of records and is not a global patient key."
        ),
        target_name="inhibitor_target",
        target_definition=(
            "1 if every known mmc3 record for the mutation reports an inhibitor, 0 if "
            "every known record reports none. Mutations with no known record (2378) "
            "and with disagreeing records (127) were dropped by the dataset's authors. "
            "Verified: reproduced exactly from mmc3 (agreement 1.000)."
        ),
        X=X, y=y, groups=None,
        numeric=nums, categorical=cats,
        audit=_audit_frame(rows),
        n_raw_rows=n_raw_rows, n_raw_cols=n_raw_cols,
        notes=[
            "Excluding the 127 conflicting mutations makes the label cleaner but "
            "biased: it removes exactly the mutations with mixed clinical evidence.",
            "Record-count columns are excluded as construction leakage; a sensitivity "
            "run quantifies what including them would buy.",
            "FVIII activity aggregates are excluded as UNKNOWN temporal validity.",
        ],
    )


# ==========================================================================
# D3 - MMC2 genomic only  (independent target: the curator's mutation summary)
# ==========================================================================
def build_mmc2_genomic() -> DatasetSpec:
    path = EXTERNAL_DATA / "BVTH_VTH-2024-000215-mmc2.csv"
    df = pd.read_csv(path, low_memory=False)
    n_raw_rows, n_raw_cols = df.shape

    y_all, raw_counts = _map_inhibitor(df["uinhibitor"])
    keep = y_all.notna()

    rows = [("uinhibitor", TARGET,
             f"curator's mutation-level inhibitor summary; raw values {raw_counts}")]
    for c in ["mut_id"]:
        rows.append((c, IDENT, "mutation row id"))
    for c in ["Date added to HADB", "ucomments", "mutations", "g_id", "g_name"]:
        if c in df.columns:
            rows.append((c, META, "curation bookkeeping / free text / constant"))
    for c in ["mut_syn", "aa_syn", "aa_change", "codon_change"]:
        rows.append((c, IDENT, f"near-unique nomenclature string ({df[c].nunique()} distinct)"))
    rows.append(("useverity", POST,
                 "clinical severity summary. Severe disease is both a strong inhibitor "
                 "risk factor and, via FVIII assays, affected by inhibitors. Excluded "
                 "from the genomic-only design to keep this dataset's contrast clean."))
    for c in ["uclotting", "uratio", "uantigen", "udiscrep", "utype"]:
        rows.append((c, UNKNOWN,
                     "FVIII activity/assay summary; timing vs inhibitor development "
                     "not established"))
    rows.append(("Count_mut_id", POTENTIAL,
                 "number of clinical records for the mutation; reporting intensity, "
                 "not biology"))
    rows.append(("aa_numb_old", META, "superseded numbering"))

    df = df.copy()
    df["locnumb_num"] = pd.to_numeric(
        df["locnumb"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    cats = ["mut_type", "mut_effect", "d_id", "location", "e_i_numb", "CpG",
            "ntchange", "aa_first", "aa_last", "codon_first", "codon_last", "n_bp"]
    cats = [c for c in cats if c in df.columns]
    nums = ["aa_numb", "locnumb_num"]
    df["nuc_numb_num"] = pd.to_numeric(
        df["nuc_numb"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    nums.append("nuc_numb_num")
    for c in cats:
        rows.append((c, SAFE, "genomic variant descriptor available at genotyping"))
    for c in nums:
        rows.append((c, SAFE, "genomic position, available at genotyping"))

    X = df.loc[keep, cats + nums].copy()
    for c in cats:
        X[c] = _norm(X[c])
    y = y_all[keep].astype(int)

    return DatasetSpec(
        name="D3_MMC2_Genomic",
        source_file="BVTH_VTH-2024-000215-mmc2.csv",
        sha256=_sha(path),
        unit="mutation (F8 variant)",
        unit_rationale="6211 rows, mut_id unique. Mutation-level by construction.",
        target_name="uinhibitor",
        target_definition=(
            "Yes -> 1 (495), No/Not affected -> 0 (1721). 'Not reported'/'Not' and "
            "blanks excluded. Case and whitespace variants normalised. This is the "
            "curator's own mutation-level summary, an INDEPENDENT label source from "
            "D2's record-derived target."
        ),
        X=X.reset_index(drop=True), y=y.reset_index(drop=True), groups=None,
        numeric=nums, categorical=cats,
        audit=_audit_frame(rows),
        n_raw_rows=n_raw_rows, n_raw_cols=n_raw_cols,
        notes=[
            "Genomic-only by design: this dataset answers 'can the variant alone "
            "predict inhibitor risk?'",
            "useverity excluded as POST-OUTCOME to keep the genomic contrast clean.",
        ],
    )


# ==========================================================================
# D4 - MMC3 clinical records  (record-level; grouped CV mandatory)
# ==========================================================================
def build_mmc3_records() -> DatasetSpec:
    path = EXTERNAL_DATA / "BVTH_VTH-2024-000215-mmc3.csv"
    mmc3 = pd.read_csv(path, low_memory=False)
    mmc2 = pd.read_csv(EXTERNAL_DATA / "BVTH_VTH-2024-000215-mmc2.csv", low_memory=False)
    n_raw_rows, n_raw_cols = mmc3.shape

    y_all, raw_counts = _map_inhibitor(mmc3["Inhibitors"])
    keep = y_all.notna()

    rows = [("Inhibitors", TARGET,
             f"per-record inhibitor status; raw values {raw_counts}"),
            ("all_id", IDENT, "clinical record id"),
            ("mut_id", IDENT, "mutation id - used as the CV grouping key, not a feature"),
            ("pa_id", IDENT,
             "lab-local patient code. 70.9% missing, values include '?' and '1', and "
             "one code carries 322 records. NOT a global patient key."),
            ("reference", META, "source publication"),
            ("ref_id", META, "source publication id"),
            ("pub_lab", META, "reporting laboratory"),
            ("date added", META, "curation date"),
            ("comments", META, "free text"),
            ("pri_comments", META, "free text"),
            ("mutations", META, "curation flag"),
            ("subgroups", META, "only 21 non-null of 10064"),
            ("bleed_tool", META, "only 2 non-null"),
            ("bleed_score", META, "entirely empty"),
            ("assay", UNKNOWN, "assay type; only 77 non-null"),
            ("discrep", UNKNOWN, "only 84 non-null"),
            ]
    for c in ["clotting", "ratio", "antigen", "act/ant", "type"]:
        rows.append((c, UNKNOWN,
                     "FVIII activity/antigen measurement. Inhibitors neutralise FVIII, "
                     "so a post-inhibitor assay reflects the outcome; the registry does "
                     "not record assay timing."))
    rows.append(("cli_phe", POTENTIAL,
                 "clinical phenotype/severity. A genuine pre-treatment risk factor, but "
                 "also partly derived from FVIII assays. Kept as SAFE only in the "
                 "clinical variant of this dataset; see notes."))
    rows.append(("pa_race", SAFE,
                 "reported ancestry, recorded at enrolment. Included because ancestry is "
                 "an established inhibitor risk factor in the literature; it is used "
                 "here descriptively and must not become a clinical decision rule."))

    df = mmc3.copy()
    # genomic context joined from mmc2, which is safe: it describes the variant
    gcols = ["mut_id", "mut_type", "mut_effect", "location", "d_id", "CpG", "e_i_numb"]
    df = df.merge(mmc2[gcols].drop_duplicates("mut_id"), on="mut_id", how="left")
    for c in gcols[1:]:
        rows.append((c, SAFE, "genomic descriptor joined from mmc2 on mut_id"))

    cats = ["cli_phe", "pa_race", "mut_type", "mut_effect", "location", "d_id",
            "CpG", "e_i_numb"]
    cats = [c for c in cats if c in df.columns]
    X = df.loc[keep, cats].copy()
    for c in cats:
        X[c] = _norm(X[c])
    y = y_all[keep].astype(int)
    groups = df.loc[keep, "mut_id"]

    return DatasetSpec(
        name="D4_MMC3_ClinicalRecord",
        source_file="BVTH_VTH-2024-000215-mmc3.csv",
        sha256=_sha(path),
        unit="clinical record (one reported observation)",
        unit_rationale=(
            "10064 rows, all_id unique, mut_id repeats. pa_id cannot serve as a patient "
            "key (70.9% missing, non-unique placeholders), so this is record-level, NOT "
            "patient-level. Rows sharing a mut_id are not independent, so all splits and "
            "CV folds are GROUPED BY mut_id."
        ),
        target_name="Inhibitors",
        target_definition=(
            "Yes -> 1 (836), No -> 0 (4130). 'Not reported'/'Not' (3366) and blanks "
            "(1730) excluded. Two rows holding severity values ('Mild','Severe') in the "
            "inhibitor column are treated as unknown and excluded."
        ),
        X=X.reset_index(drop=True), y=y.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        numeric=[], categorical=cats,
        audit=_audit_frame(rows),
        n_raw_rows=n_raw_rows, n_raw_cols=n_raw_cols,
        notes=[
            "GROUPED splitting by mut_id is mandatory here. A random split would put "
            "records of the same mutation in both train and test and inflate scores.",
            "cli_phe (clinical phenotype) is retained as a predictor: it is the closest "
            "thing this benchmark has to a genuine clinical risk factor. It is partly "
            "assay-derived, so it is flagged and a sensitivity run drops it.",
            "This is the only dataset supporting a clinical (non-genomic-only) design.",
        ],
    )


# ==========================================================================
# D5 - Merged  (REJECTED for the primary benchmark; leakage demonstration only)
# ==========================================================================
def build_merged_leaky() -> DatasetSpec:
    path = EXTERNAL_DATA / "HemophiliaA_Merged_MMC2_MMC3.csv"
    df = pd.read_csv(path, low_memory=False)
    n_raw_rows, n_raw_cols = df.shape

    y_all = pd.Series(np.nan, index=df.index, dtype="float64")
    known = df["inhibitor_total_known"].fillna(0) > 0
    y_all[known] = (df.loc[known, "inhibitor_yes_count"] > 0).astype(float)
    keep = y_all.notna()

    rows = [
        ("inhibitor_positive_rate", DEFINITE, "the target expressed as a rate"),
        ("inhibitor_yes_count", DEFINITE, "count of positive outcomes for this mutation"),
        ("inhibitor_no_count", DEFINITE, "count of negative outcomes for this mutation"),
        ("inhibitor_total_known", DEFINITE, "denominator of the outcome aggregation"),
        ("uinhibitor", DEFINITE, "a second copy of the inhibitor outcome"),
    ]
    leaky = ["inhibitor_yes_count", "inhibitor_no_count",
             "inhibitor_total_known", "inhibitor_positive_rate"]
    X = df.loc[keep, leaky].copy()
    y = y_all[keep].astype(int)

    return DatasetSpec(
        name="D5_Merged_LEAKY",
        source_file="HemophiliaA_Merged_MMC2_MMC3.csv",
        sha256=_sha(path),
        unit="mutation (F8 variant)",
        unit_rationale="6211 rows, mut_id unique.",
        target_name="derived from inhibitor_yes_count > 0",
        target_definition="1 if inhibitor_yes_count > 0 among mutations with any known record.",
        X=X.reset_index(drop=True), y=y.reset_index(drop=True), groups=None,
        numeric=leaky, categorical=[],
        audit=_audit_frame(rows),
        n_raw_rows=n_raw_rows, n_raw_cols=n_raw_cols,
        valid_for_supervised=False,
        rejection_reason=(
            "Carries four outcome-derived aggregate columns (inhibitor_yes_count, "
            "inhibitor_no_count, inhibitor_total_known, inhibitor_positive_rate) plus a "
            "duplicate outcome column (uinhibitor). Any target definable from this file "
            "is a deterministic function of those columns. Excluded from the primary "
            "benchmark and run ONLY to quantify how large a leakage-driven score can look."
        ),
        notes=[
            "Not a valid predictive dataset. Its non-leaky genomic content is exactly "
            "D3_MMC2_Genomic, which is benchmarked properly.",
        ],
    )


def build_all() -> list[DatasetSpec]:
    return [build_champ(), build_mlready(), build_mmc2_genomic(),
            build_mmc3_records(), build_merged_leaky()]
