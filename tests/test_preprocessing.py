"""CHAMP data integrity and preprocessing correctness.

Several of these encode specific defects found in the Phase 0 audit, so a
regression would fail the suite rather than being rediscovered later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import champ


@pytest.fixture(scope="module")
def raw():
    return champ.load_champ()


@pytest.fixture(scope="module")
def prepared():
    return champ.load_champ_features()


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------


def test_champ_file_loads_with_expected_shape(raw):
    assert len(raw) == 4050
    # The 10 trailing all-empty "Unnamed:" columns are dropped on load.
    assert not [c for c in raw.columns if c.startswith("Unnamed:")]


def test_validation_passes_on_the_shipped_dataset(raw):
    report = champ.validate_champ(raw)
    assert report.ok
    assert report.missing_columns == []


def test_validation_reports_missing_columns():
    report = champ.validate_champ(pd.DataFrame({"Variant Type": ["Missense"]}))
    assert not report.ok
    assert champ.LABEL_COLUMN in report.missing_columns
    with pytest.raises(ValueError, match="missing required columns"):
        report.raise_if_invalid()


def test_label_counts_match_the_audit(prepared):
    _, y, labels, _ = prepared
    assert labels.n_labelled == 2296
    assert labels.n_positive == 461
    assert y.sum() == 461
    assert labels.positive_rate == pytest.approx(0.2008, abs=1e-4)


def test_every_row_is_accounted_for(prepared):
    """No record may be dropped without appearing in the exclusion report."""
    _, _, labels, _ = prepared
    assert labels.n_labelled + labels.n_excluded_unlabelled == labels.n_total
    assert sum(labels.excluded_values.values()) == labels.n_excluded_unlabelled
    # The 12 blank labels must be named, not silently swallowed.
    assert labels.excluded_values["(missing)"] == 12
    assert labels.excluded_values["not reported"] == 1742


def test_no_duplicate_variants_among_labelled_rows(raw):
    normalised = raw[champ.LABEL_COLUMN].astype(str).str.strip().str.lower()
    labelled = raw[normalised.isin({"yes", "no"})]
    assert labelled["HGVS cDNA"].duplicated().sum() == 0


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_domain_acidic_regions_are_not_merged_into_a_domains(prepared):
    """a1/a2/a3 are the acidic regions and are biologically distinct from the
    A1/A2/A3 domains. Case-folding this column would corrupt 96 rows."""
    X, _, _, _ = prepared
    domains = set(X["Domain"].dropna().unique())
    for lower, upper in (("a1", "A1"), ("a2", "A2"), ("a3", "A3")):
        assert lower in domains, f"{lower} was merged away"
        assert upper in domains
    assert "Domain" not in champ._CASE_INSENSITIVE_COLUMNS


def test_case_typos_are_normalised(prepared):
    X, _, _, _ = prepared
    mechanisms = set(X["Mechanism"].dropna().unique())
    assert "deletion" not in mechanisms and "DuplIcation" not in mechanisms
    assert "Deletion" in mechanisms and "Duplication" in mechanisms

    severities = set(X["Reported Clinical Severity"].dropna().unique())
    assert "severe" not in severities and "Not Reported" not in severities

    assert set(X["In Poly A"].dropna().unique()) <= {"N", "Y"}


def test_exon_column_splits_into_number_and_intron_flag(prepared):
    """The legacy pipeline tested `isinstance(x, str)` on an int column, making
    its exon_risk feature constant. Exon is genuinely mixed: '14' and 'Intron 22'."""
    X, _, _, _ = prepared
    assert X["is_intron"].sum() == 187
    assert X["is_intron"].nunique() == 2, "is_intron must not be constant"
    assert X["exon_number"].nunique() > 20


@pytest.mark.parametrize(
    "value,expected",
    [("14", (14.0, 0)), ("Intron 22", (22.0, 1)), ("intron 4", (4.0, 1)), (None, None)],
)
def test_parse_exon(value, expected):
    number, is_intron = champ._parse_exon(value)
    if expected is None:
        assert np.isnan(number) and is_intron == 0
    else:
        assert (number, is_intron) == expected


def test_normalisation_never_changes_row_count_or_labels(raw):
    out = champ.normalise_champ(raw)
    assert len(out) == len(raw)
    pd.testing.assert_series_equal(
        out[champ.LABEL_COLUMN], raw[champ.LABEL_COLUMN], check_names=False
    )


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------


def test_reference_number_is_never_a_feature():
    """Source publication leaks the label: some studies are 100% positive."""
    assert "Reference Number" not in champ.FEATURE_COLUMNS
    assert "Reference Number" in champ.EXCLUDED_COLUMNS
    assert "LEAKAGE" in champ.EXCLUDED_COLUMNS["Reference Number"]


def test_identifier_columns_are_never_features():
    for col in ("HGVS cDNA", "hg19 Coordinates", "HGVS Protein", "Mature Protein"):
        assert col not in champ.FEATURE_COLUMNS
        assert col in champ.EXCLUDED_COLUMNS


def test_label_column_is_not_among_the_features():
    assert champ.LABEL_COLUMN not in champ.FEATURE_COLUMNS


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------


def test_preprocessor_output_has_no_missing_values(prepared):
    X, _, _, _ = prepared
    matrix = champ.build_preprocessor().fit_transform(X)
    assert not np.isnan(matrix).any()
    assert matrix.shape[0] == len(X)


def test_missing_categoricals_become_an_explicit_unknown_category(prepared):
    """SimpleImputer does not treat Python None as missing on object arrays, so
    nulls previously became a literal 'None' category."""
    X, _, _, _ = prepared
    pre = champ.build_preprocessor().fit(X)
    names = champ.encoded_feature_names(pre)
    assert any(n.endswith(f"_{champ.MISSING_CATEGORY}") for n in names)
    assert not any(n.endswith("_None") for n in names)


def test_feature_order_is_stable_across_transforms(prepared):
    X, _, _, _ = prepared
    pre = champ.build_preprocessor().fit(X)
    first = champ.encoded_feature_names(pre)
    second = champ.encoded_feature_names(pre)
    assert first == second
    assert len(first) == pre.transform(X.head(5)).shape[1]


def test_transform_is_row_order_independent(prepared):
    """One row transformed alone must equal that row inside a batch — the
    property the serving path depends on."""
    X, _, _, _ = prepared
    pre = champ.build_preprocessor().fit(X)
    batch = pre.transform(X.head(10))
    single = pre.transform(X.iloc[[3]])
    np.testing.assert_allclose(single[0], batch[3])


def test_encoded_names_map_back_to_source_champ_columns(prepared):
    X, _, _, _ = prepared
    pre = champ.build_preprocessor().fit(X)
    known = set(champ.FEATURE_COLUMNS)
    for name in champ.encoded_feature_names(pre):
        assert champ.source_column_for(name) in known, name


def test_category_value_extraction():
    assert champ.source_column_for("cat__Variant Type_Missense") == "Variant Type"
    assert champ.category_value_for("cat__Variant Type_Missense") == "Missense"
    assert champ.category_value_for("num__exon_number") is None


def test_fitted_categories_exposes_the_training_vocabulary(prepared):
    X, _, _, _ = prepared
    pre = champ.build_preprocessor().fit(X)
    categories = champ.fitted_categories(pre)
    assert set(categories) == set(champ.CATEGORICAL_FEATURES)
    assert "Missense" in categories["Variant Type"]
    assert "a1" in categories["Domain"] and "A1" in categories["Domain"]
