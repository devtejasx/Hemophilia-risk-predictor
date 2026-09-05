"""MMC2 + MMC3 data integrity: loading, the join on mut_id, and the target.

These numbers are asserted so a record cannot be dropped, duplicated or
relabelled without the suite failing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.preprocessing import hemophilia_a as ha


@pytest.fixture(scope="module")
def mmc2():
    return ha.load_mmc2()


@pytest.fixture(scope="module")
def mmc3():
    return ha.load_mmc3()


@pytest.fixture(scope="module")
def prepared():
    return ha.load_merged()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_mmc2_loads_with_the_expected_shape(mmc2):
    assert len(mmc2) == 6211
    assert ha.GROUP_COLUMN in mmc2.columns
    assert ha.GENE_COLUMN in mmc2.columns
    assert not [c for c in mmc2.columns if c.startswith("Unnamed:")]


def test_mmc3_loads_with_the_expected_shape(mmc3):
    assert len(mmc3) == 10064
    assert ha.GROUP_COLUMN in mmc3.columns
    assert ha.LABEL_COLUMN in mmc3.columns
    assert not [c for c in mmc3.columns if c.startswith("Unnamed:")]


def test_mmc2_holds_one_row_per_mutation(mmc2):
    assert mmc2[ha.GROUP_COLUMN].nunique() == len(mmc2)
    assert mmc2.duplicated().sum() == 0


def test_mmc3_repeats_mut_id_because_a_mutation_has_many_records(mmc3):
    """This is why the split must be grouped rather than row-level."""
    assert mmc3[ha.GROUP_COLUMN].duplicated().sum() == 3852
    assert mmc3[ha.GROUP_COLUMN].nunique() == 6212
    assert mmc3.duplicated().sum() == 0


def test_validation_reports_a_missing_column():
    report = ha.validate_mmc3(pd.DataFrame({"mut_id": [1]}))
    assert not report.ok
    assert ha.LABEL_COLUMN in report.missing_columns
    with pytest.raises(ValueError, match="missing required columns"):
        report.raise_if_invalid()


def test_missing_file_names_the_configuration_knob(tmp_path):
    with pytest.raises(FileNotFoundError, match="MMC2_PATH"):
        ha.load_mmc2(tmp_path / "nope.csv")


# --------------------------------------------------------------------------
# Gene filter and target encoding
# --------------------------------------------------------------------------


def test_only_f8_mutations_are_kept(mmc2):
    f8 = ha.filter_gene(mmc2)
    assert len(f8) == 6205
    assert set(f8[ha.GENE_COLUMN].str.strip().str.upper()) == {"F8"}


def test_target_is_yes_1_no_0(mmc3):
    labelled, report = ha.encode_target(mmc3)
    assert report.n_labelled == 4966
    assert report.n_positive == 836
    assert labelled[ha.TARGET_COLUMN].sum() == 836
    assert set(labelled[ha.TARGET_COLUMN].unique()) == {0, 1}
    assert report.positive_rate == pytest.approx(0.1683, abs=1e-4)


def test_every_mmc3_record_is_accounted_for(mmc3):
    """No record may be dropped without appearing in the exclusion report."""
    _, report = ha.encode_target(mmc3)
    assert report.n_labelled + report.n_excluded_unlabelled == report.n_total == 10064
    assert sum(report.excluded_values.values()) == report.n_excluded_unlabelled
    # Named, not silently swallowed.
    assert report.excluded_values["not reported"] == 2090
    assert report.excluded_values["(missing)"] == 1730
    assert report.excluded_values["not"] == 1276


def test_ambiguous_inhibitor_values_are_excluded_not_guessed(mmc3):
    _, report = ha.encode_target(mmc3)
    # A severity typed into the inhibitor field is not evidence either way.
    assert report.excluded_values.get("severe") == 1
    assert report.excluded_values.get("mild") == 1


def test_target_encoding_is_case_and_space_insensitive():
    frame = pd.DataFrame(
        {"mut_id": [1, 2, 3, 4], ha.LABEL_COLUMN: ["Yes", "yes", "No ", "Not reported"]}
    )
    labelled, report = ha.encode_target(frame)
    assert list(labelled[ha.TARGET_COLUMN]) == [1, 1, 0]
    assert report.n_excluded_unlabelled == 1


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def test_merge_counts(prepared):
    merged, labels, report, _ = prepared
    assert report.n_mmc2_rows == 6211
    assert report.n_mmc2_f8_rows == 6205
    assert report.n_mmc3_rows == 10064
    assert report.n_mmc3_labelled == 4966
    assert report.n_merged_rows == 4962 == len(merged)
    assert report.n_unique_mut_id == 2639
    assert report.n_unmatched_clinical == 4
    assert labels.n_positive == 836


def test_merge_does_not_duplicate_clinical_records(prepared):
    """One genomic row per mutation, so the join cannot multiply MMC3 rows."""
    merged, _, report, _ = prepared
    assert report.n_merged_rows <= report.n_mmc3_labelled
    assert report.n_duplicate_rows == 0


def test_merge_keeps_every_record_of_a_repeated_mutation(prepared):
    """De-duplicating MMC3 on mut_id would discard real clinical records."""
    merged, _, report, _ = prepared
    assert report.records_per_mutation_max == 104
    assert len(merged) > merged[ha.GROUP_COLUMN].nunique()


def test_conflicting_labels_are_reported_and_retained(prepared):
    """Two records of one mutation may disagree; that is data, not an error."""
    merged, _, report, _ = prepared
    assert report.n_conflicting_label_groups == 124
    per_group = merged.groupby(ha.GROUP_COLUMN)[ha.TARGET_COLUMN].nunique()
    assert int((per_group > 1).sum()) == 124


def test_target_distribution(prepared):
    _, _, report, _ = prepared
    assert report.target_distribution == {"0": 4126, "1": 836}


def test_case_only_spelling_variants_are_folded_and_reported(prepared):
    merged, _, report, _ = prepared
    assert report.case_collapses["mut_effect"]["missense"] == "Missense"
    assert report.case_collapses["cli_phe"]["severe"] == "Severe"
    for column in ("mut_effect", "cli_phe", "location"):
        values = merged[column].dropna().astype(str)
        assert len(set(values)) == len({v.casefold() for v in values}), column


def test_whitespace_variants_are_normalised(prepared):
    merged, _, _, _ = prepared
    for column in ("mut_type", "location", "cli_phe"):
        values = merged[column].dropna().astype(str)
        assert not any(v != v.strip() for v in values), column


def test_missing_values_are_reported_per_feature(prepared):
    _, _, report, _ = prepared
    assert report.missing_by_feature["mut_type"] == 0.0
    # discrep is measured for almost nobody; the report must say so rather than
    # letting a 99%-empty column look like a usable one.
    assert report.missing_by_feature["discrep"] > 0.99
    assert set(report.missing_by_feature) <= set(ha.GENOMIC_CANDIDATES) | set(
        ha.CLINICAL_CANDIDATES
    )


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------


def test_the_target_is_never_a_feature(prepared):
    merged, _, _, _ = prepared
    for spec in ha.build_feature_specs(merged).values():
        assert ha.LABEL_COLUMN not in spec.columns
        assert ha.TARGET_COLUMN not in spec.columns


def test_the_grouping_key_is_never_a_feature(prepared):
    merged, _, _, _ = prepared
    for spec in ha.build_feature_specs(merged).values():
        assert ha.GROUP_COLUMN not in spec.columns


def test_curated_inhibitor_status_is_excluded_as_leakage(prepared):
    """uinhibitor is MMC2's own answer to the question being predicted."""
    merged, _, _, _ = prepared
    assert "uinhibitor" in ha.EXCLUDED_COLUMNS
    assert "LEAKAGE" in ha.EXCLUDED_COLUMNS["uinhibitor"]
    for spec in ha.build_feature_specs(merged).values():
        assert "uinhibitor" not in spec.columns


def test_identifier_and_publication_columns_are_excluded(prepared):
    merged, _, _, _ = prepared
    columns = set(ha.build_feature_specs(merged)["merged"].columns)
    for column in ("pa_id", "all_id", "ref_id", "reference", "pub_lab", "Count_mut_id"):
        assert column in ha.EXCLUDED_COLUMNS
        assert column not in columns
