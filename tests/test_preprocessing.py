"""Feature resolution, the fitted transformer, and the grouped split.

Several of these encode specific defects the pipeline is designed to avoid, so a
regression fails the suite rather than being rediscovered later.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing import hemophilia_a as ha
from tests.conftest import requires_model


@pytest.fixture(scope="module")
def merged():
    frame, _, _, _ = ha.load_merged()
    return frame


@pytest.fixture(scope="module")
def specs(merged):
    return ha.build_feature_specs(merged)


# --------------------------------------------------------------------------
# Feature resolution
# --------------------------------------------------------------------------


def test_the_three_feature_sets_exist(specs):
    assert set(specs) == set(ha.FEATURE_SET_NAMES)


def test_genomic_and_clinical_features_come_from_the_right_table(specs):
    assert set(specs["genomic"].columns) <= set(ha.GENOMIC_CANDIDATES)
    assert set(specs["clinical"].columns) <= set(ha.CLINICAL_CANDIDATES)
    assert not set(specs["genomic"].columns) & set(specs["clinical"].columns)


def test_merged_is_the_union_of_the_other_two(specs):
    assert set(specs["merged"].columns) == set(specs["genomic"].columns) | set(
        specs["clinical"].columns
    )


def test_resolved_feature_counts(specs):
    assert len(specs["genomic"].columns) == 20
    assert len(specs["clinical"].columns) == 9
    assert len(specs["merged"].columns) == 29


def test_candidates_absent_from_the_data_are_dropped_not_invented(specs, merged):
    """`mutations` is in both tables, so the join suffixes it and neither
    `mutations_clinical` nor `mutations_genomic` is the candidate column."""
    assert "mutations" in specs["genomic"].dropped
    assert "mutations" not in merged.columns
    assert "mutations_genomic" in merged.columns


def test_all_null_candidates_are_dropped(specs, merged):
    for column in ("bleed_tool", "bleed_score"):
        assert column in specs["clinical"].dropped
        assert merged[column].isna().all()


def test_every_resolved_feature_exists_and_varies(specs, merged):
    for spec in specs.values():
        for column in spec.columns:
            assert column in merged.columns, column
            assert merged[column].nunique(dropna=False) > 1, column


def test_numeric_and_categorical_partition_the_columns(specs):
    for spec in specs.values():
        assert set(spec.categorical) | set(spec.numeric) == set(spec.columns)
        assert not set(spec.categorical) & set(spec.numeric)


def test_required_columns_are_the_ones_that_are_nearly_always_present(specs, merged):
    for spec in specs.values():
        for column in spec.required:
            assert merged[column].isna().mean() <= ha.REQUIRED_MAX_MISSING_RATE


def test_feature_spec_round_trips_through_metadata(specs):
    for spec in specs.values():
        assert ha.FeatureSpec.from_dict(spec.as_dict()) == spec


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(merged, specs):
    spec = specs["merged"]
    pre = ha.build_preprocessor(spec).fit(merged[spec.columns])
    return spec, pre


def test_preprocessor_output_has_no_missing_values(fitted, merged):
    spec, pre = fitted
    matrix = pre.transform(merged[spec.columns])
    assert not np.isnan(matrix).any()
    assert matrix.shape[0] == len(merged)


def test_missing_categoricals_become_an_explicit_unknown_level(fitted):
    """SimpleImputer does not treat Python None as missing on object arrays, so
    nulls would otherwise become a literal 'None' category."""
    _, pre = fitted
    names = ha.encoded_feature_names(pre)
    assert any(n.endswith(f"_{ha.MISSING_CATEGORY}") for n in names)
    assert not any(n.endswith("_None") for n in names)


def test_numeric_missingness_is_recorded_not_hidden(fitted):
    """An imputed median must stay distinguishable from an observed one."""
    _, pre = fitted
    names = ha.encoded_feature_names(pre)
    assert any("missingindicator_" in n for n in names)


def test_feature_order_is_stable_across_transforms(fitted, merged):
    spec, pre = fitted
    first = ha.encoded_feature_names(pre)
    assert first == ha.encoded_feature_names(pre)
    assert len(first) == pre.transform(merged[spec.columns].head(5)).shape[1]


def test_transform_is_row_order_independent(fitted, merged):
    """One row transformed alone must equal that row inside a batch — the
    property the serving path depends on."""
    spec, pre = fitted
    frame = merged[spec.columns]
    batch = pre.transform(frame.head(10))
    single = pre.transform(frame.iloc[[3]])
    np.testing.assert_allclose(single[0], batch[3])


def test_encoded_names_map_back_to_source_columns(fitted):
    spec, pre = fitted
    known = set(spec.columns)
    for name in ha.encoded_feature_names(pre):
        assert spec.source_column_for(name) in known, name


def test_source_column_mapping_prefers_the_longest_match(specs):
    spec = specs["merged"]
    # aa_numb and aa_numb_old share a prefix; a naive scan would mis-attribute.
    assert spec.source_column_for("num__aa_numb_old") == "aa_numb_old"
    assert spec.source_column_for("num__aa_numb") == "aa_numb"
    assert spec.source_column_for("cat__mut_type_Point") == "mut_type"
    assert spec.category_value_for("cat__mut_type_Point") == "Point"
    assert spec.category_value_for("num__aa_numb") is None


def test_fitted_categories_expose_the_training_vocabulary(fitted):
    spec, pre = fitted
    categories = ha.fitted_categories(pre, spec)
    assert set(categories) == set(spec.categorical)
    assert "Point" in categories["mut_type"]
    assert "Severe" in categories["cli_phe"]


def test_offered_categories_exclude_the_infrequent_bucket(fitted):
    spec, pre = fitted
    seen = ha.fitted_categories(pre, spec)
    offered = ha.frequent_categories(pre, spec)
    for column in spec.categorical:
        assert set(offered[column]) <= set(seen[column])
    # mut_syn is a near-identifier: most of its values are folded away.
    assert len(offered["mut_syn"]) < len(seen["mut_syn"])


def test_open_vocabulary_columns_are_the_identifier_like_ones(fitted):
    spec, pre = fitted
    open_columns = ha.open_vocabulary_columns(pre, spec)
    assert {"mut_syn", "aa_syn", "nuc_numb", "clotting"} <= open_columns
    assert not {"mut_type", "location", "cli_phe", "CpG"} & open_columns


def test_a_feature_set_with_no_columns_is_refused():
    empty = ha.FeatureSpec(name="empty", categorical=(), numeric=())
    with pytest.raises(ValueError, match="no usable columns"):
        ha.build_preprocessor(empty)


# --------------------------------------------------------------------------
# Grouped split (leakage)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split(merged):
    from scripts.train_inhibitor_model import grouped_split

    return grouped_split(merged, seed=42, test_size=0.20, val_size=0.20)


def test_grouped_split_covers_every_row_exactly_once(merged, split):
    train, val, test = split
    assert len(train) + len(val) + len(test) == len(merged)
    assert len(set(train) | set(val) | set(test)) == len(merged)


def test_no_mutation_appears_in_two_splits(merged, split):
    train, val, test = split
    groups = merged[ha.GROUP_COLUMN]
    gt, gv, gs = set(groups.iloc[train]), set(groups.iloc[val]), set(groups.iloc[test])
    assert gt.isdisjoint(gv)
    assert gt.isdisjoint(gs)
    assert gv.isdisjoint(gs)
    assert len(gt) + len(gv) + len(gs) == groups.nunique()


def test_leakage_assertion_rejects_an_overlapping_split(merged, split):
    from scripts.train_inhibitor_model import assert_no_group_overlap

    train, val, test = split
    assert_no_group_overlap(merged, train, val, test)  # the real split is clean
    with pytest.raises(RuntimeError, match="Group leakage"):
        assert_no_group_overlap(merged, train, val, np.concatenate([test, train[:5]]))


def test_a_random_row_split_would_have_leaked(merged):
    """The reason the split is grouped: a plain shuffle puts records of the same
    mutation on both sides."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        np.arange(len(merged)), test_size=0.2, random_state=42
    )
    groups = merged[ha.GROUP_COLUMN]
    assert set(groups.iloc[train]) & set(groups.iloc[test])


def test_every_split_contains_both_classes(merged, split):
    y = merged[ha.TARGET_COLUMN]
    for part in split:
        assert set(y.iloc[part].unique()) == {0, 1}


# --------------------------------------------------------------------------
# Train / inference agreement
# --------------------------------------------------------------------------


@requires_model
def test_the_served_preprocessor_matches_its_recorded_feature_space(service):
    """The artifact's metadata, its preprocessor and its estimator must agree."""
    encoded = ha.encoded_feature_names(service.bundle.preprocessor)
    assert encoded == service.bundle.feature_names
    assert service.bundle.model.n_features_in_ == len(encoded)


@requires_model
def test_inference_columns_are_exactly_the_training_columns(service):
    frame = service.validate(_minimal_payload(service))
    assert list(frame.columns) == service.spec.columns


def _minimal_payload(service) -> dict:
    schema = service.input_schema()
    payload = {}
    for column in schema["required"]:
        payload[column] = (
            schema["categorical"][column][0]
            if column in schema["categorical"]
            else 1000.0
        )
    return payload
