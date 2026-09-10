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
def bundle():
    return ha.load_mutation_table()


@pytest.fixture(scope="module")
def merged(bundle):
    """The mutation-level table - one row per mut_id - which is what is modelled."""
    return bundle.mutations


@pytest.fixture(scope="module")
def records(bundle):
    """The record-level table the mutation table was aggregated from."""
    return bundle.records


@pytest.fixture(scope="module")
def specs(merged):
    return ha.build_feature_specs(merged)


# --------------------------------------------------------------------------
# Feature resolution
# --------------------------------------------------------------------------


def test_the_three_feature_sets_exist(specs):
    assert set(specs) == set(ha.FEATURE_SET_NAMES)


def test_genomic_and_clinical_features_come_from_the_right_table(specs):
    """MMC2 columns stay in the genomic block, MMC3 columns in the clinical one.

    Genomic columns survive aggregation under their own names, so they can be
    compared to the candidate list directly. Clinical columns cannot: each raw
    field becomes several aggregates (``clotting`` -> ``clotting_mean``, ...),
    so every model column is mapped back to the field it was derived from
    before the comparison.
    """
    genomic, clinical = specs["genomic"], specs["clinical"]

    assert set(genomic.columns) <= set(ha.GENOMIC_CANDIDATES)
    assert set(genomic.inputs) <= set(ha.GENOMIC_CANDIDATES)

    clinical_sources = {ha.aggregate_source_column(c) for c in clinical.columns}
    assert clinical_sources <= set(ha.CLINICAL_CANDIDATES)
    assert set(clinical.inputs) <= set(ha.CLINICAL_CANDIDATES)

    # Disjoint as model columns and as the raw fields a caller supplies: no
    # column may be claimed by both blocks under either name.
    assert not set(genomic.columns) & set(clinical.columns)
    assert not set(genomic.inputs) & set(clinical.inputs)


def test_merged_is_the_union_of_the_other_two(specs):
    assert set(specs["merged"].columns) == set(specs["genomic"].columns) | set(
        specs["clinical"].columns
    )


def test_resolved_feature_counts(specs):
    """Merged is genomic + clinical, and each block is non-trivial.

    Asserted as a relationship rather than three magic numbers: the counts move
    whenever a column is excluded for a documented reason, and a test that has
    to be edited for every such change stops being a check.
    """
    genomic, clinical, merged = (specs[k] for k in ("genomic", "clinical", "merged"))
    assert len(genomic.columns) >= 10
    assert len(clinical.columns) >= 10
    assert len(merged.columns) == len(genomic.columns) + len(clinical.columns)
    # The clinical block is aggregates of a handful of raw fields, so it must
    # expand: five statistics per measurement plus the severity features.
    assert len(clinical.columns) > len(clinical.inputs)


def test_candidates_absent_from_the_data_are_dropped_not_invented(specs, records):
    """`mutations` is in both tables, so the join suffixes it.

    Neither `mutations_clinical` nor `mutations_genomic` is the candidate
    column, and the constant column must not reach a feature set under any name.
    """
    assert "mutations" in ha.EXCLUDED_COLUMNS
    assert "mutations" not in records.columns
    assert "mutations_genomic" in records.columns
    for spec in specs.values():
        assert not [c for c in spec.columns if c.startswith("mutations")]


def test_all_null_candidates_are_dropped(specs, records):
    """bleed_tool and bleed_score are empty in the source, so they are excluded.

    They are named in EXCLUDED_COLUMNS with that reason rather than being
    silently filtered, so the exclusion travels with the pipeline.
    """
    for column in ("bleed_tool", "bleed_score"):
        assert column in ha.EXCLUDED_COLUMNS
        assert records[column].isna().all()
        for spec in specs.values():
            assert column not in spec.columns


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
    """`required` names the RAW fields a caller fills in, not model columns.

    The missing rate is therefore read off the aggregates that field produces.
    Both directions are asserted: a required field is nearly always present, and
    a field left optional genuinely is not - otherwise the API could quietly
    stop asking for something it depends on.
    """
    for spec in specs.values():
        assert set(spec.required) <= set(spec.inputs)
        for field in spec.inputs:
            derived = [c for c in spec.columns if ha.aggregate_source_column(c) == field]
            assert derived, f"{spec.name}: no model column derives from {field}"
            rates = [float(merged[c].isna().mean()) for c in derived]
            if field in spec.required:
                assert max(rates) <= ha.REQUIRED_MAX_MISSING_RATE, field
            else:
                assert min(rates) > ha.REQUIRED_MAX_MISSING_RATE, field


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


#: The two spellings a null can reach the encoder under. ``MISSING_CATEGORY``
#: is the value ``SimpleImputer`` fills in. ``"None"`` is what pandas'
#: ``groupby.first()`` leaves behind when a mutation's records are all null on
#: an object column: it re-materialises the gap as Python ``None``, and
#: ``SimpleImputer``'s mask is ``X != X``, which ``None`` does not satisfy - so
#: it is not imputed and the encoder fits it as its own literal level.
NULL_MARKERS = frozenset({ha.MISSING_CATEGORY, "None"})


def test_missing_categoricals_become_an_explicit_level(fitted, merged, records):
    """A null must reach the model as its own level, never folded into a real one.

    That is the guarantee: a mutation with no recorded ``CpG`` status must be
    representable, and must not be indistinguishable from one that has a
    recorded status. It is asserted three ways, so it holds whichever of the two
    mechanisms above produced the level:

    * a column with nulls gets exactly one null marker - not two, not zero;
    * a column without nulls gets none, so no marker is invented;
    * no marker is a spelling the source files themselves use, which is what
      would make "not reported" collide with a genuine value.

    The imputer path is asserted separately, because it is the one the pipeline
    controls deliberately and a silent regression there would leave every
    categorical relying on the pandas accident.
    """
    spec, pre = fitted
    categories = ha.fitted_categories(pre, spec)

    assert any(
        ha.MISSING_CATEGORY in categories[c] for c in spec.categorical
    ), "the explicit-missing imputation path never reached the encoder"

    for column in spec.categorical:
        source = ha.aggregate_source_column(column)
        observed = (
            set(records[source].dropna().astype(str))
            if source in records.columns
            else set()
        )
        assert not NULL_MARKERS & observed, (
            f"{source} uses {sorted(NULL_MARKERS & observed)} as a real value, "
            "so a missing value can no longer be told apart from a recorded one"
        )
        markers = NULL_MARKERS & set(categories[column])
        if merged[column].isna().any():
            assert len(markers) == 1, f"{column}: {sorted(markers)}"
        else:
            assert not markers, f"{column}: {sorted(markers)}"


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
    """Every encoded column resolves to a RAW field the caller supplies.

    ``source_column_for`` now peels off two layers - the encoder's prefix and
    one-hot suffix, then the aggregation suffix - so it lands on ``clotting``
    rather than ``clotting_median``. An explanation therefore names something
    the caller actually filled in, which is only true if the result is always a
    member of ``spec.inputs``.
    """
    spec, pre = fitted
    known = set(spec.inputs)
    resolved = set()
    for name in ha.encoded_feature_names(pre):
        source = spec.source_column_for(name)
        assert source in known, name
        resolved.add(source)
    # No input is unreachable: every field the API asks for shows up behind at
    # least one encoded column, so nothing is requested that the model ignores.
    assert resolved == known


def test_source_column_mapping_prefers_the_longest_match(specs):
    spec = specs["merged"]
    # clotting_mean and clotting_median share a prefix; a naive scan would
    # mis-attribute one to the other. Both must land on the field a caller
    # actually filled in.
    assert spec.source_column_for("num__clotting_mean") == "clotting"
    assert spec.source_column_for("num__clotting_median") == "clotting"
    assert spec.source_column_for("num__clotting_censored_rate") == "clotting"
    assert spec.source_column_for("num__severity_prop_severe") == "cli_phe"
    assert spec.source_column_for("cat__cli_phe_mode_Severe") == "cli_phe"
    assert spec.source_column_for("num__aa_numb") == "aa_numb"
    assert spec.source_column_for("cat__mut_type_Point") == "mut_type"
    assert spec.category_value_for("cat__mut_type_Point") == "Point"
    assert spec.category_value_for("num__aa_numb") is None


def test_fitted_categories_expose_the_training_vocabulary(fitted):
    """The vocabulary is keyed on MODEL columns, so severity lives on the
    aggregate ``cli_phe_mode`` - the dominant phenotype across a mutation's
    records - and not on the raw ``cli_phe`` field it was derived from."""
    spec, pre = fitted
    categories = ha.fitted_categories(pre, spec)
    assert set(categories) == set(spec.categorical)
    assert "Point" in categories["mut_type"]
    assert "Severe" in categories[f"{ha.SEVERITY_COLUMN}_mode"]
    assert ha.SEVERITY_COLUMN not in categories
    assert ha.SEVERITY_COLUMN in spec.inputs


def test_offered_categories_exclude_the_infrequent_bucket(fitted):
    spec, pre = fitted
    seen = ha.fitted_categories(pre, spec)
    offered = ha.frequent_categories(pre, spec)
    for column in spec.categorical:
        assert set(offered[column]) <= set(seen[column])
    # aa_last has a long tail of rare spellings; those are folded into the
    # encoder's "infrequent" bucket and must not be offered as choices.
    assert len(offered["aa_last"]) < len(seen["aa_last"])


def test_open_vocabulary_columns_are_the_high_cardinality_ones(fitted):
    """A wide categorical accepts unseen values; a small closed one does not.

    aa_last carries hundreds of spellings, so rejecting an unseen one would be
    wrong. mut_type has six, so an unrecognised value there is a caller error
    and must produce a named 422 rather than a bucketed guess.
    """
    spec, pre = fitted
    open_columns = ha.open_vocabulary_columns(pre, spec)
    assert "aa_last" in open_columns
    assert not {"mut_type", "location", "cli_phe_mode", "CpG"} & open_columns


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


def test_a_random_row_split_of_the_records_would_have_leaked(records, merged):
    """Why aggregation happens *before* the split, not after.

    A random row split of the mutation table cannot leak - each mutation is
    exactly one row there, so the property is trivially true and asserting it
    would test nothing. The safety is manufactured one step earlier: the
    record-level table this mutation table was aggregated from carries many
    rows per ``mut_id``, and a plain shuffle of *those* rows puts records of the
    same mutation - and its identical genomic block - on both sides.

    So the guarantee is asserted where it can still fail: the records would have
    leaked, and aggregation is what removes the possibility.
    """
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        np.arange(len(records)), test_size=0.2, random_state=42
    )
    groups = records[ha.GROUP_COLUMN]
    assert set(groups.iloc[train]) & set(groups.iloc[test])

    # The collapse is real - there are strictly fewer mutations than records -
    # and it is complete: no mut_id survives twice, so no split of the modelled
    # table can put one mutation on both sides.
    assert records[ha.GROUP_COLUMN].duplicated().any()
    assert len(merged) < len(records)
    assert not merged[ha.GROUP_COLUMN].duplicated().any()


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
