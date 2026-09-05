"""The machinery this migration introduced: parsing, fusion, and its guarantees.

Four things here have no coverage elsewhere, and each is a place where a plausible
"simplification" would silently destroy the pipeline's honesty:

* measurement parsing — most clotting readings are bounds, not numbers;
* mutation-level aggregation — the reason a frequently reported mutation cannot
  be memorised across the split;
* the conflicting-label exclusion — the reason a disputed outcome is not voted on;
* train/serve consistency — the reason inference cannot drift from training.

The hand-built frames are deliberately small enough to check by eye, and each is
followed by the same assertion against the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import hemophilia_a as ha


@pytest.fixture(scope="module")
def bundle():
    return ha.load_mutation_table()


@pytest.fixture(scope="module")
def mutations(bundle):
    return bundle.mutations


@pytest.fixture(scope="module")
def specs(mutations):
    return ha.build_feature_specs(mutations)


def _records(rows: list[dict]) -> pd.DataFrame:
    """A record-level frame shaped like the merged table, ready to aggregate."""
    frame = pd.DataFrame(rows)
    return ha.normalise_frame(frame, frame.columns)


# --------------------------------------------------------------------------
# Measurement parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "value", "censoring"),
    [
        # Plain readings.
        ("3", 3.0, ha.OBSERVED),
        ("0.5", 0.5, ha.OBSERVED),
        (2.0, 2.0, ha.OBSERVED),
        # Below the assay's detection limit — 3,113 clotting entries say this.
        ("<1", 1.0, ha.LEFT_CENSORED),
        ("<1.5", 1.5, ha.LEFT_CENSORED),
        # Above a reporting ceiling.
        (">5", 5.0, ha.RIGHT_CENSORED),
        (">20", 20.0, ha.RIGHT_CENSORED),
        # The same interval, typed three different ways in the source files.
        ("1 to 5", 3.0, ha.RANGE),
        ("1--5", 3.0, ha.RANGE),
        ("48-53", 50.5, ha.RANGE),
        # n_bp records large deletions in kilobases.
        ("7kb", 7000.0, ha.OBSERVED),
        (">55kb", 55000.0, ha.RIGHT_CENSORED),
        # A trailing annotation is not part of the number.
        ("13 (post)", 13.0, ha.OBSERVED),
        # A promoter position. The leading minus belongs to the number: reading
        # it as a range separator would turn one value into a fabricated interval.
        ("-860", -860.0, ha.OBSERVED),
    ],
)
def test_a_measurement_parses_to_its_value_and_censoring(raw, value, censoring):
    parsed, code = ha.parse_measurement(raw)
    assert parsed == pytest.approx(value)
    assert code == censoring


@pytest.mark.parametrize(
    "raw",
    [
        "Not reported",
        "not reported",
        "?",
        "Null",
        "NA",
        "",
        "   ",
        None,
        float("nan"),
        # Excel has mangled a couple of locnumb ranges into dates. That is not a
        # measurement and must not be guessed at.
        "Jul-13",
        "08-Sep",
    ],
)
def test_an_unparseable_measurement_is_missing_never_guessed(raw):
    value, code = ha.parse_measurement(raw)
    assert np.isnan(value)
    assert code is None


def test_parse_measurement_series_keeps_the_index(mutations):
    series = pd.Series(["<1", "3", "bad", ">5"], index=[10, 11, 12, 13])
    values, codes = ha.parse_measurement_series(series)
    assert list(values.index) == [10, 11, 12, 13]
    assert values.tolist()[:2] == [1.0, 3.0]
    assert np.isnan(values.loc[12])
    assert codes.tolist() == [ha.LEFT_CENSORED, ha.OBSERVED, None, ha.RIGHT_CENSORED]


# --------------------------------------------------------------------------
# Severity scoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "score"),
    [
        ("Mild", 1.0),
        ("Moderate", 2.0),
        ("Severe", 3.0),
        ("Severe/moderate", 2.5),
        ("Mild/Moderate", 1.5),
        # Case and whitespace are noise in this column, not meaning.
        ("severe", 3.0),
        ("  SEVERE  ", 3.0),
    ],
)
def test_severity_scores_are_ordinal(raw, score):
    assert ha.severity_score(raw) == pytest.approx(score)


@pytest.mark.parametrize("raw", ["Unclassified", "Not reported", None, float("nan")])
def test_an_unranked_severity_stays_missing(raw):
    """Not a middle value: "Unclassified" is an absence of evidence, and scoring
    it 2.0 would put it exactly where "Moderate" is."""
    assert np.isnan(ha.severity_score(raw))


# --------------------------------------------------------------------------
# Mutation-level aggregation
# --------------------------------------------------------------------------


@pytest.fixture()
def hand_built():
    """Two mutations: one reported four times, one reported once.

    m1's clotting readings are 2, 4, "<1" and "1 to 5" -> 2.0, 4.0, 1.0, 3.0,
    so mean 2.5, median 2.5, min 1.0, max 4.0, and half of them censored.
    Its severities are Severe, Severe, Mild, Moderate.
    """
    return _records(
        [
            {"mut_id": "m1", "target": 0, "clotting": "2", "cli_phe": "Severe",
             "mut_type": "Point", "aa_numb": "100"},
            {"mut_id": "m1", "target": 0, "clotting": "4", "cli_phe": "Severe",
             "mut_type": "Point", "aa_numb": "100"},
            {"mut_id": "m1", "target": 0, "clotting": "<1", "cli_phe": "Mild",
             "mut_type": "Point", "aa_numb": "100"},
            {"mut_id": "m1", "target": 0, "clotting": "1 to 5", "cli_phe": "Moderate",
             "mut_type": "Point", "aa_numb": "100"},
            {"mut_id": "m2", "target": 1, "clotting": "9", "cli_phe": "Mild",
             "mut_type": "Deletion", "aa_numb": "200"},
        ]
    )


def test_aggregation_yields_one_row_per_mutation(hand_built):
    table, report = ha.aggregate_to_mutations(hand_built)
    assert len(table) == 2
    assert table[ha.GROUP_COLUMN].tolist() == ["m1", "m2"]
    assert not table[ha.GROUP_COLUMN].duplicated().any()
    assert report.n_mutations == 2
    assert report.records_per_mutation_max == 4


def test_measurement_statistics_are_computed_over_every_record(hand_built):
    table, _ = ha.aggregate_to_mutations(hand_built)
    m1 = table.set_index(ha.GROUP_COLUMN).loc["m1"]
    assert m1["clotting_mean"] == pytest.approx(2.5)
    assert m1["clotting_median"] == pytest.approx(2.5)
    assert m1["clotting_min"] == pytest.approx(1.0)
    assert m1["clotting_max"] == pytest.approx(4.0)
    # "<1" is left-censored and "1 to 5" is a range; "2" and "4" are point values.
    assert m1["clotting_censored_rate"] == pytest.approx(0.5)


def test_severity_becomes_a_score_and_bucket_proportions(hand_built):
    table, _ = ha.aggregate_to_mutations(hand_built)
    m1 = table.set_index(ha.GROUP_COLUMN).loc["m1"]
    # Severe, Severe, Mild, Moderate -> (3 + 3 + 1 + 2) / 4
    assert m1["severity_score_mean"] == pytest.approx(2.25)
    assert m1["severity_score_min"] == pytest.approx(1.0)
    assert m1["severity_score_max"] == pytest.approx(3.0)
    assert m1["severity_prop_severe"] == pytest.approx(0.5)
    assert m1["severity_prop_mild"] == pytest.approx(0.25)
    assert m1["severity_prop_moderate"] == pytest.approx(0.25)
    assert m1["severity_n_distinct"] == pytest.approx(3)
    assert m1[f"{ha.SEVERITY_COLUMN}_mode"] == "Severe"


def test_a_single_record_mutation_collapses_to_its_one_value(hand_built):
    """This is what makes inference possible: a caller supplies one record, and
    the aggregates it produces are exactly this shape."""
    table, _ = ha.aggregate_to_mutations(hand_built)
    m2 = table.set_index(ha.GROUP_COLUMN).loc["m2"]
    for stat in ha.MEASUREMENT_AGGREGATES:
        assert m2[f"clotting_{stat}"] == pytest.approx(9.0)
    assert m2["clotting_censored_rate"] == pytest.approx(0.0)
    assert m2["severity_n_distinct"] == pytest.approx(1)


def test_genomic_columns_are_carried_through_unchanged(hand_built):
    table, _ = ha.aggregate_to_mutations(hand_built)
    by_id = table.set_index(ha.GROUP_COLUMN)
    assert by_id.loc["m1", "mut_type"] == "Point"
    assert by_id.loc["m2", "mut_type"] == "Deletion"
    # aa_numb is a position, so it is parsed rather than one-hot encoded.
    assert by_id.loc["m1", "aa_numb"] == pytest.approx(100.0)


def test_the_real_dataset_aggregates_to_one_row_per_mutation(mutations, bundle):
    assert len(mutations) == mutations[ha.GROUP_COLUMN].nunique()
    assert not mutations[ha.GROUP_COLUMN].duplicated().any()
    assert len(mutations) < len(bundle.records)
    assert mutations[ha.TARGET_COLUMN].notna().all()


def test_aggregation_leaves_no_python_none_in_an_object_column(mutations):
    """``groupby.first()`` re-materialises an all-null object group as ``None``,
    which SimpleImputer's ``X != X`` mask does not detect — so a null would be
    fitted as a literal "None" category instead of the explicit missing level."""
    for column in mutations.columns:
        if mutations[column].dtype == object:
            assert not any(v is None for v in mutations[column]), column


# --------------------------------------------------------------------------
# Conflicting labels
# --------------------------------------------------------------------------


@pytest.fixture()
def disputed():
    """Three mutations: unanimously negative, unanimously positive, and disputed."""
    return _records(
        [
            {"mut_id": "all_no", "target": 0, "clotting": "1", "cli_phe": "Severe",
             "mut_type": "Point"},
            {"mut_id": "all_no", "target": 0, "clotting": "2", "cli_phe": "Severe",
             "mut_type": "Point"},
            {"mut_id": "all_yes", "target": 1, "clotting": "1", "cli_phe": "Severe",
             "mut_type": "Point"},
            {"mut_id": "disputed", "target": 1, "clotting": "1", "cli_phe": "Severe",
             "mut_type": "Point"},
            {"mut_id": "disputed", "target": 0, "clotting": "2", "cli_phe": "Mild",
             "mut_type": "Point"},
        ]
    )


def test_unanimous_mutations_take_their_records_label(disputed):
    table, _ = ha.aggregate_to_mutations(disputed)
    labels = table.set_index(ha.GROUP_COLUMN)[ha.TARGET_COLUMN]
    assert labels.loc["all_no"] == 0
    assert labels.loc["all_yes"] == 1


def test_a_disputed_mutation_is_excluded_not_voted_on(disputed):
    """A majority vote would resolve toward whichever class happens to have more
    records, manufacturing a certainty the source data does not contain."""
    table, report = ha.aggregate_to_mutations(disputed)
    assert "disputed" not in set(table[ha.GROUP_COLUMN])
    assert report.n_conflicting == 1
    assert "disputed" in report.conflicting_mut_ids
    assert report.n_modelled == 2


def test_the_group_report_arithmetic_is_self_consistent(disputed):
    _, report = ha.aggregate_to_mutations(disputed)
    assert report.n_positive + report.n_negative == report.n_modelled
    assert report.n_modelled + report.n_conflicting == report.n_mutations


def test_the_real_conflict_count_matches_what_is_reported(bundle, mutations):
    report = bundle.groups
    assert report.n_conflicting > 0, "the real data does contain disputes"
    assert report.n_modelled == len(mutations)
    assert report.n_positive + report.n_negative == report.n_modelled
    assert report.n_modelled + report.n_conflicting == report.n_mutations
    assert len(report.conflicting_mut_ids) == report.n_conflicting
    # An excluded mutation must not reappear in the modelled table.
    assert not set(report.conflicting_mut_ids) & set(mutations[ha.GROUP_COLUMN])


def test_the_population_census_adds_up(bundle):
    census = bundle.population()
    assert (
        census["n_positive"] + census["n_negative"]
        == census["n_final_modelling_population"]
    )
    assert (
        census["n_final_modelling_population"] + census["n_conflicting_excluded"]
        == census["n_mutations_with_a_label"]
    )
    assert (
        census["n_mutations_with_a_label"] + census["n_mutations_unknown"]
        == census["n_mmc2_f8_mutations"]
    )


# --------------------------------------------------------------------------
# Leakage guards introduced by the migration
# --------------------------------------------------------------------------


def test_every_exclusion_records_why(mutations):
    """The exclusion list is documentation as much as behaviour: a column removed
    without a reason is indistinguishable from one removed by accident."""
    for column, reason in ha.EXCLUDED_COLUMNS.items():
        assert isinstance(reason, str) and reason.strip(), column
        # Two words is enough when the column speaks for itself ("free text"),
        # but a bare restatement of the column name is not a reason.
        assert len(reason.split()) >= 2, f"{column}: {reason!r} is not an explanation"
        assert reason.strip().casefold() != column.casefold(), column


@pytest.mark.parametrize(
    "column",
    [
        # Derived from inhibitor testing, so both value and presence are
        # post-outcome information.
        "uinhibitor",
        "type",
        "utype",
        "assay",
        # Reporting artifacts that track which cohorts were published.
        "pa_race",
        "reference",
        "ref_id",
        "pub_lab",
        "Count_mut_id",
        "n_clinical_records",
        # Near-unique identifiers, which let a model memorise mutations.
        "mut_syn",
        "aa_syn",
        "aa_change",
        "codon_change",
    ],
)
def test_a_leaky_column_reaches_no_feature_set(specs, column):
    assert column in ha.EXCLUDED_COLUMNS
    for name, spec in specs.items():
        assert column not in spec.columns, f"{column} is a {name} model feature"
        assert column not in spec.inputs, f"{column} is a {name} input"
        # …and no aggregate may be derived from it either.
        derived = [c for c in spec.columns if ha.aggregate_source_column(c) == column]
        assert not derived, f"{name} derives {derived} from {column}"


def test_no_identifier_like_feature_reaches_the_model(specs, mutations):
    for name, spec in specs.items():
        offenders = ha.identifier_like_columns(mutations, spec.columns)
        assert offenders == {}, f"{name}: {offenders}"


def test_forcing_an_identifier_into_the_candidates_is_refused(monkeypatch, mutations):
    """The guard has to be enforcement, not documentation. Re-admitting mut_syn
    is the single easiest way to make this model look far better than it is."""
    monkeypatch.setattr(
        ha, "GENOMIC_CANDIDATES", (*ha.GENOMIC_CANDIDATES, "mut_syn")
    )
    without = {k: v for k, v in ha.EXCLUDED_COLUMNS.items() if k != "mut_syn"}
    monkeypatch.setattr(ha, "EXCLUDED_COLUMNS", without)

    records = ha.load_merged()[0]
    table, _ = ha.aggregate_to_mutations(records)
    with pytest.raises(ValueError, match="Identifier-like columns cannot be features"):
        ha.build_feature_specs(table)


# --------------------------------------------------------------------------
# Train/serve consistency
# --------------------------------------------------------------------------


def test_a_served_row_matches_what_training_built_for_the_same_mutation(
    bundle, mutations, specs
):
    """The guarantee that inference cannot drift from training.

    Take a real mutation with exactly one clinical record, push its raw fields
    through the serving path, and require the result to equal that mutation's
    own row in the table the model was fitted on — column for column.
    """
    spec = specs["merged"]
    records = bundle.records

    counts = records[ha.GROUP_COLUMN].value_counts()
    singles = set(counts[counts == 1].index) & set(mutations[ha.GROUP_COLUMN])
    assert singles, "expected at least one single-record mutation"

    mut_id = sorted(singles)[0]
    record = records.loc[records[ha.GROUP_COLUMN] == mut_id].iloc[0]
    expected = mutations.loc[mutations[ha.GROUP_COLUMN] == mut_id].iloc[0]

    payload = {field: record.get(field) for field in spec.inputs}
    served = ha.build_input_row(spec, payload)

    assert list(served.columns) == spec.columns
    for column in spec.columns:
        got, want = served.iloc[0][column], expected[column]
        if isinstance(want, float) and np.isnan(want):
            assert pd.isna(got), column
        elif isinstance(want, float):
            assert got == pytest.approx(want), column
        else:
            assert got == want or (pd.isna(got) and pd.isna(want)), column


def test_a_censored_reading_survives_all_the_way_to_the_feature_row(specs):
    """"<1" and "1" must not produce the same features. If they did, the single
    most common form of clotting measurement would be silently discarded."""
    spec = specs["merged"]
    base = {"mut_type": "Point", "mut_effect": "Missense", "cli_phe": "Severe"}

    censored = ha.build_input_row(spec, {**base, "clotting": "<1"})
    observed = ha.build_input_row(spec, {**base, "clotting": "1"})

    # The value is the same bound…
    assert censored.iloc[0]["clotting_mean"] == pytest.approx(1.0)
    assert observed.iloc[0]["clotting_mean"] == pytest.approx(1.0)
    # …but the model is told one was a bound and the other a reading.
    assert censored.iloc[0]["clotting_censored_rate"] == pytest.approx(1.0)
    assert observed.iloc[0]["clotting_censored_rate"] == pytest.approx(0.0)


def test_an_absent_optional_field_arrives_as_missing_not_as_a_default(specs):
    spec = specs["merged"]
    row = ha.build_input_row(spec, {"mut_type": "Point", "cli_phe": "Severe"})
    assert pd.isna(row.iloc[0]["antigen_mean"])
    assert pd.isna(row.iloc[0]["discrep_mean"])
