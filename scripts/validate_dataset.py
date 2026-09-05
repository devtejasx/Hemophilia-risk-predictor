"""Report what the MMC2 + MMC3 files contain, before any model is fitted.

    python scripts/validate_dataset.py            # human-readable
    python scripts/validate_dataset.py --json     # machine-readable

Prints row counts, unique and duplicate ``mut_id``, missing values, the result
of the join, the target distribution, and every feature resolved for the
genomic / clinical / merged blocks. Exits non-zero if a required column is
missing, so it can gate a training run in CI.

Nothing here fits anything or writes to the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing import hemophilia_a as ha  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmc2", default=None, help="Overrides MMC2_PATH")
    parser.add_argument("--mmc3", default=None, help="Overrides MMC3_PATH")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    p2 = Path(args.mmc2) if args.mmc2 else ha.mmc2_path()
    p3 = Path(args.mmc3) if args.mmc3 else ha.mmc3_path()

    try:
        bundle = ha.load_mutation_table(p2, p3)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    labels, merge, sources = bundle.labels, bundle.merge, bundle.sources
    groups, population = bundle.groups, bundle.population()
    specs = ha.build_feature_specs(bundle.mutations)

    payload = {
        "sources": [s.as_dict() for s in sources],
        "labels": labels.as_dict(),
        "merge": merge.as_dict(),
        "mutation_level": groups.as_dict(),
        "population": population,
        "missing_by_feature": merge.missing_by_feature,
        "feature_sets": {name: spec.as_dict() for name, spec in specs.items()},
        "identifier_like_features": {
            name: ha.identifier_like_columns(bundle.mutations, spec.columns)
            for name, spec in specs.items()
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    line = "=" * 72
    print(line)
    print("MMC2 + MMC3 dataset validation")
    print(line)

    print("\nSOURCE FILES")
    for s in sources:
        print(f"  {s.name}  {s.path}")
        print(f"    rows                 {s.n_rows}")
        print(f"    columns              {s.n_columns}")
        print(f"    unique mut_id        {s.n_unique_mut_id}")
        print(f"    duplicate rows       {s.n_duplicate_rows}")
        print(f"    repeated mut_id      {s.n_duplicate_mut_id}")
        print(f"    missing values       {s.missing_values_total}")

    print("\nTARGET  (Inhibitors: Yes -> 1, No -> 0)")
    print(f"  MMC3 records             {labels.n_total}")
    print(f"  explicit Yes/No          {labels.n_labelled}")
    print(f"  positive                 {labels.n_positive} ({labels.positive_rate:.2%})")
    print(f"  excluded (no Yes/No)     {labels.n_excluded_unlabelled}")
    for value, count in sorted(
        labels.excluded_values.items(), key=lambda kv: -kv[1]
    ):
        print(f"      {value!r:20} {count}")

    print("\nMERGE ON mut_id")
    for key, value in merge.as_dict().items():
        if key == "case_collapses":
            continue
        print(f"  {key:28} {value}")

    if merge.case_collapses:
        print("\n  case-only spelling variants folded onto the dominant spelling:")
        for col, mapping in merge.case_collapses.items():
            for variant, canonical in mapping.items():
                print(f"      {col:14} {variant!r} -> {canonical!r}")

    print("\nFUSION TO ONE ROW PER MUTATION")
    for key, value in population.items():
        print(f"  {key:34} {value}")
    print(f"  {'records_per_mutation_mean':34} {groups.records_per_mutation_mean:.2f}")
    if groups.n_conflicting:
        sample = ", ".join(groups.conflicting_mut_ids[:10])
        print(
            f"\n  {groups.n_conflicting} mutations have clinical records that "
            "disagree about the\n  inhibitor outcome and are EXCLUDED from "
            f"supervised training (mut_id {sample}, ...).\n  They are not "
            "resolved by majority vote: the source data does not contain a\n"
            "  single answer for them."
        )

    print("\nMISSING VALUES BY CANDIDATE FEATURE (record-level merged frame)")
    for col, rate in sorted(
        merge.missing_by_feature.items(), key=lambda kv: -kv[1]
    ):
        print(f"  {col:16} {rate:7.2%}")

    print("\nFEATURE SETS  (model features are aggregates of the source fields)")
    for name, spec in specs.items():
        print(
            f"  {name}  ({len(spec.columns)} model features from "
            f"{len(spec.inputs)} source fields)"
        )
        print(f"    source fields  {list(spec.inputs)}")
        print(f"    categorical    {list(spec.categorical)}")
        print(f"    numeric        {list(spec.numeric)}")
        print(f"    required       {list(spec.required)}")

    print("\nIDENTIFIER GUARD")
    print(
        f"  A categorical feature may not be more than "
        f"{ha.MAX_CATEGORY_UNIQUENESS_RATIO:.0%} distinct."
    )
    for name, spec in specs.items():
        offenders = ha.identifier_like_columns(bundle.mutations, spec.columns)
        print(f"  {name:9} {offenders or 'clean'}")

    print("\nEXCLUDED COLUMNS")
    for col, reason in ha.EXCLUDED_COLUMNS.items():
        print(f"  {col:16} {reason}")

    print(f"\n{line}\nOK\n{line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
