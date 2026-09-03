"""Train and evaluate the MMC2 + MMC3 inhibitor-risk models.

    python scripts/train_inhibitor_model.py                    # all three sets
    python scripts/train_inhibitor_model.py --feature-set merged
    python scripts/train_inhibitor_model.py --seed 7 --version-suffix v2

Pipeline order matters and is enforced here:

    load MMC2 + MMC3 -> validate -> filter F8 -> encode target
         -> merge on mut_id
         -> GROUPED SPLIT FIRST (mut_id never crosses a split boundary)
         -> resolve feature sets from the columns that actually exist
         -> fit preprocessor on TRAIN ONLY
         -> model selection by grouped cross-validated ROC-AUC
         -> isotonic calibration on train
         -> choose the decision threshold on the VALIDATION split
         -> evaluate ONCE on the held-out test split
         -> write model + preprocessor + background + metadata + metrics

Every number in metrics.json comes from this run. Nothing is copied from a
report or hand-edited.

Reference: final(1).ipynb, cells 3-10. Deviations are listed in
``hemophilia_a.NOTEBOOK_DEVIATIONS`` and repeated in each metadata.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sklearn  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from xgboost import XGBClassifier  # noqa: E402

from ml.preprocessing import hemophilia_a as ha  # noqa: E402

PREPROCESSING_VERSION = "mmc-preprocessing-1"
DATASET_NAME = "MMC2+MMC3"
VERSION_PREFIX = "mmc"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def candidate_models(seed: int) -> dict[str, object]:
    """The families compared before one is fitted.

    Unchanged from the previous pipeline: class weighting handles the ~17%
    positive rate without resampling, and SMOTE is evaluated separately inside
    cross-validation folds only.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest_smote": ImbPipeline(
            [
                ("smote", SMOTE(random_state=seed, k_neighbors=5)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_split(
    merged: pd.DataFrame, seed: int, test_size: float, val_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices so no ``mut_id`` appears in two partitions.

    A plain row-level split would put clinical records of the *same* mutation
    into training and test. Those records share an identical genomic block, so
    the model would be scored on mutations it had already memorised.

    Follows final(1).ipynb cell 5: GroupShuffleSplit for train+val / test, then
    again for train / val, with a different seed for the inner split.
    """
    groups = merged[ha.GROUP_COLUMN].to_numpy()
    y = merged[ha.TARGET_COLUMN].to_numpy()
    index = np.arange(len(merged))

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_idx, test_idx = next(outer.split(index, y, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    rel_train, rel_val = next(
        inner.split(fit_idx, y[fit_idx], groups=groups[fit_idx])
    )
    return fit_idx[rel_train], fit_idx[rel_val], test_idx


def assert_no_group_overlap(
    merged: pd.DataFrame, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> dict[str, int]:
    g = merged[ha.GROUP_COLUMN]
    gt, gv, gs = set(g.iloc[train]), set(g.iloc[val]), set(g.iloc[test])
    overlaps = {
        "train_validation": len(gt & gv),
        "train_test": len(gt & gs),
        "validation_test": len(gv & gs),
    }
    for pair, count in overlaps.items():
        if count:
            raise RuntimeError(
                f"Group leakage: {count} mut_id values appear in both halves of "
                f"{pair}."
            )
    return {
        "train_mutations": len(gt),
        "validation_mutations": len(gv),
        "test_mutations": len(gs),
        "overlaps": overlaps,
    }


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
    """Probability cut-off that maximises F1 on the validation split.

    F1 rather than accuracy: at a ~17% positive rate a model that answers "no
    inhibitor" for everyone scores 83% accuracy and is useless. (The reference
    notebook selects on accuracy; see NOTEBOOK_DEVIATIONS.) The chosen value is
    written to metadata, so nothing downstream assumes 0.5.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), "max F1 on the validation split"


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(np.mean(y_true)),
        "at_threshold": {
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
            "specificity": specificity,
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
    }


def leakage_probe(merged: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Per-column association with the target, as a leakage tell-tale.

    A column whose value almost determines the label is reported here so it can
    be reviewed. ``uinhibitor`` — MMC2's curated inhibitor status — scores near
    1.0, which is why it is on the exclusion list.
    """
    y = merged[ha.TARGET_COLUMN]
    scores: dict[str, float] = {}
    for col in columns:
        if col not in merged.columns:
            continue
        grouped = y.groupby(merged[col].astype(str).fillna("__NA__"))
        sizes = grouped.size()
        means = grouped.mean()
        keep = sizes >= 10
        if not keep.any():
            continue
        # Weighted mean distance from the base rate: 0 = no signal, 0.5 = the
        # column separates the classes completely.
        weights = sizes[keep] / sizes[keep].sum()
        scores[col] = float((weights * (means[keep] - y.mean()).abs()).sum())
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# --------------------------------------------------------------------------
# Training one feature set
# --------------------------------------------------------------------------


def train_one(
    *,
    merged: pd.DataFrame,
    spec: ha.FeatureSpec,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_summary: dict,
    seed: int,
    version: str,
    artifacts_root: Path,
    dataset_block: dict,
) -> dict:
    out_dir = artifacts_root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    X = merged[spec.columns]
    y = merged[ha.TARGET_COLUMN].to_numpy()
    groups = merged[ha.GROUP_COLUMN].to_numpy()

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    print(f"\n{'=' * 72}\n{version}  ({spec.name}: {len(spec.columns)} source columns)\n{'=' * 72}")
    print(f"  train {len(X_train)}   validation {len(X_val)}   test {len(X_test)}")
    print(
        f"  positives: train {int(y_train.sum())}  val {int(y_val.sum())}  "
        f"test {int(y_test.sum())}"
    )

    # -- fit the preprocessor on TRAIN ONLY ------------------------------
    preprocessor = ha.build_preprocessor(spec)
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)
    feature_names = ha.encoded_feature_names(preprocessor)
    print(f"  encoded columns: {len(feature_names)} (fitted on the training split only)")

    # -- model selection, grouped 5-fold on the training split -----------
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(cv.split(Xt_train, y_train, groups=groups[train_idx]))

    cv_results: dict[str, dict[str, float]] = {}
    for name, model in candidate_models(seed).items():
        auc = cross_val_score(model, Xt_train, y_train, cv=folds, scoring="roc_auc")
        ap = cross_val_score(
            model, Xt_train, y_train, cv=folds, scoring="average_precision"
        )
        cv_results[name] = {
            "roc_auc_mean": float(auc.mean()),
            "roc_auc_std": float(auc.std()),
            "pr_auc_mean": float(ap.mean()),
        }
        print(
            f"    {name:22} ROC-AUC {auc.mean():.4f} +/- {auc.std():.4f}"
            f"   PR-AUC {ap.mean():.4f}"
        )

    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    print(f"    -> selected {best_name}")

    # -- calibrate, using the same grouped folds -------------------------
    calibrated = CalibratedClassifierCV(
        candidate_models(seed)[best_name], method="isotonic", cv=folds
    )
    calibrated.fit(Xt_train, y_train)

    # -- threshold on the VALIDATION split -------------------------------
    val_prob = calibrated.predict_proba(Xt_val)[:, 1]
    threshold, criterion = choose_threshold(y_val, val_prob)
    validation_metrics = evaluate(y_val, val_prob, threshold)
    print(f"  threshold {threshold:.4f}  ({criterion})")

    # -- evaluate ONCE on the held-out test split ------------------------
    test_prob = calibrated.predict_proba(Xt_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)
    at = test_metrics["at_threshold"]
    print(
        f"  TEST  acc {at['accuracy']:.4f}  prec {at['precision']:.4f}  "
        f"rec {at['recall_sensitivity']:.4f}  F1 {at['f1']:.4f}  "
        f"ROC-AUC {test_metrics['roc_auc']:.4f}  PR-AUC {test_metrics['pr_auc']:.4f}"
    )
    print(f"  confusion {at['confusion_matrix']}")

    metrics = {
        "generated_by": "scripts/train_inhibitor_model.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "feature_set": spec.name,
        "split": split_summary,
        "cross_validation": cv_results,
        "selected_model": best_name,
        "threshold": {"value": threshold, "criterion": criterion},
        "validation": validation_metrics,
        "held_out_test": test_metrics,
        "baselines": {
            "majority_class_accuracy": float(1 - y_test.mean()),
            "note": (
                "At a ~17% positive rate a constant 'no inhibitor' answer scores "
                "~0.83 accuracy, which is why accuracy is reported alongside "
                "PR-AUC and recall rather than on its own."
            ),
        },
    }

    metadata = {
        "model_version": version,
        "model_type": f"CalibratedClassifierCV(isotonic) over {best_name}",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": dataset_block,
        "features": {
            "n": len(feature_names),
            "names": feature_names,
            "source_columns": spec.columns,
            "excluded_columns": ha.EXCLUDED_COLUMNS,
            **spec.as_dict(),
        },
        "split": split_summary,
        "calibration": {
            "method": "isotonic",
            "cv": "StratifiedGroupKFold(5) on mut_id",
            "fitted_on": "training split",
        },
        "threshold": {
            "value": threshold,
            "selected_on": "validation split",
            "criterion": criterion,
        },
        "test_metrics": test_metrics,
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "notebook_reference": "final(1).ipynb",
        "notebook_deviations": list(ha.NOTEBOOK_DEVIATIONS),
        "limitations": [
            "A record is one clinical report, and several reports may describe "
            "the same mutation. Splits are grouped on mut_id so a mutation never "
            "spans two partitions, but the genomic block is still repeated within "
            "a group.",
            f"{dataset_block['labels']['n_excluded_unlabelled']} MMC3 records "
            "without an explicit Yes/No inhibitor value were excluded. That "
            "exclusion is unlikely to be random, so metrics describe the reported "
            "subset of this dataset.",
            "uinhibitor (MMC2's curated inhibitor status) is excluded as a leakage "
            "feature; it reproduces the label almost exactly.",
            "No external validation cohort. Not clinically validated.",
        ],
    }

    joblib.dump(calibrated, out_dir / "model.joblib")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Background sample for SHAP/LIME, drawn from the TRAINING split only.
    rng = np.random.default_rng(seed)
    picks = rng.choice(
        len(Xt_train), size=min(200, len(Xt_train)), replace=False
    )
    np.save(out_dir / "background.npy", np.asarray(Xt_train)[picks])

    print(f"  wrote {out_dir}")
    return {"version": version, "metrics": metrics, "metadata": metadata}


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        choices=[*ha.FEATURE_SET_NAMES, "all"],
        default="all",
        help="Which block to train. Default trains genomic, clinical and merged.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--mmc2", default=None, help="Overrides MMC2_PATH")
    parser.add_argument("--mmc3", default=None, help="Overrides MMC3_PATH")
    parser.add_argument("--artifacts-dir", default=None, help="Overrides ml/artifacts")
    parser.add_argument(
        "--version-suffix",
        default="v1",
        help="Artifact version becomes mmc-<feature set>-<suffix>.",
    )
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    artifacts_root = Path(args.artifacts_dir or (REPO_ROOT / "ml" / "artifacts"))

    print("=" * 72)
    print("Hemophilia A inhibitor-risk training  (MMC2 + MMC3)")
    print("=" * 72)

    # -- 1. load, validate, merge ----------------------------------------
    p2 = Path(args.mmc2) if args.mmc2 else ha.mmc2_path()
    p3 = Path(args.mmc3) if args.mmc3 else ha.mmc3_path()
    merged, labels, merge_report, sources = ha.load_merged(p2, p3)

    print("\n[1] Sources")
    for report in sources:
        print(
            f"    {report.name:5} {report.n_rows:6} rows  {report.n_columns:3} cols  "
            f"{report.n_unique_mut_id:5} unique mut_id  "
            f"{report.n_duplicate_rows} duplicate rows  "
            f"{report.n_duplicate_mut_id} repeated mut_id  "
            f"{report.missing_values_total} missing values"
        )
    print(
        f"    labels    {labels.n_labelled} explicit Yes/No "
        f"({labels.n_positive} positive, {labels.positive_rate:.2%}); "
        f"{labels.n_excluded_unlabelled} excluded {labels.excluded_values}"
    )
    print("\n[2] Merge on mut_id")
    for key, value in merge_report.as_dict().items():
        print(f"    {key:28} {value}")

    # -- 2. GROUPED SPLIT, before anything is fitted ---------------------
    train_idx, val_idx, test_idx = grouped_split(
        merged, seed, args.test_size, args.val_size
    )
    group_summary = assert_no_group_overlap(merged, train_idx, val_idx, test_idx)
    split_summary = {
        "strategy": "GroupShuffleSplit on mut_id (80/20, then 80/20 of the remainder)",
        "seed": seed,
        "train": int(len(train_idx)),
        "validation": int(len(val_idx)),
        "test": int(len(test_idx)),
        "train_positive": int(merged[ha.TARGET_COLUMN].iloc[train_idx].sum()),
        "validation_positive": int(merged[ha.TARGET_COLUMN].iloc[val_idx].sum()),
        "test_positive": int(merged[ha.TARGET_COLUMN].iloc[test_idx].sum()),
        "unique_mutations": int(merged[ha.GROUP_COLUMN].nunique()),
        **group_summary,
    }
    print("\n[3] Grouped split (mut_id never crosses a boundary)")
    for key in (
        "train",
        "validation",
        "test",
        "unique_mutations",
        "train_mutations",
        "validation_mutations",
        "test_mutations",
        "overlaps",
    ):
        print(f"    {key:20} {split_summary[key]}")

    # -- 3. feature sets, required-ness judged on the TRAINING split -----
    specs = ha.build_feature_specs(merged, merged.iloc[train_idx])
    print("\n[4] Feature sets")
    for name, spec in specs.items():
        print(
            f"    {name:9} {len(spec.columns):3} columns "
            f"({len(spec.categorical)} categorical, {len(spec.numeric)} numeric); "
            f"dropped {list(spec.dropped) or 'none'}"
        )

    probe = leakage_probe(merged, [*specs["merged"].columns, "uinhibitor"])
    print("\n[5] Leakage probe (weighted |class rate - base rate|, top 6)")
    for col, score in list(probe.items())[:6]:
        flag = "  <-- EXCLUDED" if col in ha.EXCLUDED_COLUMNS else ""
        print(f"    {col:16} {score:.4f}{flag}")

    dataset_block = {
        "name": DATASET_NAME,
        "description": (
            "Hemophilia A supplementary tables MMC2 (mutations) and MMC3 "
            "(clinical records), joined on mut_id."
        ),
        "files": {
            "mmc2": {
                "path": str(p2.relative_to(REPO_ROOT)) if p2.is_relative_to(REPO_ROOT) else str(p2),
                "sha256": file_sha256(p2),
            },
            "mmc3": {
                "path": str(p3.relative_to(REPO_ROOT)) if p3.is_relative_to(REPO_ROOT) else str(p3),
                "sha256": file_sha256(p3),
            },
        },
        "join": {"key": ha.GROUP_COLUMN, "how": "inner", "genomic_rows_per_mutation": 1},
        "unit_of_observation": "one clinical record of one F8 mutation",
        "group_key": ha.GROUP_COLUMN,
        "label_column": ha.LABEL_COLUMN,
        "labels": labels.as_dict(),
        "merge": merge_report.as_dict(),
        "leakage_probe": probe,
    }

    wanted = ha.FEATURE_SET_NAMES if args.feature_set == "all" else (args.feature_set,)
    results = []
    for name in wanted:
        results.append(
            train_one(
                merged=merged,
                spec=specs[name],
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                split_summary=split_summary,
                seed=seed,
                version=f"{VERSION_PREFIX}-{name}-{args.version_suffix}",
                artifacts_root=artifacts_root,
                dataset_block=dataset_block,
            )
        )

    print("\n" + "=" * 72)
    print("Held-out test set summary (each split touched once)")
    print("=" * 72)
    header = f"{'version':22}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'ROC-AUC':>10}{'PR-AUC':>9}"
    print(header)
    for result in results:
        test = result["metrics"]["held_out_test"]
        at = test["at_threshold"]
        print(
            f"{result['version']:22}{at['accuracy']:>8.4f}{at['precision']:>8.4f}"
            f"{at['recall_sensitivity']:>8.4f}{at['f1']:>8.4f}"
            f"{test['roc_auc']:>10.4f}{test['pr_auc']:>9.4f}"
        )

    summary_path = artifacts_root / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_block,
                "split": split_summary,
                "models": {
                    r["version"]: r["metrics"]["held_out_test"] for r in results
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
