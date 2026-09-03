"""Train and evaluate the CHAMP inhibitor-risk model.

    python scripts/train_champ.py [--version champ-v1] [--seed 42]

Pipeline order matters and is enforced here:

    load -> validate -> normalise -> label
         -> SPLIT FIRST
         -> fit preprocessor on TRAIN ONLY
         -> model selection by cross-validated ROC-AUC (SMOTE inside folds)
         -> calibrate on train
         -> choose threshold on a validation split
         -> evaluate ONCE on the held-out test set
         -> write artifacts + metadata + metrics

Everything written to metrics.json comes from this run. Nothing is copied from
a previous report or hand-edited.
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
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from ml.preprocessing import champ  # noqa: E402

PREPROCESSING_VERSION = "champ-preprocessing-1"


def candidate_models(seed: int) -> dict[str, object]:
    """Model families to compare. Class weighting handles the 20% positive rate
    without resampling; SMOTE is evaluated separately, inside CV folds only."""
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
    """Pick the probability cut-off that maximises F1 on the validation split.

    F1 rather than accuracy: at a 20% positive rate, a model that predicts
    "no inhibitor" for everyone scores 80% accuracy and is useless. The choice
    is recorded in metadata so nothing downstream has to assume 0.5.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more element than thresholds
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), "max F1 on the validation split"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="champ-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--data", default=str(champ.DEFAULT_CHAMP_PATH))
    args = parser.parse_args()

    seed = args.seed
    data_path = Path(args.data)
    out_dir = REPO_ROOT / "ml" / "artifacts" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CHAMP inhibitor-risk training")
    print("=" * 72)

    # -- 1. load / validate / label -------------------------------------
    X_all, y_all, labels, validation = champ.load_champ_features(data_path)
    print(f"\n[1] Dataset          {data_path}")
    print(f"    rows in file     {validation.n_rows}")
    print(f"    labelled         {labels.n_labelled}")
    print(f"    positive         {labels.n_positive} ({labels.positive_rate:.2%})")
    print(f"    excluded         {labels.n_excluded_unlabelled} {labels.excluded_values}")

    # -- 2. SPLIT FIRST --------------------------------------------------
    X_fit, X_test, y_fit, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=seed, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_fit, y_fit, test_size=args.val_size, random_state=seed, stratify=y_fit
    )
    print(f"\n[2] Split (before any fitting)")
    print(f"    train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")

    # -- 3. fit preprocessor on TRAIN ONLY -------------------------------
    preprocessor = champ.build_preprocessor()
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)
    feature_names = champ.encoded_feature_names(preprocessor)
    print(f"\n[3] Preprocessor fitted on {len(X_train)} training rows only")
    print(f"    encoded features {len(feature_names)}")

    # -- 4. model selection by cross-validated ROC-AUC -------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    print("\n[4] Model comparison (5-fold CV on the training split)")
    cv_results: dict[str, dict[str, float]] = {}
    for name, model in candidate_models(seed).items():
        auc = cross_val_score(model, Xt_train, y_train, cv=cv, scoring="roc_auc")
        ap = cross_val_score(
            model, Xt_train, y_train, cv=cv, scoring="average_precision"
        )
        cv_results[name] = {
            "roc_auc_mean": float(auc.mean()),
            "roc_auc_std": float(auc.std()),
            "average_precision_mean": float(ap.mean()),
        }
        print(f"    {name:22} ROC-AUC {auc.mean():.4f} +/- {auc.std():.4f}"
              f"   PR-AUC {ap.mean():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    print(f"    -> selected: {best_name}")

    # -- 5. calibrate ----------------------------------------------------
    base = candidate_models(seed)[best_name]
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(Xt_train, y_train)
    print(f"\n[5] Calibrated with isotonic regression (5-fold, training split)")

    # -- 6. threshold on the VALIDATION split ----------------------------
    val_prob = calibrated.predict_proba(Xt_val)[:, 1]
    threshold, criterion = choose_threshold(y_val.to_numpy(), val_prob)
    print(f"\n[6] Threshold {threshold:.4f}  ({criterion})")

    # -- 7. evaluate ONCE on the held-out test set -----------------------
    test_prob = calibrated.predict_proba(Xt_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    metrics = {
        "generated_by": "scripts/train_champ.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
            "test_positive": int(y_test.sum()),
        },
        "cross_validation": cv_results,
        "selected_model": best_name,
        "threshold": {"value": threshold, "criterion": criterion},
        "held_out_test": {
            "roc_auc": float(roc_auc_score(y_test, test_prob)),
            "average_precision": float(average_precision_score(y_test, test_prob)),
            "brier_score": float(brier_score_loss(y_test, test_prob)),
            "at_threshold": {
                "precision": float(precision_score(y_test, test_pred, zero_division=0)),
                "recall": float(recall_score(y_test, test_pred, zero_division=0)),
                "f1": float(f1_score(y_test, test_pred, zero_division=0)),
                "confusion_matrix": {
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
                },
            },
        },
        "baselines": {
            "majority_class_accuracy": float(1 - y_test.mean()),
            "note": (
                "At a ~20% positive rate a constant 'no inhibitor' prediction "
                "scores ~0.80 accuracy, which is why accuracy is not reported "
                "as a headline metric."
            ),
        },
    }

    print("\n[7] Held-out test set (touched once)")
    ht = metrics["held_out_test"]
    print(f"    ROC-AUC          {ht['roc_auc']:.4f}")
    print(f"    PR-AUC           {ht['average_precision']:.4f}")
    print(f"    Brier            {ht['brier_score']:.4f}")
    print(f"    precision        {ht['at_threshold']['precision']:.4f}")
    print(f"    recall           {ht['at_threshold']['recall']:.4f}")
    print(f"    F1               {ht['at_threshold']['f1']:.4f}")
    print(f"    confusion        {ht['at_threshold']['confusion_matrix']}")

    # -- 8. write artifacts ----------------------------------------------
    metadata = {
        "model_version": args.version,
        "model_type": f"CalibratedClassifierCV(isotonic) over {best_name}",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": {
            "name": "CHAMP",
            "description": "CDC Hemophilia A Mutation Project variant registry",
            "file": str(data_path.relative_to(REPO_ROOT)),
            "file_sha256": file_sha256(data_path),
            "unit_of_observation": "F8 variant (NOT an individual patient)",
            "label_column": champ.LABEL_COLUMN,
            **labels.as_dict(),
        },
        "features": {
            "n": len(feature_names),
            "names": feature_names,
            "source_columns": champ.FEATURE_COLUMNS,
            "excluded_columns": champ.EXCLUDED_COLUMNS,
        },
        "calibration": {"method": "isotonic", "cv": 5, "fitted_on": "training split"},
        "threshold": {
            "value": threshold,
            "selected_on": "validation split",
            "criterion": criterion,
        },
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "CHAMP rows are F8 variants, not patients. Predictions are "
            "variant-attributable and are not an individual's probability.",
            f"{labels.n_excluded_unlabelled} rows without a reported inhibitor "
            "history were excluded. That exclusion is very unlikely to be "
            "random, so metrics describe the reported subset of this registry.",
            "Reference Number (source publication) was excluded as a leakage "
            "feature; some studies are 100% inhibitor-positive by selection.",
            "No external validation cohort. Not clinically validated.",
        ],
    }

    joblib.dump(calibrated, out_dir / "model.joblib")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    # Background sample for SHAP/LIME, drawn from the TRAINING split only.
    np.save(out_dir / "background.npy", Xt_train[
        np.random.default_rng(seed).choice(
            len(Xt_train), size=min(200, len(Xt_train)), replace=False
        )
    ])

    print(f"\n[8] Wrote {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"      {f.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
