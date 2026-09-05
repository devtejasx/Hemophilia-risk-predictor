"""Main entry point: run the benchmark over every dataset and persist everything."""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DL, EXP_DIR, FOCAL, N_FOLDS, SEED, TEST_SIZE
from data import build_all
from benchmark import BASE_MODELS, run_dataset

LOG = EXP_DIR / "logs" / "benchmark.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def environment() -> dict:
    import catboost, lightgbm, shap, sklearn, torch, xgboost
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
        "torch": torch.__version__, "shap": shap.__version__,
        "seed": SEED, "n_folds": N_FOLDS, "test_size": TEST_SIZE,
        "dl_settings": DL, "focal_loss_params": FOCAL,
        "imbalance_strategy": (
            "classical: class_weight/scale_pos_weight = n_neg/n_pos; "
            "deep: BCEWithLogitsLoss(pos_weight=n_neg/n_pos). Identical across "
            "all datasets. SMOTE deliberately NOT used."),
        "meta_learner": "LogisticRegression(class_weight='balanced') on OOF probabilities",
        "weighted_ensemble_rule": "w_i proportional to max(OOF ROC-AUC_i - 0.5, 0), normalised",
        "threshold_rule": "Youden's J maximised on OOF predictions, then frozen",
        "calibration_rule": "sigmoid and isotonic fitted on OOF; lowest OOF Brier chosen",
    }


def main() -> int:
    t_start = time.perf_counter()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")

    log("=" * 78)
    log("Hemophilia A inhibitor-risk : dataset benchmark")
    log("=" * 78)

    env = environment()
    (EXP_DIR / "configs" / "environment.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8")
    log(f"env: python {env['python']}, sklearn {env['scikit_learn']}, torch {env['torch']}")

    log("\nBuilding datasets ...")
    specs = build_all()

    # ---- dataset overview + leakage audit -------------------------------
    overview = pd.DataFrame([s.summary() for s in specs])
    overview["valid_for_supervised"] = [s.valid_for_supervised for s in specs]
    overview["rejection_reason"] = [s.rejection_reason for s in specs]
    overview.to_csv(EXP_DIR / "tables" / "dataset_overview.csv", index=False)

    audits = []
    for s in specs:
        a = s.audit.copy(); a.insert(0, "dataset", s.name)
        audits.append(a)
    audit_all = pd.concat(audits, ignore_index=True)
    audit_all.to_csv(EXP_DIR / "tables" / "leakage_audit.csv", index=False)

    meta = {s.name: {
        "source_file": s.source_file, "sha256": s.sha256,
        "prediction_unit": s.unit, "unit_rationale": s.unit_rationale,
        "target_name": s.target_name, "target_definition": s.target_definition,
        "valid_for_supervised": s.valid_for_supervised,
        "rejection_reason": s.rejection_reason,
        "notes": s.notes,
        "safe_features": {"categorical": s.categorical, "numeric": s.numeric},
        "audit_counts": s.audit.classification.value_counts().to_dict(),
    } for s in specs}
    (EXP_DIR / "configs" / "datasets.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    for s in specs:
        flag = "VALID" if s.valid_for_supervised else "REJECTED"
        log(f"  {s.name:24} [{flag:8}] unit={s.unit:42} n={len(s.y):5} "
            f"pos={int(s.y.sum()):4} ({s.y.mean():5.1%}) feats={s.X.shape[1]}")

    # ---- run ------------------------------------------------------------
    bundles = {}
    for s in specs:
        log(f"\n--- {s.name} ---")
        if not s.valid_for_supervised:
            log(f"  REJECTED for the primary benchmark: {s.rejection_reason[:120]}...")
            log("  running leakage demonstration only (4 outcome-derived columns)")
        t0 = time.perf_counter()
        try:
            bundles[s.name] = run_dataset(s, log)
            bundles[s.name]["valid_for_supervised"] = s.valid_for_supervised
            bundles[s.name]["elapsed_seconds"] = time.perf_counter() - t0
            log(f"  {s.name} finished in {(time.perf_counter()-t0)/60:.1f} min")
        except Exception as exc:            # recorded, never silently skipped
            log(f"  !! {s.name} FAILED: {type(exc).__name__}: {exc}")
            bundles[s.name] = {"dataset": s.name, "failed": True,
                               "error": f"{type(exc).__name__}: {exc}"}

    (EXP_DIR / "metrics" / "raw_results.json").write_text(
        json.dumps(bundles, indent=2), encoding="utf-8")

    # ---- tidy metric table ---------------------------------------------
    rows = []
    for name, b in bundles.items():
        if b.get("failed"):
            continue
        for model, r in b["results"].items():
            rows.append({"dataset": name, "model": model,
                         "valid_for_supervised": b["valid_for_supervised"],
                         "n_train": b["n_train"], "n_test": b["n_test"],
                         "n_features_encoded": b["n_features_encoded"],
                         "test_pos": b["test_pos"], "test_neg": b["test_neg"], **r})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(EXP_DIR / "metrics" / "all_metrics.csv", index=False)

    for metric in ["roc_auc", "pr_auc", "f1", "recall_sensitivity",
                   "specificity", "brier"]:
        piv = metrics.pivot_table(index="dataset", columns="model",
                                  values=metric, aggfunc="first")
        cols = [c for c in BASE_MODELS + ["StackingEnsemble", "WeightedEnsemble"]
                if c in piv.columns]
        piv[cols].round(4).to_csv(EXP_DIR / "tables" / f"model_comparison_{metric}.csv")

    # ---- predictions for downstream plots -------------------------------
    for name, b in bundles.items():
        if b.get("failed"):
            continue
        pd.DataFrame({"y_true": b["y_test"], **b["test_probabilities"]}).to_csv(
            EXP_DIR / "predictions" / f"test_{name}.csv", index=False)
        pd.DataFrame({"y_true": b["y_trainval"], **b["oof_probabilities"]}).to_csv(
            EXP_DIR / "predictions" / f"oof_{name}.csv", index=False)

    log(f"\nTotal wall time {(time.perf_counter()-t_start)/60:.1f} min")
    log(f"Wrote {EXP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
