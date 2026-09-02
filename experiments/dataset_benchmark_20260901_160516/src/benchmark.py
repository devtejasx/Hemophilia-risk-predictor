"""Benchmark runner.

Protocol (identical for every dataset):

  1. Outer split 80/20, stratified. Grouped by mut_id where rows share an entity.
     The 20% test set is then untouched until step 7.
  2. 5-fold stratified (grouped) CV inside the 80% -> out-of-fold probabilities
     for all 8 base models. Deep models carve an extra 15% early-stopping split
     out of each fold's TRAIN part, never out of the fold's held-out part.
  3. Stacking meta-learner fitted on the OOF matrix.
  4. Weighted-ensemble weights from OOF ROC-AUC only.
  5. Calibration (sigmoid and isotonic) fitted on OOF only.
  6. Decision threshold from Youden's J on OOF only, then frozen.
  7. Refit base models on the full 80%, predict the held-out 20% once.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from config import N_FOLDS, SEED, TEST_SIZE
from models import (classical_zoo, deep_zoo, make_onehot_preprocessor,
                    make_ordinal_preprocessor, meta_learner)

warnings.filterwarnings("ignore")

BASE_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
               "DeepMLP", "ResidualMLP", "TabTransformer", "1D-CNN"]


# --------------------------------------------------------------------------
def metrics_at(y_true, prob, threshold) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def youden_threshold(y_true, prob) -> float:
    """Threshold maximising sensitivity + specificity - 1, on validation data only."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = thr[k]
    return float(min(max(t, 1e-6), 1 - 1e-6)) if np.isfinite(t) else 0.5


def _splitter(groups, n_splits):
    return (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            if groups is not None
            else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED))


def outer_split(X, y, groups):
    """80/20 stratified, group-aware. Returns index arrays."""
    if groups is None:
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, random_state=SEED, stratify=y)
        return idx_tr, idx_te
    # group-aware: one split off a 5-fold grouped stratified partition
    sgkf = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True,
                                random_state=SEED)
    idx_tr, idx_te = next(sgkf.split(X, y, groups))
    return idx_tr, idx_te


# --------------------------------------------------------------------------
def _prep_matrices(spec, Xa, Xb):
    """Fit both preprocessing paths on Xa, transform Xa and Xb."""
    oh = make_onehot_preprocessor(spec.numeric, spec.categorical)
    A_oh = oh.fit_transform(Xa)
    B_oh = oh.transform(Xb)

    od = make_ordinal_preprocessor(spec.numeric, spec.categorical)
    A_od = od.fit_transform(Xa)
    B_od = od.transform(Xb)

    ncat = len(spec.categorical)
    cards = []
    if ncat:
        enc = od.named_transformers_["cat"].named_steps["encode"]
        cards = [len(c) for c in enc.categories_]

    def split_ord(M):
        cat = M[:, :ncat].astype(np.int64) + 1          # -1 unknown -> 0
        num = M[:, ncat:].astype(np.float32)
        return cat, num

    return {
        "oh": (np.asarray(A_oh, dtype=np.float32), np.asarray(B_oh, dtype=np.float32)),
        "ord": (split_ord(np.asarray(A_od)), split_ord(np.asarray(B_od))),
        "cards": cards,
        "n_numeric": A_od.shape[1] - ncat,
        "feature_names": [str(n) for n in oh.get_feature_names_out()],
        "oh_transformer": oh,
    }


def _fit_predict_all(spec, Xtr, ytr, Xte, loss="weighted_bce", seed_offset=0):
    """Fit all 8 base models on (Xtr,ytr); return probabilities for Xte."""
    from sklearn.model_selection import train_test_split as tts
    mats = _prep_matrices(spec, Xtr, Xte)
    A_oh, B_oh = mats["oh"]
    (Acat, Anum), (Bcat, Bnum) = mats["ord"]

    # early-stopping split carved out of TRAIN only
    idx = np.arange(len(ytr))
    try:
        i_fit, i_es = tts(idx, test_size=0.15, random_state=SEED + seed_offset, stratify=ytr)
    except ValueError:
        i_fit, i_es = idx, idx

    out, info = {}, {}
    for name, m in classical_zoo().items():
        m.fit(A_oh, ytr)
        t = time.perf_counter()
        out[name] = m.predict_proba(B_oh)
        info[name] = {"fit_seconds": m.fit_seconds,
                      "inference_seconds": time.perf_counter() - t}

    dz = deep_zoo(A_oh.shape[1], mats["cards"], mats["n_numeric"], loss=loss)
    for name, m in dz.items():
        if m.needs_ordinal:
            Xa, Xb = (Acat[i_fit], Anum[i_fit]), (Acat[i_es], Anum[i_es])
            Xpred = (Bcat, Bnum)
        else:
            Xa, Xb = A_oh[i_fit], A_oh[i_es]
            Xpred = B_oh
        m.fit(Xa, ytr[i_fit], Xb, ytr[i_es])
        t = time.perf_counter()
        out[name] = m.predict_proba(Xpred)
        info[name] = {"fit_seconds": m.fit_seconds,
                      "inference_seconds": time.perf_counter() - t,
                      "epochs_completed": m.record.epochs_completed,
                      "best_epoch": m.record.best_epoch,
                      "best_val_auc": m.record.best_val_auc,
                      "best_val_loss": m.record.best_val_loss,
                      "early_stopped": m.record.converged}
    return out, info, mats


def run_dataset(spec, log) -> dict:
    """Full protocol for one dataset. Returns a result bundle."""
    X, y = spec.X.reset_index(drop=True), spec.y.reset_index(drop=True).to_numpy()
    groups = spec.groups.reset_index(drop=True).to_numpy() if spec.groups is not None else None

    idx_tr, idx_te = outer_split(X, y, groups)
    Xtv, ytv = X.iloc[idx_tr], y[idx_tr]
    Xte, yte = X.iloc[idx_te], y[idx_te]
    gtv = groups[idx_tr] if groups is not None else None

    log(f"  split: trainval={len(ytv)} (pos {int(ytv.sum())}) "
        f"test={len(yte)} (pos {int(yte.sum())})"
        + (f" | grouped by mut_id, {len(set(groups[idx_tr]) & set(groups[idx_te]))} "
           f"groups shared between train and test" if groups is not None else ""))

    # ---- 2. OOF predictions inside trainval -----------------------------
    oof = {m: np.zeros(len(ytv)) for m in BASE_MODELS}
    fold_info: list[dict] = []
    splitter = _splitter(gtv, N_FOLDS)
    for k, (i_in, i_out) in enumerate(splitter.split(Xtv, ytv, gtv), 1):
        t0 = time.perf_counter()
        preds, info, _ = _fit_predict_all(
            spec, Xtv.iloc[i_in], ytv[i_in], Xtv.iloc[i_out], seed_offset=k)
        for m in BASE_MODELS:
            oof[m][i_out] = preds[m]
        fold_info.append({"fold": k, "n_train": len(i_in), "n_out": len(i_out),
                          "seconds": time.perf_counter() - t0, "models": info})
        log(f"    fold {k}/{N_FOLDS} done in {time.perf_counter()-t0:5.1f}s")

    oof_auc = {m: float(roc_auc_score(ytv, oof[m])) for m in BASE_MODELS}
    log("    OOF ROC-AUC: " + ", ".join(f"{m}={v:.3f}" for m, v in oof_auc.items()))

    # ---- 3/4. ensembles built from OOF only -----------------------------
    OOF = np.column_stack([oof[m] for m in BASE_MODELS])
    meta = meta_learner().fit(OOF, ytv)
    oof["StackingEnsemble"] = meta.predict_proba(OOF)[:, 1]

    w = np.array([max(oof_auc[m] - 0.5, 0.0) for m in BASE_MODELS])
    w = w / w.sum() if w.sum() > 0 else np.ones(len(BASE_MODELS)) / len(BASE_MODELS)
    weights = dict(zip(BASE_MODELS, w.round(5).tolist()))
    oof["WeightedEnsemble"] = OOF @ w

    all_models = BASE_MODELS + ["StackingEnsemble", "WeightedEnsemble"]

    # ---- 5/6. calibration + threshold, from OOF only --------------------
    calibrators, thresholds = {}, {}
    for m in all_models:
        p = oof[m]
        sig = LogisticRegression(max_iter=1000).fit(p.reshape(-1, 1), ytv)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, ytv)
        b_raw = brier_score_loss(ytv, p)
        b_sig = brier_score_loss(ytv, sig.predict_proba(p.reshape(-1, 1))[:, 1])
        b_iso = brier_score_loss(ytv, iso.predict(p))
        best = min([("raw", b_raw), ("sigmoid", b_sig), ("isotonic", b_iso)],
                   key=lambda kv: kv[1])
        calibrators[m] = {"sigmoid": sig, "isotonic": iso,
                          "chosen": best[0],
                          "oof_brier": {"raw": b_raw, "sigmoid": b_sig, "isotonic": b_iso}}
        thresholds[m] = youden_threshold(ytv, p)

    # ---- 7. refit on full trainval, predict the held-out test once -------
    log("    refitting on full trainval and scoring the held-out test set")
    test_pred, test_info, mats = _fit_predict_all(spec, Xtv, ytv, Xte, seed_offset=99)
    TE = np.column_stack([test_pred[m] for m in BASE_MODELS])
    test_pred["StackingEnsemble"] = meta.predict_proba(TE)[:, 1]
    test_pred["WeightedEnsemble"] = TE @ w

    results = {}
    for m in all_models:
        p_raw = test_pred[m]
        cal = calibrators[m]
        p_sig = cal["sigmoid"].predict_proba(p_raw.reshape(-1, 1))[:, 1]
        p_iso = cal["isotonic"].predict(p_raw)
        p_cal = {"raw": p_raw, "sigmoid": p_sig, "isotonic": p_iso}[cal["chosen"]]

        r = metrics_at(yte, p_raw, thresholds[m])
        r["brier_calibrated"] = float(brier_score_loss(yte, p_cal))
        r["calibration_method"] = cal["chosen"]
        r["oof_roc_auc"] = oof_auc.get(m, float(roc_auc_score(ytv, oof[m])))
        r["fit_seconds"] = test_info.get(m, {}).get("fit_seconds")
        r["inference_seconds"] = test_info.get(m, {}).get("inference_seconds")
        for extra in ("epochs_completed", "best_epoch", "best_val_auc", "early_stopped"):
            if m in test_info and extra in test_info[m]:
                r[extra] = test_info[m][extra]
        results[m] = r

    return {
        "dataset": spec.name,
        "n_train": int(len(ytv)), "n_test": int(len(yte)),
        "n_features_encoded": int(mats["oh"][0].shape[1]),
        "test_pos": int(yte.sum()), "test_neg": int(len(yte) - yte.sum()),
        "train_pos": int(ytv.sum()), "train_neg": int(len(ytv) - ytv.sum()),
        "results": results,
        "oof_auc": oof_auc,
        "ensemble_weights": weights,
        "thresholds": thresholds,
        "calibration": {m: calibrators[m]["oof_brier"] | {"chosen": calibrators[m]["chosen"]}
                        for m in all_models},
        "fold_info": fold_info,
        "feature_names": mats["feature_names"],
        "y_test": yte.tolist(),
        "test_probabilities": {m: test_pred[m].tolist() for m in all_models},
        "oof_probabilities": {m: oof[m].tolist() for m in all_models},
        "y_trainval": ytv.tolist(),
    }
