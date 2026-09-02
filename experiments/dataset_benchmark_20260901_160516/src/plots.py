"""Figures: ROC, PR, calibration, confusion matrices, cross-dataset comparison."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from config import EXP_DIR

PLOTS = EXP_DIR / "plots"
PALETTE = ["#0F6E72", "#A3312B", "#8A5D0B", "#2C6642", "#4C5CA8", "#8C4A76",
           "#3D7A9E", "#7A5C2E", "#555F6B", "#B5651D"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

ALL_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "DeepMLP",
              "ResidualMLP", "TabTransformer", "1D-CNN",
              "StackingEnsemble", "WeightedEnsemble"]


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(EXP_DIR / "predictions" / f"test_{name}.csv")


def roc_pr_for_dataset(name: str) -> None:
    df = _load(name)
    y = df["y_true"].to_numpy()
    models = [m for m in ALL_MODELS if m in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, m in enumerate(models):
        p = df[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.4,
                     label=f"{m} ({roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=PALETTE[i % len(PALETTE)], lw=1.4,
                     label=f"{m} ({average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title=f"ROC - {name} (held-out test)")
    axes[1].axhline(y.mean(), ls="--", c="k", lw=0.8,
                    label=f"base rate ({y.mean():.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title=f"Precision-Recall - {name} (held-out test)")
    for ax in axes:
        ax.legend(fontsize=6.2, loc="lower left" if ax is axes[0] else "upper right")
    fig.savefig(PLOTS / f"roc_pr_{name}.png", bbox_inches="tight")
    plt.close(fig)


def confusion_grid(name: str, thresholds: dict) -> None:
    df = _load(name)
    y = df["y_true"].to_numpy()
    models = [m for m in ALL_MODELS if m in df.columns]
    ncol = 5
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, models):
        pred = (df[m].to_numpy() >= thresholds.get(m, 0.5)).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_title(m, fontsize=7.5)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=6.5)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=6.5)
        ax.grid(False)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices at frozen thresholds - {name}", fontsize=10)
    fig.savefig(PLOTS / f"confusion_{name}.png", bbox_inches="tight")
    plt.close(fig)


def calibration_plot(name: str, bundle: dict, top_models: list[str]) -> None:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    test = _load(name)
    oof = pd.read_csv(EXP_DIR / "predictions" / f"oof_{name}.csv")
    y_te = test["y_true"].to_numpy()
    y_tv = oof["y_true"].to_numpy()

    fig, axes = plt.subplots(1, len(top_models), figsize=(4.2 * len(top_models), 4),
                             squeeze=False)
    for ax, m in zip(axes[0], top_models):
        p_raw = test[m].to_numpy()
        o = oof[m].to_numpy()
        sig = LogisticRegression(max_iter=1000).fit(o.reshape(-1, 1), y_tv)
        iso = IsotonicRegression(out_of_bounds="clip").fit(o, y_tv)
        variants = {
            "raw": p_raw,
            "sigmoid": sig.predict_proba(p_raw.reshape(-1, 1))[:, 1],
            "isotonic": iso.predict(p_raw),
        }
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        for i, (lab, p) in enumerate(variants.items()):
            nb = min(10, max(3, int(len(y_te) / 40)))
            try:
                frac, mean_pred = calibration_curve(y_te, p, n_bins=nb, strategy="quantile")
            except ValueError:
                continue
            from sklearn.metrics import brier_score_loss
            ax.plot(mean_pred, frac, "o-", ms=3.5, lw=1.3,
                    color=PALETTE[i], label=f"{lab} (Brier {brier_score_loss(y_te, p):.4f})")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
               title=f"{m}\n{name}", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.savefig(PLOTS / f"calibration_{name}.png", bbox_inches="tight")
    plt.close(fig)


def cross_dataset_bars(metrics: pd.DataFrame) -> None:
    valid = metrics[metrics.valid_for_supervised]
    for metric, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                          ("f1", "F1"), ("recall_sensitivity", "Sensitivity")]:
        piv = valid.pivot_table(index="model", columns="dataset",
                                values=metric, aggfunc="first")
        piv = piv.reindex([m for m in ALL_MODELS if m in piv.index])
        fig, ax = plt.subplots(figsize=(10, 4.4))
        x = np.arange(len(piv))
        w = 0.8 / max(len(piv.columns), 1)
        for i, ds in enumerate(piv.columns):
            ax.bar(x + i * w, piv[ds].to_numpy(), w, label=ds,
                   color=PALETTE[i % len(PALETTE)])
        if metric in ("roc_auc",):
            ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
        ax.set_xticks(x + 0.4 - w / 2, piv.index, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel=label, title=f"{label} by model and dataset (held-out test)")
        ax.legend(fontsize=7.5)
        fig.savefig(PLOTS / f"cross_dataset_{metric}.png", bbox_inches="tight")
        plt.close(fig)


def leakage_figure(metrics: pd.DataFrame) -> None:
    """The point of the rejected dataset: what leakage buys you."""
    best = (metrics.sort_values("roc_auc", ascending=False)
                   .groupby("dataset").first().reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#A3312B" if not v else "#0F6E72" for v in best.valid_for_supervised]
    ax.barh(best.dataset, best.roc_auc, color=colors)
    for i, (v, m) in enumerate(zip(best.roc_auc, best.model)):
        ax.text(v + 0.01, i, f"{v:.3f}  ({m})", va="center", fontsize=8)
    ax.axvline(0.5, ls="--", c="k", lw=0.8)
    ax.set(xlim=(0, 1.15), xlabel="Best ROC-AUC on held-out test",
           title="Best model per dataset\n(red = rejected: outcome-derived features)")
    fig.savefig(PLOTS / "leakage_comparison.png", bbox_inches="tight")
    plt.close(fig)


def dl_training_curves(bundles: dict) -> None:
    """Only plots what actually trained; silent if no history was recorded."""
    rows = []
    for name, b in bundles.items():
        if b.get("failed"):
            continue
        for m, r in b["results"].items():
            if "epochs_completed" in r:
                rows.append({"dataset": name, "model": m,
                             "epochs": r["epochs_completed"],
                             "best_epoch": r.get("best_epoch"),
                             "best_val_auc": r.get("best_val_auc"),
                             "early_stopped": r.get("early_stopped")})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(EXP_DIR / "tables" / "deep_learning_training.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ds in enumerate(df.dataset.unique()):
        sub = df[df.dataset == ds]
        ax.scatter(sub.model, sub.best_epoch, s=42, label=ds,
                   color=PALETTE[i % len(PALETTE)])
    ax.set(ylabel="Best epoch (validation ROC-AUC)",
           title="Deep models: epoch of best validation AUC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7.5)
    fig.savefig(PLOTS / "dl_best_epoch.png", bbox_inches="tight")
    plt.close(fig)


def make_all() -> None:
    bundles = json.loads((EXP_DIR / "metrics" / "raw_results.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(EXP_DIR / "metrics" / "all_metrics.csv")

    for name, b in bundles.items():
        if b.get("failed"):
            continue
        roc_pr_for_dataset(name)
        confusion_grid(name, b["thresholds"])
        top = (metrics[metrics.dataset == name]
               .sort_values("roc_auc", ascending=False).model.head(2).tolist())
        calibration_plot(name, b, top)
    cross_dataset_bars(metrics)
    leakage_figure(metrics)
    dl_training_curves(bundles)
    print(f"figures written to {PLOTS}")


if __name__ == "__main__":
    make_all()
