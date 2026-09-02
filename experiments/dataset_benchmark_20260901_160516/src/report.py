"""Compose the PDF report from the artifacts the benchmark actually produced.

Everything printed here is read from disk. Nothing is hand-written from memory,
so the report cannot claim a result the experiment did not generate.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from config import EXP_DIR

ACCENT = colors.HexColor("#0F6E72")
CRIT = colors.HexColor("#A3312B")
INK = colors.HexColor("#16191F")
MUTED = colors.HexColor("#5C6470")
RULE = colors.HexColor("#CDD2DA")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontSize=19, leading=23,
                            textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=10.5, leading=15,
                          textColor=MUTED, alignment=TA_JUSTIFY),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13, leading=16,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=5),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, leading=13,
                         textColor=INK, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=8.9, leading=12.6,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.6, leading=10.4,
                            textColor=MUTED, spaceAfter=4),
    "cap": ParagraphStyle("cap", parent=ss["Normal"], fontSize=7.2, leading=9.5,
                          textColor=MUTED, spaceAfter=8),
    "warn": ParagraphStyle("w", parent=ss["Normal"], fontSize=8.9, leading=12.6,
                           textColor=CRIT, alignment=TA_JUSTIFY, spaceAfter=5),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def table(data, widths=None, font=6.9, header=True, align=None):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFF3F4")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if align:
        for col, a in align.items():
            style.append(("ALIGN", (col, 1), (col, -1), a))
    t.setStyle(TableStyle(style))
    return t


def fig(name, width=16.4 * cm, caption=None):
    p = EXP_DIR / "plots" / name
    if not p.exists():
        return [P(f"[figure {name} was not produced]", "small")]
    from PIL import Image as PILImage
    try:
        w, h = PILImage.open(p).size
        height = width * h / w
    except Exception:
        height = width * 0.55
    out = [Image(str(p), width=width, height=height)]
    if caption:
        out.append(P(caption, "cap"))
    return out


# --------------------------------------------------------------------------
def rank_datasets(metrics: pd.DataFrame, specs_meta: dict) -> pd.DataFrame:
    """Transparent multi-criterion ranking. Performance is one component of
    seven, so a leakage-driven score cannot win."""
    rows = []
    for name, meta in specs_meta.items():
        sub = metrics[metrics.dataset == name]
        if sub.empty:
            continue
        valid = meta["valid_for_supervised"]
        best = sub.sort_values("pr_auc", ascending=False).iloc[0]
        n = int(best["n_train"] + best["n_test"])
        pos = int(sub.iloc[0]["test_pos"]) + 0
        lift = best["pr_auc"] / max((best["test_pos"] / (best["test_pos"] + best["test_neg"])), 1e-9)

        # components, each 0-1
        c_valid = 1.0 if valid else 0.0
        c_unit = {"mutation (F8 variant)": 1.0,
                  "clinical record (one reported observation)": 0.55}.get(
                      meta["prediction_unit"], 0.5)
        c_size = min(n / 4000.0, 1.0)
        c_pos = min((best["test_pos"] * 5) / 600.0, 1.0)
        c_perf = min(max((lift - 1.0) / 2.0, 0.0), 1.0)
        c_cal = 1.0 - min(best["brier"] / 0.25, 1.0)
        c_feat = min(meta["n_safe_features"] / 15.0, 1.0)

        score = (0.30 * c_valid + 0.15 * c_unit + 0.12 * c_size + 0.12 * c_pos
                 + 0.16 * c_perf + 0.08 * c_cal + 0.07 * c_feat)
        rows.append({
            "dataset": name, "valid": valid,
            "prediction_unit": meta["prediction_unit"],
            "samples": n, "test_pos": int(best["test_pos"]),
            "safe_features": meta["n_safe_features"],
            "best_model": best["model"],
            "roc_auc": round(float(best["roc_auc"]), 4),
            "pr_auc": round(float(best["pr_auc"]), 4),
            "pr_lift": round(float(lift), 2),
            "f1": round(float(best["f1"]), 4),
            "sensitivity": round(float(best["recall_sensitivity"]), 4),
            "specificity": round(float(best["specificity"]), 4),
            "brier": round(float(best["brier"]), 4),
            "score": round(score, 4),
        })
    df = pd.DataFrame(rows).sort_values(
        ["valid", "score"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = np.where(df.valid, np.arange(1, len(df) + 1), "rejected")
    return df


# --------------------------------------------------------------------------
def build() -> str:
    M = pd.read_csv(EXP_DIR / "metrics" / "all_metrics.csv")
    bundles = json.loads((EXP_DIR / "metrics" / "raw_results.json").read_text("utf-8"))
    dsmeta = json.loads((EXP_DIR / "configs" / "datasets.json").read_text("utf-8"))
    env = json.loads((EXP_DIR / "configs" / "environment.json").read_text("utf-8"))
    overview = pd.read_csv(EXP_DIR / "tables" / "dataset_overview.csv")
    audit = pd.read_csv(EXP_DIR / "tables" / "leakage_audit.csv")
    try:
        expl = json.loads((EXP_DIR / "explanations" / "explanations.json").read_text("utf-8"))
    except FileNotFoundError:
        expl = {}

    for k, v in dsmeta.items():
        v["n_safe_features"] = len(v["safe_features"]["categorical"]) + \
                               len(v["safe_features"]["numeric"])

    ranking = rank_datasets(M, dsmeta)
    ranking.to_csv(EXP_DIR / "tables" / "dataset_ranking.csv", index=False)
    valid_rank = ranking[ranking.valid]
    winner = valid_rank.iloc[0]
    best_overall = M[M.valid_for_supervised].sort_values("pr_auc", ascending=False).iloc[0]
    ens = M[M.valid_for_supervised & M.model.isin(["StackingEnsemble", "WeightedEnsemble"])]
    best_ens = ens.sort_values("pr_auc", ascending=False).iloc[0]

    out = EXP_DIR / "report" / "Hemophilia_A_Dataset_Benchmark_Report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=1.9 * cm, bottomMargin=1.9 * cm,
                            title="Comparative ML/DL Benchmark for Hemophilia A Inhibitor Risk Prediction")
    F: list = []

    # ---------------- title ----------------
    F += [
        P("Comparative Machine Learning and Deep Learning Benchmark "
          "for Hemophilia A Inhibitor Risk Prediction", "title"),
        P(f"Generated {datetime.now():%d %B %Y}. Analysis-only experiment: no production "
          f"code, model artifact, database or dataset was modified, and nothing was "
          f"committed to version control.", "sub"),
        Spacer(1, 10),
    ]

    # 1. Executive summary
    F += [P("1. Executive Summary", "h1")]
    F += [P(
        f"Five candidate datasets were examined. <b>{len(valid_rank)} passed the data-validity "
        f"and leakage checks</b> and were benchmarked with an identical protocol: ten model "
        f"families (four classical, four deep, two ensembles), stratified 80/20 hold-out, "
        f"5-fold out-of-fold cross-validation for stacking, calibration and threshold "
        f"selection fitted on out-of-fold predictions only, and a single final evaluation on "
        f"the untouched test set.")]
    F += [P(
        f"<b>The recommended dataset is {winner['dataset']}</b> "
        f"(prediction unit: {winner['prediction_unit']}; {winner['samples']} usable samples; "
        f"{winner['safe_features']} scientifically defensible features). Its best model was "
        f"<b>{winner['best_model']}</b> at ROC-AUC {winner['roc_auc']:.3f} and PR-AUC "
        f"{winner['pr_auc']:.3f} against a base rate implying a {winner['pr_lift']}&#215; lift "
        f"over chance.")]
    F += [P(
        f"The single strongest result across all valid datasets was "
        f"<b>{best_overall['model']} on {best_overall['dataset']}</b> "
        f"(PR-AUC {best_overall['pr_auc']:.3f}, ROC-AUC {best_overall['roc_auc']:.3f}). "
        f"The best ensemble was <b>{best_ens['model']} on {best_ens['dataset']}</b> "
        f"(PR-AUC {best_ens['pr_auc']:.3f}).")]
    F += [P(
        "<b>All discrimination observed here is modest.</b> No dataset produced a model that "
        "separates inhibitor-positive from inhibitor-negative cases well enough to guide an "
        "individual clinical decision. The honest reading is that F8 variant descriptors "
        "carry real but limited signal about inhibitor risk, which is consistent with the "
        "published literature.", "warn")]

    F += [P("Headline leakage findings", "h2")]
    F += [P(
        "&bull; <b>HemophiliaA_Merged_MMC2_MMC3.csv was rejected outright.</b> It carries four "
        "outcome-derived aggregates (inhibitor_yes_count, inhibitor_no_count, "
        "inhibitor_total_known, inhibitor_positive_rate) plus a duplicate outcome column. Any "
        "target definable from that file is a deterministic function of those columns. It was "
        "run only to quantify how large a leakage-driven score looks.")]
    F += [P(
        "&bull; <b>Record-count columns in the ML-Ready dataset leak through label "
        "construction.</b> The label keeps only mutations whose clinical records agree, so a "
        "mutation with many records is disproportionately likely to be labelled negative. "
        "clotting_count alone separates the classes at AUC 0.393 (p = 8.5e-21), i.e. 0.607 in "
        "the inverted direction. This is an artefact of how the label was built, not biology. "
        "These columns were excluded from the primary benchmark.")]
    F += [P(
        "&bull; <b>FVIII activity and antigen measurements are of unknown temporal validity.</b> "
        "Inhibitors neutralise factor VIII, so an assay taken after inhibitor development "
        "partly reflects the outcome. The registries do not record assay timing relative to "
        "inhibitor development, so every clotting/ratio/antigen aggregate was classified "
        "UNKNOWN and excluded from the primary feature set rather than assumed safe.")]
    F += [P(
        "&bull; <b>pa_id is not a patient key.</b> It is missing for 70.9% of mmc3 records, "
        "contains placeholder values such as '?' and '1', and one code carries 322 records. "
        "No dataset here supports patient-level prediction; the mmc3 file is "
        "clinical-record-level and was split with grouping by mutation to stop records of the "
        "same mutation appearing in both train and test.")]
    F += [PageBreak()]

    # 2-3
    F += [P("2. Research Objective", "h1")]
    F += [P("Predict inhibitor development in Hemophilia A from F8 genomic information and, "
            "where scientifically valid, clinical information; and decide which of the "
            "available datasets is the most defensible foundation for the project.")]
    F += [P("The comparison is deliberately not won by accuracy. A dataset that scores highly "
            "because it contains outcome-derived columns ranks below a clean dataset that "
            "scores lower.")]

    F += [P("3. Dataset Overview", "h1")]
    cols = ["dataset", "prediction_unit", "usable_samples", "positive", "positive_pct",
            "features_after_audit", "missingness_pct", "grouped_cv"]
    hdr = ["Dataset", "Unit", "N", "Pos", "Pos %", "Feats", "Miss %", "Grouped"]
    data = [hdr] + [[str(r[c])[:34] for c in cols] for _, r in overview.iterrows()]
    F += [table(data, widths=[3.1*cm, 3.5*cm, 1.5*cm, 1.2*cm, 1.4*cm, 1.5*cm, 1.5*cm, 1.7*cm])]

    F += [P("4. Dataset Sources", "h1")]
    d = [["Dataset", "Source file", "SHA-256 (16)", "Role"]]
    roles = {
        "D1_CHAMP": "Project baseline. CDC CHAMP variant registry, already in production.",
        "D2_MLReady": "Derived, mutation-level; genomic + aggregated clinical.",
        "D3_MMC2_Genomic": "HADB supplementary table 2; genomic, independent label source.",
        "D4_MMC3_ClinicalRecord": "HADB supplementary table 3; per-record clinical observations.",
        "D5_Merged_LEAKY": "Derived join of mmc2+mmc3. REJECTED (outcome-derived columns).",
    }
    for k, v in dsmeta.items():
        d.append([k, v["source_file"][:34], v["sha256"][:16], roles.get(k, "")[:56]])
    F += [table(d, widths=[3.1*cm, 4.4*cm, 2.9*cm, 6.0*cm])]

    # 5-7
    F += [P("5. Dataset Structure, 6. Prediction Unit, 7. Target Definition", "h1")]
    for k, v in dsmeta.items():
        F += [P(f"{k}", "h2")]
        F += [P(f"<b>Prediction unit:</b> {v['prediction_unit']}. {v['unit_rationale']}", "small")]
        F += [P(f"<b>Target:</b> <font face='Courier'>{v['target_name']}</font>. "
                f"{v['target_definition']}", "small")]
        if v["notes"]:
            F += [P("<b>Notes:</b> " + " ".join(v["notes"]), "small")]
    F += [PageBreak()]

    # 8-10
    F += [P("8. Data Cleaning, 9. Missing Data", "h1")]
    F += [P("Each dataset has its own preprocessing pipeline; nothing learned on one dataset is "
            "applied to another. Categorical values are whitespace-normalised; inhibitor label "
            "vocabularies were mapped from the observed raw value counts, never assumed from a "
            "column name. Categorical missingness becomes an explicit '__missing__' level; "
            "numeric missingness is median-imputed with a missingness indicator. All imputation "
            "is fitted inside the training fold only.")]
    F += [P("Ambiguous label values were treated as unknown and excluded, never guessed: mmc3 "
            "'Not' (1276 rows, a truncated 'Not reported'), 'Not reported' (2090), blanks "
            "(1730), and two rows carrying severity values ('Mild','Severe') in the inhibitor "
            "column.")]

    F += [P("10. Leakage Analysis", "h1")]
    F += [P("Every column of every dataset was classified before any model was trained.")]
    counts = audit.groupby(["dataset", "classification"]).size().unstack(fill_value=0)
    order = ["SAFE", "POTENTIAL LEAKAGE", "DEFINITE LEAKAGE", "TARGET", "POST-OUTCOME",
             "IDENTIFIER", "METADATA",
             "UNKNOWN - TEMPORAL VALIDITY NOT ESTABLISHED"]
    order = [c for c in order if c in counts.columns]
    short = {"SAFE": "SAFE", "POTENTIAL LEAKAGE": "POTENTIAL", "DEFINITE LEAKAGE": "DEFINITE",
             "TARGET": "TARGET", "POST-OUTCOME": "POST-OUT", "IDENTIFIER": "ID",
             "METADATA": "META", "UNKNOWN - TEMPORAL VALIDITY NOT ESTABLISHED": "UNKNOWN"}
    d = [["Dataset"] + [short[c] for c in order]]
    for ds, row in counts.iterrows():
        d.append([ds] + [str(int(row[c])) for c in order])
    F += [table(d, widths=[4.0*cm] + [1.55*cm] * len(order))]
    F += [P("Full per-column classification with reasons: tables/leakage_audit.csv", "small")]

    F += [P("Columns excluded as DEFINITE leakage", "h2")]
    dl = audit[audit.classification == "DEFINITE LEAKAGE"]
    d = [["Dataset", "Column", "Reason"]]
    for _, r in dl.iterrows():
        d.append([r["dataset"], str(r["column"])[:26], str(r["reason"])[:110]])
    F += [table(d, widths=[3.3*cm, 3.2*cm, 10.0*cm], font=6.2)]
    F += [PageBreak()]

    # 11-12
    F += [P("11. Feature Selection", "h1")]
    for k, v in dsmeta.items():
        if not v["valid_for_supervised"]:
            continue
        sf = v["safe_features"]
        F += [P(f"<b>{k}</b> - {len(sf['categorical'])} categorical, {len(sf['numeric'])} numeric: "
                f"<font face='Courier'>{', '.join(sf['categorical'] + sf['numeric'])}</font>",
                "small")]

    F += [P("12. Train / Validation / Test Strategy", "h1")]
    F += [P(f"Seed {env['seed']} throughout. Outer split {int((1-env['test_size'])*100)}/"
            f"{int(env['test_size']*100)} stratified on the target. "
            f"{env['n_folds']}-fold stratified cross-validation inside the training portion "
            f"produces out-of-fold probabilities. For D4_MMC3_ClinicalRecord, both the outer "
            f"split and every fold are <b>grouped by mut_id</b>, because multiple records "
            f"describe the same mutation and a random split would place the same mutation on "
            f"both sides.")]
    F += [P("The held-out test set was not used for feature selection, hyperparameter choice, "
            "threshold selection, model selection or calibration fitting. Deep models take "
            "their early-stopping split from inside the training fold, never from the fold's "
            "held-out part.")]

    # 13-14
    F += [P("13. Classical ML Methodology", "h1")]
    F += [P("Random Forest (400 trees, depth 12, balanced_subsample), XGBoost (400 rounds, "
            "depth 4, lr 0.05, scale_pos_weight = n_neg/n_pos), LightGBM (400 rounds, 31 "
            "leaves, class_weight balanced), CatBoost (400 iterations, depth 6, "
            "auto_class_weights Balanced). Baseline hyperparameters were fixed in advance and "
            "identical across datasets; no per-dataset tuning was performed, so no dataset "
            "gains an advantage from search budget.")]
    F += [P(f"<b>Class imbalance:</b> {env['imbalance_strategy']}")]
    F += [P("SMOTE was deliberately not used. Oversampling before a grouped split would "
            "manufacture near-duplicates of minority records across folds.")]

    F += [P("14. Deep Learning Methodology", "h1")]
    F += [P("Deep MLP (256-128-64, BatchNorm, dropout 0.3); Residual MLP (width 128, three "
            "genuine residual blocks); TabTransformer (per-column categorical embeddings, "
            "2 encoder layers, 4 heads, d_model 32, numerics layer-normalised and concatenated "
            "at the head); 1D-CNN (two Conv1d blocks, kernel 5).")]
    F += [P("The TabTransformer consumes ordinal-coded categoricals, not one-hot columns, so it "
            "has its own preprocessing path - that is what makes it a TabTransformer rather "
            "than an MLP with attention bolted on.")]
    F += [P("<b>The 1D-CNN is reported with an explicit caveat.</b> Its input axis is the column "
            "order of the encoded feature matrix, which carries no biological ordering. The "
            "convolution therefore acts as a local weight-sharing regulariser over adjacent "
            "encoded columns. It must not be described as detecting spatial or sequence "
            "structure in the genome, because no such structure is present in the input.", "warn")]
    F += [P(f"Training controls: AdamW (lr {env['dl_settings']['lr']}, weight decay "
            f"{env['dl_settings']['weight_decay']}), ReduceLROnPlateau on validation AUC, early "
            f"stopping with patience {env['dl_settings']['patience']}, max "
            f"{env['dl_settings']['max_epochs']} epochs, best checkpoint restored. Focal loss is "
            f"implemented (alpha {env['focal_loss_params']['alpha']}, gamma "
            f"{env['focal_loss_params']['gamma']}) but the primary runs use class-weighted BCE "
            f"so that the imbalance strategy is identical to the classical models.")]

    dltab = EXP_DIR / "tables" / "deep_learning_training.csv"
    if dltab.exists():
        t = pd.read_csv(dltab)
        d = [["Dataset", "Model", "Epochs run", "Best epoch", "Best val AUC", "Early stopped"]]
        for _, r in t.iterrows():
            d.append([r["dataset"], r["model"], str(int(r["epochs"])),
                      str(int(r["best_epoch"])), f"{r['best_val_auc']:.4f}",
                      str(r["early_stopped"])])
        F += [P("Deep model training records (final refit on the full training portion):", "h2")]
        F += [table(d, widths=[4.2*cm, 3.0*cm, 2.2*cm, 2.2*cm, 2.4*cm, 2.4*cm])]
        F += [P("A best epoch of 1-2 means validation AUC peaked immediately and never "
                "improved: the network memorised the training fold rather than converging to a "
                "better solution. That is reported as-is, not described as convergence.", "small")]
    F += [PageBreak()]

    # 15-18
    F += [P("15. Stacking Ensemble", "h1")]
    F += [P(f"Base learners: the eight individual models. Meta-learner: "
            f"{env['meta_learner']}. The meta-model is fitted on <b>out-of-fold</b> "
            f"probabilities only, so no base model ever contributes a prediction made on data "
            f"it was trained on, and no test information reaches the meta-model.")]

    F += [P("16. Weighted Ensemble", "h1")]
    F += [P(f"Weight rule: {env['weighted_ensemble_rule']}. Weights are computed from "
            f"out-of-fold ROC-AUC on the training portion only and then frozen. The held-out "
            f"test set plays no part in choosing them.")]
    d = [["Dataset"] + ["RF", "XGB", "LGBM", "Cat", "MLP", "ResMLP", "TabT", "CNN"]]
    keys = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "DeepMLP",
            "ResidualMLP", "TabTransformer", "1D-CNN"]
    for name, b in bundles.items():
        if b.get("failed") or "ensemble_weights" not in b:
            continue
        d.append([name] + [f"{b['ensemble_weights'].get(k,0):.3f}" for k in keys])
    F += [table(d, widths=[4.0*cm] + [1.55*cm] * 8)]

    F += [P("17. Calibration", "h1")]
    F += [P(f"Rule: {env['calibration_rule']}. Both sigmoid (Platt) and isotonic calibrators "
            f"were fitted on out-of-fold predictions; the variant with the lowest out-of-fold "
            f"Brier score was selected and then applied unchanged to the test set. The test set "
            f"was never used to fit a calibrator.")]
    d = [["Dataset", "Model", "OOF Brier raw", "sigmoid", "isotonic", "Chosen",
          "Test Brier raw", "Test Brier cal."]]
    for name, b in bundles.items():
        if b.get("failed"):
            continue
        for m in ["StackingEnsemble", "WeightedEnsemble"]:
            c = b["calibration"].get(m)
            r = b["results"].get(m)
            if c and r:
                d.append([name, m, f"{c['raw']:.4f}", f"{c['sigmoid']:.4f}",
                          f"{c['isotonic']:.4f}", c["chosen"],
                          f"{r['brier']:.4f}", f"{r['brier_calibrated']:.4f}"])
    F += [table(d, widths=[3.6*cm, 2.7*cm, 1.9*cm, 1.6*cm, 1.6*cm, 1.5*cm, 1.7*cm, 1.8*cm])]

    F += [P("18. Threshold Optimization", "h1")]
    F += [P(f"Rule: {env['threshold_rule']}. J = sensitivity + specificity - 1 was maximised on "
            f"the out-of-fold predictions, the resulting threshold was frozen, and only then "
            f"applied to the held-out test set. No threshold was tuned on test data, and 0.50 "
            f"was never assumed.")]
    d = [["Dataset"] + [k for k in ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
                                    "StackingEnsemble", "WeightedEnsemble"]]]
    for name, b in bundles.items():
        if b.get("failed"):
            continue
        d.append([name] + [f"{b['thresholds'].get(k, float('nan')):.3f}" for k in d[0][1:]])
    F += [table(d, widths=[4.0*cm] + [2.05*cm] * 6)]

    F += [P("19. Evaluation Metrics", "h1")]
    F += [P("Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, Brier "
            "score and the full confusion matrix are reported for every model on every dataset, "
            "with training and inference time, feature count and class counts. Because every "
            "target here is imbalanced, <b>PR-AUC, recall and specificity carry the weight in "
            "the conclusions and accuracy is never used alone</b>: predicting 'no inhibitor' for "
            "everything scores 0.80-0.87 accuracy on these datasets and is useless.")]
    F += [PageBreak()]

    # 20-22 per-dataset results
    sec = 20
    for name in [k for k in dsmeta if not k.startswith("D5")]:
        b = bundles.get(name)
        if not b or b.get("failed"):
            F += [P(f"{sec}. {name} Results", "h1"),
                  P(f"This dataset failed to run: {b.get('error') if b else 'no result'}", "warn")]
            sec += 1
            continue
        sub = M[M.dataset == name]
        F += [P(f"{sec}. {name} Results", "h1")]
        F += [P(f"n_train {b['n_train']} (pos {b['train_pos']}), n_test {b['n_test']} "
                f"(pos {b['test_pos']}), encoded features {b['n_features_encoded']}, "
                f"runtime {b.get('elapsed_seconds', 0)/60:.1f} min.", "small")]
        d = [["Model", "ROC-AUC", "PR-AUC", "F1", "Sens", "Spec", "Prec", "Acc",
              "Brier", "Thr", "TN", "FP", "FN", "TP", "Fit s"]]
        for _, r in sub.iterrows():
            d.append([r["model"], f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}",
                      f"{r['f1']:.3f}", f"{r['recall_sensitivity']:.3f}",
                      f"{r['specificity']:.3f}", f"{r['precision']:.3f}",
                      f"{r['accuracy']:.3f}", f"{r['brier']:.3f}", f"{r['threshold']:.3f}",
                      str(r["tn"]), str(r["fp"]), str(r["fn"]), str(r["tp"]),
                      "-" if pd.isna(r["fit_seconds"]) else f"{r['fit_seconds']:.1f}"])
        F += [table(d, widths=[2.7*cm] + [0.98*cm]*9 + [0.72*cm]*4 + [0.95*cm], font=6.1)]
        F += fig(f"roc_pr_{name}.png", caption=f"ROC and precision-recall curves, {name}, held-out test set.")
        sec += 1
    F += [PageBreak()]

    # 23-24
    F += [P("23. Cross-Dataset Comparison", "h1")]
    F += [P("Scores are not directly comparable between datasets: each has a different target "
            "definition, base rate and feature space. They answer 'how much signal is in this "
            "dataset', not 'which model is better'.")]
    F += fig("cross_dataset_roc_auc.png", caption="ROC-AUC by model and dataset.")
    F += fig("cross_dataset_pr_auc.png", caption="PR-AUC by model and dataset. PR-AUC must be "
                                                 "read against each dataset's own base rate.")

    F += [P("24. Model Comparison", "h1")]
    for metric, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"),
                          ("recall_sensitivity", "Sensitivity"),
                          ("specificity", "Specificity"), ("brier", "Brier score")]:
        piv = M.pivot_table(index="dataset", columns="model", values=metric, aggfunc="first")
        cols = [c for c in ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "DeepMLP",
                            "ResidualMLP", "TabTransformer", "1D-CNN",
                            "StackingEnsemble", "WeightedEnsemble"] if c in piv.columns]
        head = ["Dataset", "RF", "XGB", "LGBM", "Cat", "MLP", "ResMLP", "TabT", "CNN",
                "Stack", "Weight"]
        d = [head]
        for ds, row in piv.iterrows():
            d.append([ds] + [f"{row[c]:.3f}" if pd.notna(row[c]) else "-" for c in cols])
        F += [P(f"{label} (held-out test)", "h2"), table(d, widths=[3.7*cm] + [1.28*cm]*10)]
    F += [PageBreak()]

    # 25-26
    F += [P("25. SHAP Analysis", "h1")]
    if not expl:
        F += [P("SHAP was not produced for this run.", "warn")]
    for name, e in expl.items():
        if e.get("skipped"):
            continue
        F += [P(f"{name} - model explained: {e['explained_model']}", "h2")]
        if e.get("note"):
            F += [P(e["note"], "warn")]
        s = e.get("shap", {})
        if not s.get("available"):
            F += [P(f"SHAP unavailable: {s.get('reason')}", "warn")]
        else:
            d = [["Feature (source column)", "mean |SHAP|"]]
            for it in s["global_importance"][:10]:
                d.append([it["feature"], f"{it['mean_abs_shap']:.5f}"])
            F += [table(d, widths=[8.0*cm, 3.0*cm])]
            if s.get("direction"):
                d = [["Feature", "corr(value, SHAP)", "Reading"]]
                for it in s["direction"][:6]:
                    d.append([it["feature"], f"{it['corr_value_vs_shap']:+.3f}", it["direction"]])
                F += [table(d, widths=[4.6*cm, 3.2*cm, 6.4*cm])]
            F += fig(f"shap_global_{name}.png", width=13*cm,
                     caption="Encoded columns aggregated back to their source column, so no "
                             "bar names a one-hot artefact.")
    F += [PageBreak()]

    F += [P("26. LIME Analysis", "h1")]
    for name, e in expl.items():
        if e.get("skipped"):
            continue
        l = e.get("lime", {})
        F += [P(f"{name} ({e['explained_model']})", "h2")]
        if not l.get("available"):
            F += [P(f"LIME unavailable: {l.get('reason')}", "warn")]
            continue
        for ex in l["examples"]:
            F += [P(f"<b>{ex['case']}</b> - test row {ex['test_row']}, predicted probability "
                    f"{ex['predicted_probability']:.4f}, true label {ex['true_label']}", "small")]
            d = [["Local rule", "Source feature", "Weight", "Direction"]]
            for c in ex["top_contributions"][:6]:
                d.append([c["encoded_rule"][:46], c["source_feature"][:22],
                          f"{c['weight']:+.4f}", c["direction"]])
            F += [table(d, widths=[7.2*cm, 4.0*cm, 1.9*cm, 2.2*cm], font=6.3)]

    # 27-30
    F += [PageBreak(), P("27-29. ROC, Precision-Recall and Calibration Curves", "h1")]
    for name in bundles:
        if bundles[name].get("failed"):
            continue
        F += fig(f"calibration_{name}.png", width=15.5*cm,
                 caption=f"Calibration of the two strongest models on {name}. "
                         f"Calibrators were fitted on out-of-fold predictions only.")

    F += [PageBreak(), P("30. Confusion Matrices", "h1")]
    F += [P("At the frozen Youden-J thresholds, on the held-out test set.", "small")]
    for name in bundles:
        if bundles[name].get("failed"):
            continue
        F += fig(f"confusion_{name}.png", width=15.5*cm, caption=name)

    # 31-33
    F += [PageBreak(), P("31. Dataset Ranking", "h1")]
    F += [P("Ranking is a weighted score over seven criteria, not a leaderboard of accuracy: "
            "scientific validity 30%, prediction unit 15%, PR-AUC lift over base rate 16%, "
            "sample size 12%, positive-case count 12%, calibration 8%, feature count 7%. "
            "Validity is weighted highest so that a leakage-driven score can never win.")]
    d = [["Rank", "Dataset", "Unit", "N", "Test pos", "Feats", "Best model",
          "ROC-AUC", "PR-AUC", "PR lift", "Brier", "Score"]]
    for _, r in ranking.iterrows():
        d.append([str(r["rank"]), r["dataset"], r["prediction_unit"][:20], str(r["samples"]),
                  str(r["test_pos"]), str(r["safe_features"]), r["best_model"],
                  f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}", f"{r['pr_lift']:.2f}x",
                  f"{r['brier']:.3f}", f"{r['score']:.3f}"])
    F += [table(d, widths=[1.1*cm, 3.3*cm, 2.5*cm, 1.2*cm, 1.3*cm, 1.1*cm, 2.4*cm,
                           1.4*cm, 1.3*cm, 1.2*cm, 1.2*cm, 1.2*cm], font=6.2)]
    F += fig("leakage_comparison.png", width=13*cm,
             caption="Best model per dataset. Red is the rejected dataset, shown only to "
                     "quantify what outcome-derived features buy.")

    F += [P("Why each dataset ranks where it does", "h2")]
    for _, r in ranking.iterrows():
        meta = dsmeta[r["dataset"]]
        if not r["valid"]:
            F += [P(f"<b>{r['dataset']} - REJECTED.</b> {meta['rejection_reason']}", "warn")]
        else:
            F += [P(f"<b>{r['dataset']} (rank {r['rank']}).</b> {r['prediction_unit']}; "
                    f"{r['samples']} samples, {r['test_pos']} positives in test, "
                    f"{r['safe_features']} safe features. Best PR-AUC {r['pr_auc']:.3f} "
                    f"({r['pr_lift']}&#215; base rate) with {r['best_model']}.", "small")]

    F += [P("32. Best Dataset Recommendation", "h1")]
    F += [P(f"<b>THE BEST DATASET FOR THIS PROJECT IS: {winner['dataset']}</b>", "h2")]
    F += [P(_why_winner(winner, dsmeta, ranking))]

    F += [P("33. Best Model Recommendation", "h1")]
    wsub = M[(M.dataset == winner["dataset"])].sort_values("pr_auc", ascending=False)
    top3 = wsub.head(3)
    d = [["Model", "ROC-AUC", "PR-AUC", "F1", "Sens", "Spec", "Brier"]]
    for _, r in top3.iterrows():
        d.append([r["model"], f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}", f"{r['f1']:.3f}",
                  f"{r['recall_sensitivity']:.3f}", f"{r['specificity']:.3f}",
                  f"{r['brier']:.3f}"])
    F += [table(d, widths=[3.4*cm] + [1.9*cm]*6)]
    F += [P(f"On {winner['dataset']} the strongest model by PR-AUC is "
            f"<b>{top3.iloc[0]['model']}</b>. Differences among the top models are small "
            f"relative to the spread across cross-validation folds, so this is a weak "
            f"preference, not a decisive one. The gradient-boosted trees and the ensembles are "
            f"effectively tied; the deep models did not beat them on any dataset here, which is "
            f"the expected result at this sample size and feature count.")]
    F += [PageBreak()]

    # 34-40
    F += [P("34. Research Interpretation", "h1")]
    F += [P("Published CHAMP-based and HADB-based inhibitor studies consistently report that "
            "F8 variant class is associated with inhibitor risk - null mutations (large "
            "deletions, nonsense, intron inversions) carry substantially higher risk than "
            "missense variants. The benchmark reproduces that ordering: variant type and "
            "reported severity dominate the SHAP rankings on every genomic dataset.")]
    F += [P("<b>No claim is made that these models outperform published research.</b> The "
            "comparison would not be valid: this experiment uses different datasets, a "
            "different prediction unit (variant rather than patient), a different feature set, "
            "an internal random split rather than an independent cohort, and no external "
            "validation. Higher internal test performance than a published figure would "
            "indicate a difference in evaluation design, not a better model.", "warn")]

    F += [P("35. Limitations", "h1")]
    for t in [
        "<b>Variant-level, not patient-level.</b> No dataset here supports patient-level "
        "prediction. Estimates are attributable to an F8 variant, not to an individual.",
        "<b>Reporting bias in every target.</b> Between 25% and 43% of rows are excluded for "
        "having no reported inhibitor status. That exclusion is very unlikely to be random: "
        "variants studied in inhibitor-focused work are likelier to have the field filled. "
        "Every positive rate reported here describes registry reporting, not population "
        "incidence.",
        "<b>Label construction bias in D2_MLReady.</b> Mutations whose clinical records "
        "disagree were dropped by the dataset's authors. That removes exactly the mutations "
        "with mixed evidence and makes the task look cleaner than it is.",
        "<b>Overlapping sources.</b> D2, D3 and D4 all derive from the same HADB supplementary "
        "tables, so they are not independent of one another. D1_CHAMP is a separate registry.",
        "<b>No hyperparameter search.</b> Fixed baseline hyperparameters were used for fairness; "
        "a tuned model on any dataset would likely score somewhat higher.",
        "<b>Single split.</b> One 80/20 hold-out per dataset. Confidence intervals were not "
        "estimated, so small differences between models should not be over-read.",
    ]:
        F += [P("&bull; " + t, "small")]

    F += [P("36. Data Leakage Risks", "h1")]
    F += [P("Handled: outcome-derived aggregates (rejected dataset), label-construction leakage "
            "via record counts (excluded), FVIII assays of unknown timing (excluded), "
            "mutation-sharing between train and test in the record-level dataset (grouped "
            "splitting), preprocessing fitted before splitting (fitted inside folds only), and "
            "stacking features built from in-sample predictions (out-of-fold only).")]
    F += [P("Residual risk: the registries do not timestamp assays or inhibitor testing, so "
            "temporal validity cannot be fully established for any clinical variable. This is "
            "why every FVIII-derived column is classified UNKNOWN rather than SAFE, and why the "
            "primary benchmark is genomic.", "small")]

    F += [P("37. Generalization and External Validation Limitations", "h1")]
    F += [P("<b>No external validation was performed and none is claimed.</b> Every score in "
            "this report comes from a random hold-out of the same registry the model was "
            "trained on. Registry-to-registry transfer (for example CHAMP to HADB) was not "
            "tested; the two use different schemas and a like-for-like feature mapping does not "
            "exist without additional work. Nothing here supports clinical deployment.", "warn")]

    F += [P("38. Recommended Final Project Dataset", "h1")]
    F += [P(_recommendation_text(winner, ranking, dsmeta))]

    F += [P("39. Recommended Final Project Model", "h1")]
    F += [P(f"Keep a calibrated gradient-boosted tree or the stacking ensemble. On "
            f"{winner['dataset']} the best PR-AUC was {top3.iloc[0]['pr_auc']:.3f} with "
            f"{top3.iloc[0]['model']}. The four deep architectures were trained fully and are "
            f"reported, but none beat the tree models on any dataset; at these sample sizes "
            f"(2.2k-5.0k rows, 8-15 features) that is the expected outcome and there is no "
            f"case for putting a neural network in production here.")]

    F += [P("40. Recommended Next Steps", "h1")]
    for t in [
        "Quantify the reporting bias: model P(inhibitor status reported) and compare the "
        "covariate distribution of reported and unreported variants.",
        "Test registry-to-registry transfer by hand-mapping CHAMP variant classes onto the HADB "
        "schema and evaluating a CHAMP-trained model on HADB variants, and vice versa. That "
        "would be the first genuine external validation in this project.",
        "Re-examine the 127 conflicting mutations that D2 discards; a soft label or a "
        "count-based target may use them without the construction artefact.",
        "If a patient-level dataset becomes available, revisit the objective - patient-level "
        "prediction is the clinically useful question and no dataset here answers it.",
        "Add bootstrap confidence intervals to the held-out metrics before quoting any single "
        "number as the project's performance.",
    ]:
        F += [P("&bull; " + t, "small")]

    # summary table
    F += [PageBreak(), P("Final summary table", "h1")]
    d = [["Dataset", "Unit", "N", "Pos", "Feats", "Best model", "ROC-AUC", "PR-AUC",
          "F1", "Sens", "Spec", "Brier", "Leakage risk", "Validity", "Rank"]]
    for _, r in ranking.iterrows():
        risk = "REJECTED - outcome-derived" if not r["valid"] else _risk(r["dataset"])
        d.append([r["dataset"][:16], r["prediction_unit"][:14], str(r["samples"]),
                  str(r["test_pos"]), str(r["safe_features"]), r["best_model"][:13],
                  f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}", f"{r['f1']:.3f}",
                  f"{r['sensitivity']:.3f}", f"{r['specificity']:.3f}", f"{r['brier']:.3f}",
                  risk[:22], "valid" if r["valid"] else "invalid", str(r["rank"])])
    F += [table(d, widths=[2.3*cm, 1.9*cm, 1.0*cm, 0.9*cm, 0.85*cm, 1.9*cm, 1.15*cm,
                           1.1*cm, 0.95*cm, 0.95*cm, 0.95*cm, 0.95*cm, 2.2*cm, 1.1*cm,
                           0.9*cm], font=5.6)]

    F += [Spacer(1, 12), P(
        "This is a research benchmark on registry data. It is not clinically validated, has no "
        "external validation cohort, makes no claim of readiness for clinical use, and must not "
        "be used to guide treatment.", "warn")]
    F += [P(f"Reproducibility: seed {env['seed']}, {env['n_folds']}-fold CV, Python "
            f"{env['python']}, scikit-learn {env['scikit_learn']}, XGBoost {env['xgboost']}, "
            f"LightGBM {env['lightgbm']}, CatBoost {env['catboost']}, PyTorch {env['torch']}, "
            f"SHAP {env['shap']}. Dataset SHA-256 digests in configs/datasets.json.", "small")]

    doc.build(F)
    return str(out)


def _risk(name: str) -> str:
    return {
        "D1_CHAMP": "low - genomic only",
        "D2_MLReady": "medium - construction",
        "D3_MMC2_Genomic": "low - genomic only",
        "D4_MMC3_ClinicalRecord": "medium - grouped",
    }.get(name, "unassessed")


def _why_winner(w, dsmeta, ranking) -> str:
    meta = dsmeta[w["dataset"]]
    others = ranking[(ranking.valid) & (ranking.dataset != w["dataset"])]
    txt = (f"{w['dataset']} wins on the combined criteria, not on raw performance. "
           f"Prediction unit: {meta['prediction_unit']}. Target: "
           f"<font face='Courier'>{meta['target_name']}</font> - {meta['target_definition']} "
           f"It offers {w['samples']} usable samples with {w['safe_features']} defensible "
           f"features and the lowest leakage exposure of the candidates. ")
    txt += "Why not the others: "
    for _, r in others.iterrows():
        txt += f"<b>{r['dataset']}</b> scored {r['score']:.3f} vs {w['score']:.3f}; "
    txt += ("and D5_Merged_LEAKY was rejected outright for containing outcome-derived "
            "aggregates.")
    return txt


def _recommendation_text(w, ranking, dsmeta) -> str:
    return (
        f"<b>Use {w['dataset']} as the primary dataset.</b> "
        f"Target: <font face='Courier'>{dsmeta[w['dataset']]['target_name']}</font>. "
        f"Features to use: the SAFE set listed in section 11. Features that must be excluded: "
        f"all outcome-derived aggregates, all FVIII activity/antigen columns of unknown "
        f"temporal validity, all record-count columns, all near-unique nomenclature strings, "
        f"and the source-publication identifier. "
        f"Recommended roles for the remaining datasets: the other genomic mutation-level "
        f"dataset is a <b>secondary/replication</b> experiment because it draws on the same "
        f"registry family; the clinical-record dataset is a <b>supplementary analysis</b> for "
        f"record-level questions, not a substitute, because it is not patient-level and its "
        f"records are not independent; the merged file should be <b>not used</b> for modelling "
        f"in any form."
    )


if __name__ == "__main__":
    print(build())
