"""Sensitivity analyses that justify the audit decisions with numbers.

Each run answers "what would including this excluded block have bought?".
LightGBM only, identical settings, same split - the point is the delta, not the
absolute score.
"""
from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark import outer_split
from config import EXP_DIR, SEED
from data import (build_champ, build_mlready, build_mmc2_genomic,
                  build_mmc3_records, EXTERNAL_DATA, _norm)
from models import classical_zoo, make_onehot_preprocessor

warnings.filterwarnings("ignore")


def _score(spec, X, y, groups) -> dict:
    itr, ite = outer_split(X, y, groups)
    pre = make_onehot_preprocessor(spec["numeric"], spec["categorical"])
    A = np.asarray(pre.fit_transform(X.iloc[itr]), dtype=np.float64)
    B = np.asarray(pre.transform(X.iloc[ite]), dtype=np.float64)
    m = classical_zoo()["LightGBM"]
    m.fit(A, y[itr])
    p = m.predict_proba(B)
    return {"roc_auc": float(roc_auc_score(y[ite], p)),
            "pr_auc": float(average_precision_score(y[ite], p)),
            "n_features_encoded": int(A.shape[1])}


def run() -> list[dict]:
    rows: list[dict] = []

    # ---- 1. D2_MLReady: what the excluded record-count columns would add ----
    s = build_mlready()
    raw = pd.read_csv(EXTERNAL_DATA / "HemophiliaA_ML_Ready_Inhibitor.csv", low_memory=False)
    y = s.y.to_numpy()
    base = {"numeric": s.numeric, "categorical": s.categorical}
    rows.append({"analysis": "D2_MLReady safe features only (primary)",
                 "dataset": "D2_MLReady", **_score(base, s.X, y, None)})

    counts = ["clinical_record_count", "clotting_count", "ratio_count",
              "antigen_count", "act_ant_count", "Count_mut_id"]
    X2 = s.X.copy()
    for c in counts:
        X2[c] = raw[c].to_numpy()
    rows.append({"analysis": "D2_MLReady + record-count columns (EXCLUDED as construction leakage)",
                 "dataset": "D2_MLReady",
                 **_score({"numeric": s.numeric + counts, "categorical": s.categorical},
                          X2, y, None)})

    # ---- 2. D2_MLReady: what the FVIII assay aggregates would add ----------
    fviii = [c for c in raw.columns
             if c.startswith(("clotting_", "ratio_", "antigen_", "act_ant_"))
             and not c.endswith("_count")]
    X3 = s.X.copy()
    for c in fviii:
        X3[c] = pd.to_numeric(raw[c], errors="coerce").to_numpy()
    rows.append({"analysis": "D2_MLReady + FVIII assay aggregates (EXCLUDED: unknown timing)",
                 "dataset": "D2_MLReady",
                 **_score({"numeric": s.numeric + fviii, "categorical": s.categorical},
                          X3, y, None)})

    # ---- 3. D4: grouped vs ungrouped splitting ----------------------------
    s4 = build_mmc3_records()
    y4 = s4.y.to_numpy()
    g4 = s4.groups.to_numpy()
    base4 = {"numeric": s4.numeric, "categorical": s4.categorical}
    rows.append({"analysis": "D4 grouped split by mut_id (primary, correct)",
                 "dataset": "D4_MMC3_ClinicalRecord", **_score(base4, s4.X, y4, g4)})
    rows.append({"analysis": "D4 UNGROUPED random split (INCORRECT - same mutation on both sides)",
                 "dataset": "D4_MMC3_ClinicalRecord", **_score(base4, s4.X, y4, None)})

    # ---- 4. D4: dropping the partly assay-derived clinical phenotype -------
    cats_no_cli = [c for c in s4.categorical if c != "cli_phe"]
    rows.append({"analysis": "D4 grouped, without cli_phe (clinical phenotype)",
                 "dataset": "D4_MMC3_ClinicalRecord",
                 **_score({"numeric": s4.numeric, "categorical": cats_no_cli},
                          s4.X[cats_no_cli], y4, g4)})

    # ---- 5. D3 genomic: adding the POST-OUTCOME severity summary ----------
    s3 = build_mmc2_genomic()
    mmc2 = pd.read_csv(EXTERNAL_DATA / "BVTH_VTH-2024-000215-mmc2.csv", low_memory=False)
    y_all, _ = __import__("data")._map_inhibitor(mmc2["uinhibitor"])
    keep = y_all.notna()
    X5 = s3.X.copy()
    X5["useverity"] = _norm(mmc2.loc[keep, "useverity"]).reset_index(drop=True)
    rows.append({"analysis": "D3 safe genomic only (primary)", "dataset": "D3_MMC2_Genomic",
                 **_score({"numeric": s3.numeric, "categorical": s3.categorical},
                          s3.X, s3.y.to_numpy(), None)})
    rows.append({"analysis": "D3 + useverity (EXCLUDED as POST-OUTCOME)",
                 "dataset": "D3_MMC2_Genomic",
                 **_score({"numeric": s3.numeric, "categorical": s3.categorical + ["useverity"]},
                          X5, s3.y.to_numpy(), None)})

    df = pd.DataFrame(rows)
    df.to_csv(EXP_DIR / "tables" / "sensitivity_analysis.csv", index=False)
    print(df.to_string(index=False))
    return rows




def champ_identifier_probe() -> pd.DataFrame:
    """Test whether encoding near-unique variant identifiers reproduces the
    ~0.9999 AUC reported in the project's own research paper.

    The paper one-hot encodes nominal categoricals including HGVS cDNA and hg19
    coordinates, reaching 1800-2500 dimensions on 4036 CHAMP rows. Those columns
    have 4038 and 3072 distinct values over 4050 rows, i.e. they are row
    identifiers. This runs that configuration and reports what actually happens.
    """
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[3]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from ml.preprocessing import champ as cm

    raw = cm.load_champ(cm.DEFAULT_CHAMP_PATH)
    norm = cm.normalise_champ(raw)
    lab = norm[cm.LABEL_COLUMN].fillna("(missing)").astype(str).str.strip().str.lower()

    rows = []
    for label_scheme in ["yes_vs_no (this benchmark)", "yes_vs_rest (all 4050 rows)"]:
        if label_scheme.startswith("yes_vs_no"):
            keep = lab.isin(["yes", "no"])
            y = lab[keep].eq("yes").astype(int).to_numpy()
            sub = norm[keep]
        else:
            keep = pd.Series(True, index=norm.index)
            y = lab.eq("yes").astype(int).to_numpy()
            sub = norm

        for cfg, cats, nums in [
            ("genomic classes only (this benchmark)",
             list(cm.CATEGORICAL_FEATURES), list(cm.NUMERIC_FEATURES) + list(cm.BINARY_FEATURES)),
            ("+ HGVS cDNA / hg19 / HGVS Protein identifiers (paper-style)",
             list(cm.CATEGORICAL_FEATURES) + ["HGVS cDNA", "hg19 Coordinates", "HGVS Protein"],
             list(cm.NUMERIC_FEATURES) + list(cm.BINARY_FEATURES)),
        ]:
            X = sub[[c for c in cats + nums if c in sub.columns]].copy()
            for c in cats:
                if c in X.columns:
                    X[c] = _norm(X[c])
            try:
                r = _score({"numeric": [c for c in nums if c in X.columns],
                            "categorical": [c for c in cats if c in X.columns]},
                           X.reset_index(drop=True), y, None)
                rows.append({"label_scheme": label_scheme, "feature_set": cfg,
                             "n_rows": int(len(y)), "n_positive": int(y.sum()), **r})
            except Exception as exc:
                rows.append({"label_scheme": label_scheme, "feature_set": cfg,
                             "n_rows": int(len(y)), "n_positive": int(y.sum()),
                             "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame(rows)
    df.to_csv(EXP_DIR / "tables" / "champ_identifier_probe.csv", index=False)
    print("\n=== CHAMP identifier probe ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
    champ_identifier_probe()
