"""SHAP and LIME for the strongest model on each valid dataset.

Two rules, same as the production service:
  * encoded feature names are mapped back to the ORIGINAL source columns, so no
    explanation names a post-encoding artefact like `cat__mut_type_Missense`;
  * if an explainer fails it is recorded as unavailable with the reason, never
    replaced by something that merely looks like an attribution.
"""
from __future__ import annotations

import json
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import outer_split
from config import EXP_DIR, SEED
from data import build_all
from models import classical_zoo, make_onehot_preprocessor

warnings.filterwarnings("ignore")
PLOTS = EXP_DIR / "plots"
OUT = EXP_DIR / "explanations"

TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}


def source_column(encoded: str, sources: list[str]) -> str:
    """`cat__mut_type_Missense` -> `mut_type`. Longest match wins so that
    `aa_numb` does not swallow `aa_numb_old`."""
    name = re.sub(r"^(cat|num|bin|remainder)__", "", encoded)
    best = ""
    for s in sources:
        if (name == s or name.startswith(s + "_")) and len(s) > len(best):
            best = s
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):]
    return best or name


def explain_dataset(spec, best_model_name: str) -> dict:
    """Refit the winning model on the training split and explain it."""
    X, y = spec.X.reset_index(drop=True), spec.y.reset_index(drop=True).to_numpy()
    groups = spec.groups.reset_index(drop=True).to_numpy() if spec.groups is not None else None
    itr, ite = outer_split(X, y, groups)

    pre = make_onehot_preprocessor(spec.numeric, spec.categorical)
    A = np.asarray(pre.fit_transform(X.iloc[itr]), dtype=np.float64)
    B = np.asarray(pre.transform(X.iloc[ite]), dtype=np.float64)
    names = [str(n) for n in pre.get_feature_names_out()]
    sources = list(spec.categorical) + list(spec.numeric)

    if best_model_name not in TREE_MODELS:
        # SHAP/LIME here are run against a tree surrogate only when the winner is
        # a tree; otherwise fall back to the strongest tree model and say so.
        fallback = "LightGBM"
    else:
        fallback = best_model_name
    model = classical_zoo()[fallback]
    model.fit(A, y[itr])

    result = {"dataset": spec.name, "explained_model": fallback,
              "requested_model": best_model_name,
              "note": (None if fallback == best_model_name else
                       f"The best model was {best_model_name}, which is not a tree "
                       f"model; TreeSHAP was run on {fallback} instead and this is "
                       f"reported as a surrogate, not as an explanation of "
                       f"{best_model_name}.")}

    # ---------------- SHAP ----------------
    try:
        import shap
        expl = shap.TreeExplainer(model.model)
        sv = expl(B)
        vals = np.asarray(sv.values)
        if vals.ndim == 3:
            vals = vals[:, :, -1]

        # aggregate encoded columns back to source columns
        agg: dict[str, float] = {}
        for j, enc in enumerate(names):
            agg[source_column(enc, sources)] = agg.get(source_column(enc, sources), 0.0) \
                + float(np.abs(vals[:, j]).mean())
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        result["shap"] = {"available": True,
                          "global_importance": [{"feature": k, "mean_abs_shap": round(v, 6)}
                                                for k, v in ranked]}

        fig, ax = plt.subplots(figsize=(7, 4.2))
        top = ranked[:12][::-1]
        ax.barh([k for k, _ in top], [v for _, v in top], color="#0F6E72")
        ax.set(xlabel="mean |SHAP value|",
               title=f"SHAP global importance - {spec.name} ({fallback})\n"
                     f"aggregated to source columns")
        fig.savefig(PLOTS / f"shap_global_{spec.name}.png", bbox_inches="tight")
        plt.close(fig)

        # beeswarm on encoded columns (kept separate: it is per encoded column)
        fig = plt.figure(figsize=(7.5, 5))
        shap.summary_plot(vals, B, feature_names=names, max_display=15, show=False)
        plt.title(f"SHAP summary - {spec.name} ({fallback})", fontsize=10)
        plt.savefig(PLOTS / f"shap_summary_{spec.name}.png", bbox_inches="tight")
        plt.close(fig)

        # direction: sign of correlation between feature value and SHAP value
        direction = []
        for k, _ in ranked[:10]:
            idx = [j for j, e in enumerate(names) if source_column(e, sources) == k]
            if not idx:
                continue
            fv = B[:, idx].sum(axis=1)
            sv_k = vals[:, idx].sum(axis=1)
            if np.std(fv) > 0 and np.std(sv_k) > 0:
                r = float(np.corrcoef(fv, sv_k)[0, 1])
                direction.append({"feature": k, "corr_value_vs_shap": round(r, 4),
                                  "direction": "higher value raises risk" if r > 0
                                  else "higher value lowers risk"})
        result["shap"]["direction"] = direction
    except Exception as exc:
        result["shap"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    # ---------------- LIME ----------------
    try:
        from lime.lime_tabular import LimeTabularExplainer
        lime_expl = LimeTabularExplainer(
            A, feature_names=names, class_names=["no inhibitor", "inhibitor"],
            mode="classification", random_state=SEED, discretize_continuous=True)
        probs = model.predict_proba(B)
        # a high-risk, a low-risk and a borderline case
        picks = {"highest_risk": int(np.argmax(probs)),
                 "lowest_risk": int(np.argmin(probs)),
                 "borderline": int(np.argmin(np.abs(probs - 0.5)))}
        examples = []
        for label, i in picks.items():
            e = lime_expl.explain_instance(B[i], model.model.predict_proba,
                                           num_features=8, labels=(1,))
            contribs = []
            for enc_desc, w in e.as_list(label=1):
                base = enc_desc.split(" ")[0].strip("()")
                contribs.append({"encoded_rule": enc_desc,
                                 "source_feature": source_column(base, sources),
                                 "weight": round(float(w), 6),
                                 "direction": "increases" if w > 0 else
                                              ("decreases" if w < 0 else "no effect")})
            examples.append({"case": label, "test_row": i,
                             "predicted_probability": round(float(probs[i]), 6),
                             "true_label": int(y[ite][i]),
                             "top_contributions": contribs})
        result["lime"] = {"available": True, "examples": examples}
    except Exception as exc:
        result["lime"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    return result


def main() -> int:
    metrics = pd.read_csv(EXP_DIR / "metrics" / "all_metrics.csv")
    specs = {s.name: s for s in build_all()}
    out = {}
    for name, spec in specs.items():
        if not spec.valid_for_supervised:
            out[name] = {"dataset": name, "skipped": True,
                         "reason": "rejected for the primary benchmark (leakage)"}
            continue
        sub = metrics[(metrics.dataset == name)]
        if sub.empty:
            continue
        best = sub.sort_values("roc_auc", ascending=False).model.iloc[0]
        print(f"explaining {name} (best model: {best})", flush=True)
        out[name] = explain_dataset(spec, best)
    (OUT / "explanations.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT/'explanations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
