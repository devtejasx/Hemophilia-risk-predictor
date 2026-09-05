# Dataset benchmark — Hemophilia A inhibitor risk

**Analysis-only experiment.** Nothing here modifies the production project. No
production code, model artifact, database or dataset was changed, and nothing
was committed to git.

## What this is

A fair comparison of every available dataset for the question *"which dataset is
the most defensible foundation for Hemophilia A inhibitor-risk prediction?"*,
using an identical protocol across ten model families.

## Layout

```
configs/       environment.json (versions, seeds), datasets.json (sources, SHA-256, targets, audits)
logs/          benchmark.log, run_stdout.log, extracted text of the reference PDFs
metrics/       all_metrics.csv (tidy, one row per dataset x model), raw_results.json
tables/        dataset_overview, leakage_audit, model_comparison_<metric>,
               dataset_ranking, sensitivity_analysis, champ_identifier_probe,
               deep_learning_training
plots/         roc_pr_*, confusion_*, calibration_*, shap_*, cross_dataset_*, leakage_comparison
predictions/   test_<dataset>.csv and oof_<dataset>.csv (probabilities per model)
explanations/  explanations.json (SHAP global + direction, LIME local cases)
report/        Hemophilia_A_Dataset_Benchmark_Report.pdf
src/           the experiment code
```

## Reproducing

Requires a Python 3.12 environment with scikit-learn 1.8.0, xgboost, lightgbm,
catboost, torch (CPU), shap, lime, imbalanced-learn, matplotlib, reportlab.

```bash
cd src
python run_benchmark.py     # ~30 min on 16 CPU cores
python plots.py
python explain.py
python sensitivity.py
python report.py
```

Seed 42 throughout. Dataset SHA-256 digests are recorded in
`configs/datasets.json` so the inputs can be verified.

## Protocol summary

* Outer split 80/20 stratified; **grouped by `mut_id`** for the record-level
  dataset so the same mutation cannot appear on both sides.
* 5-fold stratified (grouped) CV inside the training portion produces
  out-of-fold probabilities.
* Stacking meta-learner, weighted-ensemble weights, calibration and the decision
  threshold are all fitted on out-of-fold predictions **only**.
* The held-out 20% is scored exactly once, at the end.
* Class imbalance: `class_weight`/`scale_pos_weight = n_neg/n_pos` for classical
  models, `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` for deep models. Identical
  across datasets. SMOTE deliberately not used.

## Datasets

| Key | Source | Unit | Status |
|---|---|---|---|
| D1_CHAMP | `ml/data/champ.csv` | mutation | valid — project baseline |
| D2_MLReady | `HemophiliaA_ML_Ready_Inhibitor.csv` | mutation | valid |
| D3_MMC2_Genomic | `BVTH_VTH-2024-000215-mmc2.csv` | mutation | valid |
| D4_MMC3_ClinicalRecord | `BVTH_VTH-2024-000215-mmc3.csv` | clinical record | valid (grouped CV) |
| D5_Merged_LEAKY | `HemophiliaA_Merged_MMC2_MMC3.csv` | mutation | **rejected** — outcome-derived columns |

## Caveats that travel with every number here

* All results are **variant-level**, not patient-level. No available dataset has
  a usable patient key.
* Every target excludes rows with unreported inhibitor status (25–43%), and that
  exclusion is not random.
* No external validation was performed and none is claimed. Nothing here
  supports clinical use.
