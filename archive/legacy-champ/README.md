# legacy-champ — the retired CHAMP pipeline

The application now predicts inhibitor risk from the **MMC2 + MMC3 Hemophilia A**
tables joined on `mut_id`. Everything in this directory belonged to the previous
CHAMP-based pipeline and is kept only so the change is auditable. **Nothing here
is imported, loaded or served.**

Moved with `git mv`, so `git log --follow` still works on each file.

| Here | Was | Replaced by |
|---|---|---|
| `champ.csv` | `ml/data/champ.csv` | `ml/data/BVTH_VTH-2024-000215-mmc2.csv` + `…-mmc3.csv` |
| `champ_preprocessing.py` | `ml/preprocessing/champ.py` | `ml/preprocessing/hemophilia_a.py` |
| `train_champ.py` | `scripts/train_champ.py` | `scripts/train_inhibitor_model.py` |
| `champ-v1/` | `ml/artifacts/champ-v1/` | `ml/artifacts/mmc-{genomic,clinical,merged}-v1/` |

## Why it was replaced

CHAMP is a *variant* registry: one row per F8 variant, with `History of Inhibitor`
aggregated over the reports for that variant. MMC3 is a *clinical record* table —
one row per reported case — joined to the MMC2 mutation description. That gives
the model clinical assay values alongside the genomic description, and it changes
the unit of observation from a variant to a report.

## What must not be reused

`champ-v1/metrics.json` records **ROC-AUC 0.723 / PR-AUC 0.501 on 460 CHAMP
variants**. Those numbers describe the CHAMP model on CHAMP data. They are not
comparable to the MMC2 + MMC3 results and must never be quoted for the current
model. The current numbers live in `ml/artifacts/*/metrics.json` and in
`docs/ML.md`, and are produced by `scripts/train_inhibitor_model.py`.

`champ-v1` is also no longer discoverable by `ml.artifacts.available_versions()`,
because artifact discovery scans `ml/artifacts/` only. Requesting it raises
rather than serving a model fitted on a feature space the API no longer accepts.
