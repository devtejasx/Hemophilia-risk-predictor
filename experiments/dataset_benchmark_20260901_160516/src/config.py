"""Experiment configuration. Analysis-only; nothing here touches production."""
from __future__ import annotations
import os
from pathlib import Path

SEED = 42
N_FOLDS = 5
TEST_SIZE = 0.20          # held-out test, never touched until final evaluation
VAL_SIZE = 0.20           # of the remaining train portion

REPO = Path(__file__).resolve().parents[3]
EXTERNAL_DATA = Path(r"C:\Users\Admin\Downloads\Hemophilia A")
CHAMP_CSV = REPO / "ml" / "data" / "champ.csv"

EXP_DIR = Path(__file__).resolve().parents[1]
for sub in ("configs","logs","metrics","plots","models","predictions","explanations","tables","report"):
    (EXP_DIR / sub).mkdir(parents=True, exist_ok=True)

# Deep-learning defaults, identical across datasets for fairness
DL = dict(max_epochs=200, patience=25, batch_size=64, lr=1e-3, weight_decay=1e-4)
FOCAL = dict(alpha=0.25, gamma=2.0)   # documented; used for DL class imbalance
