"""Model zoo: 4 classical, 4 deep, 2 ensembles.

Every model exposes the same interface so the comparison is fair:
    fit(X_train, y_train, X_val, y_val) -> self
    predict_proba(X) -> np.ndarray of P(inhibitor)

Class imbalance is handled identically everywhere:
  * classical models  -> class_weight / scale_pos_weight = n_neg / n_pos
  * deep models       -> BCEWithLogitsLoss(pos_weight = n_neg / n_pos)
Focal loss is implemented (FocalLoss, alpha/gamma documented) and run as a
labelled sensitivity variant, not as the primary setting, so that the imbalance
strategy is identical across all ten models and all datasets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import DL, FOCAL, SEED

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------
# Preprocessing (fitted on training folds only, never on test)
# --------------------------------------------------------------------------
def make_onehot_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Dense one-hot matrix: used by the tree models, MLPs and the 1D-CNN."""
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=5, sparse_output=False,
                                 dtype=np.float64)),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    blocks = []
    if categorical:
        blocks.append(("cat", cat, categorical))
    if numeric:
        blocks.append(("num", num, numeric))
    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=True)


def make_ordinal_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Integer-coded categoricals + scaled numerics: used by TabTransformer,
    which embeds each categorical column rather than consuming one-hot columns."""
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1, encoded_missing_value=-1)),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    blocks = []
    if categorical:
        blocks.append(("cat", cat, categorical))
    if numeric:
        blocks.append(("num", num, numeric))
    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=True)


# --------------------------------------------------------------------------
# Classical models
# --------------------------------------------------------------------------
def _pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y)); neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


class ClassicalModel:
    """Wraps a scikit-learn-style estimator behind the common interface."""
    kind = "classical"

    def __init__(self, name: str, factory):
        self.name = name
        self._factory = factory
        self.model = None
        self.fit_seconds = None

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.model = self._factory(_pos_weight(ytr))
        t = time.perf_counter()
        self.model.fit(Xtr, ytr)
        self.fit_seconds = time.perf_counter() - t
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def classical_zoo() -> dict[str, ClassicalModel]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "RandomForest": ClassicalModel("RandomForest", lambda w: RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=4)),
        "XGBoost": ClassicalModel("XGBoost", lambda w: XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", scale_pos_weight=w,
            random_state=SEED, n_jobs=4, verbosity=0)),
        "LightGBM": ClassicalModel("LightGBM", lambda w: LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, class_weight="balanced", random_state=SEED,
            n_jobs=4, verbose=-1)),
        "CatBoost": ClassicalModel("CatBoost", lambda w: CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, loss_function="Logloss",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            allow_writing_files=False)),
    }


# --------------------------------------------------------------------------
# Deep models
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for binary logits. alpha weights the positive class."""
    def __init__(self, alpha: float = FOCAL["alpha"], gamma: float = FOCAL["gamma"]):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


class DeepMLP(nn.Module):
    def __init__(self, d_in: int, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ResBlock(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.BatchNorm1d(d))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.body(x))   # genuine residual connection


class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, width=128, blocks=3, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(width, dropout) for _ in range(blocks)])
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


class TabTransformer(nn.Module):
    """Transformer over per-column categorical embeddings (Huang et al., 2020).

    Each categorical column becomes one token; self-attention runs across
    columns. Numeric features are layer-normalised and concatenated at the head.
    This consumes ORDINAL-coded categoricals, not one-hot columns, which is why
    it has its own preprocessing path.
    """
    def __init__(self, cardinalities: list[int], n_numeric: int,
                 d_model=32, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.has_cat = len(cardinalities) > 0
        if self.has_cat:
            self.embeds = nn.ModuleList(
                [nn.Embedding(c + 2, d_model) for c in cardinalities])
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.n_numeric = n_numeric
        self.num_norm = nn.LayerNorm(n_numeric) if n_numeric else None
        d_head = (len(cardinalities) * d_model if self.has_cat else 0) + n_numeric
        self.head = nn.Sequential(
            nn.Linear(d_head, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x_cat, x_num):
        parts = []
        if self.has_cat:
            tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
            parts.append(self.encoder(tokens).flatten(1))
        if self.n_numeric:
            parts.append(self.num_norm(x_num))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


class CNN1D(nn.Module):
    """1-D CNN over the encoded feature vector.

    IMPORTANT: the feature axis here carries NO biological ordering - it is the
    column order of the one-hot matrix. The convolution therefore acts as a
    local weight-sharing regulariser over adjacent encoded columns, and must not
    be described as detecting spatial or sequence structure in the genome.
    """
    def __init__(self, d_in: int, channels=(32, 64), kernel=5, dropout=0.3):
        super().__init__()
        layers, prev = [], 1
        for c in channels:
            layers += [nn.Conv1d(prev, c, kernel_size=kernel, padding=kernel // 2),
                       nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            prev = c
        self.conv = nn.Sequential(*layers)
        reduced = d_in
        for _ in channels:
            reduced //= 2
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev * max(reduced, 1), 64),
            nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


@dataclass
class TrainingRecord:
    epochs_completed: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("nan")
    best_val_auc: float = float("nan")
    converged: bool = False
    history: list[dict] = field(default_factory=list)


class TorchModel:
    """Common training loop: early stopping on validation AUC, LR scheduling,
    fixed seeds, best-checkpoint restore."""
    kind = "deep"

    def __init__(self, name: str, builder, needs_ordinal: bool = False,
                 loss: str = "weighted_bce"):
        self.name = name
        self._builder = builder
        self.needs_ordinal = needs_ordinal
        self.loss_name = loss
        self.model = None
        self.fit_seconds = None
        self.record = TrainingRecord()

    def _loss_fn(self, ytr):
        if self.loss_name == "focal":
            return FocalLoss(alpha=FOCAL["alpha"], gamma=FOCAL["gamma"])
        w = torch.tensor(_pos_weight(ytr), dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=w)

    @staticmethod
    def _tensors(X):
        if isinstance(X, tuple):     # (categorical ints, numerics)
            xc, xn = X
            return (torch.tensor(xc, dtype=torch.long),
                    torch.tensor(xn, dtype=torch.float32))
        return (torch.tensor(np.asarray(X), dtype=torch.float32),)

    def _forward(self, batch):
        return self.model(*batch)

    def fit(self, Xtr, ytr, Xva, yva):
        from sklearn.metrics import roc_auc_score
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        tr = self._tensors(Xtr)
        va = self._tensors(Xva)
        ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.float32)

        self.model = self._builder().to(DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=DL["lr"],
                                weight_decay=DL["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                           factor=0.5, patience=8)
        lossf = self._loss_fn(np.asarray(ytr))

        n = len(ytr_t)
        bs = min(DL["batch_size"], max(8, n // 4))
        best_auc, best_state, best_epoch, bad = -np.inf, None, 0, 0
        t0 = time.perf_counter()

        for epoch in range(1, DL["max_epochs"] + 1):
            self.model.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:      # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                out = self._forward(tuple(t[idx] for t in tr))
                loss = lossf(out, ytr_t[idx])
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                vlogit = self._forward(va)
                vloss = float(nn.functional.binary_cross_entropy_with_logits(
                    vlogit, torch.tensor(np.asarray(yva), dtype=torch.float32)))
                vp = torch.sigmoid(vlogit).numpy()
            try:
                vauc = float(roc_auc_score(yva, vp))
            except ValueError:
                vauc = float("nan")

            self.record.history.append(
                {"epoch": epoch, "val_loss": vloss, "val_auc": vauc})
            sched.step(vauc if np.isfinite(vauc) else 0.0)

            if np.isfinite(vauc) and vauc > best_auc + 1e-5:
                best_auc, best_epoch, bad = vauc, epoch, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.record.best_val_loss = vloss
            else:
                bad += 1
                if bad >= DL["patience"]:
                    break

        self.fit_seconds = time.perf_counter() - t0
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.record.epochs_completed = epoch
        self.record.best_epoch = best_epoch
        self.record.best_val_auc = best_auc if np.isfinite(best_auc) else float("nan")
        # "converged" = early stopping actually triggered, not just ran out of epochs
        self.record.converged = bad >= DL["patience"]
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self._forward(self._tensors(X))).numpy()


def deep_zoo(d_in: int, cardinalities: list[int], n_numeric: int,
             loss: str = "weighted_bce") -> dict[str, TorchModel]:
    return {
        "DeepMLP": TorchModel("DeepMLP", lambda: DeepMLP(d_in), loss=loss),
        "ResidualMLP": TorchModel("ResidualMLP", lambda: ResidualMLP(d_in), loss=loss),
        "TabTransformer": TorchModel(
            "TabTransformer",
            lambda: TabTransformer(cardinalities, n_numeric),
            needs_ordinal=True, loss=loss),
        "1D-CNN": TorchModel("1D-CNN", lambda: CNN1D(d_in), loss=loss),
    }


MODEL_ORDER = ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
               "DeepMLP", "ResidualMLP", "TabTransformer", "1D-CNN",
               "StackingEnsemble", "WeightedEnsemble"]


def meta_learner() -> LogisticRegression:
    """Stacking meta-model: L2 logistic regression on out-of-fold probabilities.
    Simple and defensible; a complex meta-model over 8 inputs would overfit."""
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
