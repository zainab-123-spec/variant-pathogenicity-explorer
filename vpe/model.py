"""Train and evaluate baseline classifiers with a gene-held-out split.

Why GroupKFold by gene?
-----------------------
Variants from the same gene share a lot: the same protein length, the same
pLDDT profile, often the same ratio of pathogenic to benign calls. If
variants of BRCA1 are in both the training and test folds, the model can
"recognise BRCA1" instead of learning what makes a substitution damaging,
and the test score is inflated. GroupKFold keeps every gene entirely in
either the training or the test fold, so the score estimates performance
on a gene the model has never seen. We also run an ordinary stratified
random split purely to *show* how much leakage inflates the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vpe.features import FEATURE_COLUMNS

RANDOM_STATE = 42


def make_models() -> dict[str, Pipeline]:
    """The two baseline models, each wrapped in a preprocessing pipeline.

    Putting the imputer/scaler *inside* the pipeline means they are re-fit
    on the training fold only, which is another way of avoiding leakage.
    """
    logreg = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    gboost = Pipeline(
        [
            # HistGradientBoosting handles NaN natively; no imputer needed.
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    return {"logistic_regression": logreg, "gradient_boosting": gboost}


@dataclass
class CVResult:
    model: str
    split: str
    fold_roc_auc: list[float] = field(default_factory=list)
    fold_pr_auc: list[float] = field(default_factory=list)
    oof_score: np.ndarray | None = None  # out-of-fold predicted probability

    def summary(self, y: np.ndarray) -> dict[str, Any]:
        return {
            "model": self.model,
            "split": self.split,
            "roc_auc_mean": float(np.mean(self.fold_roc_auc)),
            "roc_auc_sd": float(np.std(self.fold_roc_auc)),
            "pr_auc_mean": float(np.mean(self.fold_pr_auc)),
            "pr_auc_sd": float(np.std(self.fold_pr_auc)),
            "roc_auc_pooled": float(roc_auc_score(y, self.oof_score)),
            "pr_auc_pooled": float(average_precision_score(y, self.oof_score)),
        }


def cross_validate(
    df: pd.DataFrame,
    model: Pipeline,
    name: str,
    split: str = "gene_held_out",
    n_splits: int = 5,
    features: list[str] | None = None,
) -> CVResult:
    """Cross-validated out-of-fold predictions and per-fold metrics."""
    cols = features or FEATURE_COLUMNS
    X, y, groups = df[cols].to_numpy(float), df["label"].to_numpy(int), df["gene"].to_numpy()
    if split == "gene_held_out":
        splitter = GroupKFold(n_splits=n_splits)
        folds = splitter.split(X, y, groups)
    elif split == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        folds = splitter.split(X, y)
    else:
        raise ValueError(split)

    res = CVResult(model=name, split=split, oof_score=np.zeros(len(y)))
    for train_idx, test_idx in folds:
        m = clone(model).fit(X[train_idx], y[train_idx])
        p = m.predict_proba(X[test_idx])[:, 1]
        res.oof_score[test_idx] = p
        res.fold_roc_auc.append(roc_auc_score(y[test_idx], p))
        res.fold_pr_auc.append(average_precision_score(y[test_idx], p))
    return res


def score_baseline(df: pd.DataFrame, column: str, higher_is_pathogenic: bool = True) -> dict[str, Any]:
    """ROC/PR-AUC of a single raw score (no training), on rows where it exists."""
    sub = df.dropna(subset=[column])
    s = sub[column].to_numpy(float) * (1 if higher_is_pathogenic else -1)
    y = sub["label"].to_numpy(int)
    return {
        "score": column,
        "n": int(len(sub)),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }


def per_gene_auc(df: pd.DataFrame, score: np.ndarray, min_each: int = 5) -> pd.DataFrame:
    """ROC-AUC within each held-out gene (needs ``min_each`` of both classes)."""
    rows = []
    for gene, idx in df.groupby("gene").indices.items():
        y = df["label"].to_numpy()[idx]
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        auc = roc_auc_score(y, score[idx]) if min(n_pos, n_neg) >= min_each else np.nan
        rows.append({"gene": gene, "n": len(y), "pathogenic": n_pos, "benign": n_neg, "roc_auc": auc})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def logreg_coefficients(df: pd.DataFrame) -> pd.Series:
    """Standardised logistic-regression coefficients fit on the full data."""
    m = make_models()["logistic_regression"].fit(df[FEATURE_COLUMNS].to_numpy(float), df["label"])
    return pd.Series(m.named_steps["clf"].coef_[0], index=FEATURE_COLUMNS).sort_values()


def grouped_permutation_importance(df: pd.DataFrame, model: Pipeline, n_splits: int = 5) -> pd.DataFrame:
    """Permutation importance (drop in ROC-AUC) measured on held-out genes.

    For each fold we fit on the training genes, then shuffle one feature at
    a time in the *test* genes and record how much ROC-AUC falls. Averaging
    over folds gives an importance that reflects generalisation, not
    memorisation.
    """
    X, y, groups = df[FEATURE_COLUMNS].to_numpy(float), df["label"].to_numpy(int), df["gene"].to_numpy()
    imps = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model).fit(X[train_idx], y[train_idx])
        r = permutation_importance(
            m, X[test_idx], y[test_idx], scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
        )
        imps.append(r.importances_mean)
    arr = np.array(imps)
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mean": arr.mean(0), "importance_sd": arr.std(0)})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def fit_final(df: pd.DataFrame, name: str = "gradient_boosting") -> Pipeline:
    """Fit a model on all panel variants, for scoring genes outside the panel."""
    return make_models()[name].fit(df[FEATURE_COLUMNS].to_numpy(float), df["label"].to_numpy(int))
