import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from vpe import model as M
from vpe.cli import main
from vpe.features import FEATURE_COLUMNS


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """Fake variant table where low pLDDT/high BLOSUM -> benign, across 10 genes."""
    rng = np.random.default_rng(0)
    n = 600
    label = rng.integers(0, 2, n)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS})
    df["plddt"] = 50 + 30 * label + rng.normal(0, 10, n)
    df["blosum62"] = -label + rng.normal(0, 1, n)
    df.loc[::17, "plddt"] = np.nan  # some missing values
    df["label"] = label
    df["gene"] = [f"G{i % 10}" for i in range(n)]
    df["protein_change"] = [f"A{i}V" for i in range(n)]
    df["pos"] = np.arange(n)
    df["alphamissense"] = np.clip(0.5 * label + rng.normal(0.25, 0.2, n), 0, 1)
    return df


def test_group_kfold_keeps_genes_separate(synthetic):
    for tr, te in GroupKFold(5).split(synthetic, groups=synthetic["gene"]):
        assert not set(synthetic.gene.iloc[tr]) & set(synthetic.gene.iloc[te])


@pytest.mark.parametrize("name", ["logistic_regression", "gradient_boosting"])
def test_cross_validate_learns_signal(synthetic, name):
    res = M.cross_validate(synthetic, M.make_models()[name], name)
    s = res.summary(synthetic["label"].to_numpy())
    assert s["roc_auc_mean"] > 0.8
    assert len(res.fold_roc_auc) == 5 and res.oof_score.shape == (len(synthetic),)


def test_baseline_direction(synthetic):
    assert M.score_baseline(synthetic, "blosum62", higher_is_pathogenic=False)["roc_auc"] > 0.6


def test_train_cli_writes_artifacts(synthetic, tmp_path):
    data = tmp_path / "variants.csv"
    synthetic.to_csv(data, index=False)
    rc = main(["train", "--data", str(data), "--out", str(tmp_path / "res"), "--figures", str(tmp_path / "fig")])
    assert rc == 0
    metrics = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert metrics["n_genes"] == 10 and metrics["n_variants"] == 600
    for f in ["roc_curves.png", "pr_curves.png", "feature_importance.png", "split_comparison.png"]:
        assert (tmp_path / "fig" / f).exists()
    assert (tmp_path / "res" / "model.pkl").exists()
    assert "<img" in (tmp_path / "res" / "index.html").read_text()
