"""Figures (PNG) and a self-contained static HTML report."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from vpe.uniprot import ProteinRecord  # noqa: E402

PATH_COLOR, BENIGN_COLOR = "#c0392b", "#2471a3"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_curves(y: np.ndarray, scores: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    """ROC and precision-recall curves for several score vectors."""
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    for label, s in scores.items():
        mask = ~np.isnan(s)
        fpr, tpr, _ = roc_curve(y[mask], s[mask])
        prec, rec, _ = precision_recall_curve(y[mask], s[mask])
        ax1.plot(fpr, tpr, label=label)
        ax2.plot(rec, prec, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (gene-held-out, pooled)")
    ax2.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"prevalence = {y.mean():.2f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (gene-held-out)")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.legend(fontsize=8, loc="lower left")
    return [_save(fig1, out_dir / "roc_curves.png"), _save(fig2, out_dir / "pr_curves.png")]


def plot_importance(perm: pd.DataFrame, coefs: pd.Series, out_dir: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5))
    p = perm.sort_values("importance_mean")
    a.barh(p["feature"], p["importance_mean"], xerr=p["importance_sd"], color="#7d3c98")
    a.set(title="Gradient boosting: permutation importance\n(drop in held-out ROC-AUC)", xlabel="ΔROC-AUC")
    b.barh(coefs.index, coefs.values, color=[PATH_COLOR if v > 0 else BENIGN_COLOR for v in coefs.values])
    b.axvline(0, color="k", lw=0.8)
    b.set(title="Logistic regression: standardised coefficients\n(+ = pushes towards pathogenic)", xlabel="coefficient")
    return _save(fig, out_dir / "feature_importance.png")


def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, bins in zip(axes, ["plddt", "contacts_10A", "blosum62"], [30, 30, np.arange(-4.5, 4)], strict=True):
        for lab, color, name in [(1, PATH_COLOR, "pathogenic"), (0, BENIGN_COLOR, "benign")]:
            ax.hist(df.loc[df.label == lab, col].dropna(), bins=bins, alpha=0.55, density=True, color=color, label=name)
        ax.set(title=col, ylabel="density")
    axes[0].legend()
    return _save(fig, out_dir / "feature_distributions.png")


def plot_leakage(summary: pd.DataFrame, out_dir: Path) -> Path:
    piv = summary.pivot(index="model", columns="split", values="roc_auc_mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    piv.plot.bar(ax=ax, rot=0, color=["#1e8449", "#b9770e"])
    ax.set(ylim=(0.5, 1.0), ylabel="mean ROC-AUC (5 folds)", title="Random split vs gene-held-out split")
    ax.legend(title="split", loc="lower right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    return _save(fig, out_dir / "split_comparison.png")


def plot_per_gene(per_gene: pd.DataFrame, out_dir: Path) -> Path:
    pg = per_gene.dropna(subset=["roc_auc"]).sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(6, max(4, 0.22 * len(pg))))
    ax.barh(pg["gene"], pg["roc_auc"], color="#5d6d7e")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set(xlim=(0.3, 1.0), xlabel="ROC-AUC on this gene when it was held out", title="Per-gene performance")
    return _save(fig, out_dir / "per_gene_auc.png")


def plot_gene_track(
    protein: ProteinRecord,
    plddt: np.ndarray | None,
    variants: pd.DataFrame,
    path: Path,
    position_score: np.ndarray | None = None,
) -> Path:
    """Lollipop-style overview: pLDDT, domains, ClinVar variants and model scores."""
    L = protein.length
    x = np.arange(1, L + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 3]})
    if plddt is not None:
        ax1.fill_between(x, plddt, color="#f5b041", alpha=0.6, label="AlphaFold pLDDT")
        ax1.axhline(70, color="grey", ls=":", lw=0.8)
    if position_score is not None:
        ax1b = ax1.twinx()
        ax1b.plot(x, position_score, color="#6c3483", lw=0.9, label="mean predicted P(pathogenic), all 19 swaps")
        ax1b.set_ylim(0, 1)
        ax1b.set_ylabel("model score", color="#6c3483")
    ax1.set(ylim=(0, 100), ylabel="pLDDT", title=f"{protein.gene} ({protein.accession}) - {protein.name}")
    for d in protein.domains():
        ax1.axvspan(d.start, d.end, color="#48c9b0", alpha=0.25)
        ax1.text((d.start + d.end) / 2, 95, d.description[:18], ha="center", va="top", fontsize=6)
    if not variants.empty:
        for lab, color, name in [
            (0, BENIGN_COLOR, "benign / likely benign"),
            (1, PATH_COLOR, "pathogenic / likely pathogenic"),
        ]:
            v = variants[variants.label == lab]
            ax2.scatter(v["pos"], v["score"], s=14, color=color, alpha=0.7, label=f"{name} (n={len(v)})")
        ax2.legend(fontsize=8, loc="lower right")
    ax2.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax2.set(ylim=(-0.02, 1.02), xlim=(1, L), xlabel="residue position", ylabel="predicted P(pathogenic)")
    return _save(fig, path)


def _img_tag(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%">'


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;
  margin:2rem auto;padding:0 1rem;color:#222}
table{border-collapse:collapse;font-size:13px;margin:1rem 0}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}
th{background:#f4f6f7}td:first-child,th:first-child{text-align:left}
.note{background:#fef9e7;padding:.6rem 1rem;border-left:4px solid #f1c40f}
"""


def write_html(path: Path, title: str, sections: list[tuple[str, Any]]) -> Path:
    """Write a standalone HTML page. Each section is (heading, content) where
    content is a DataFrame, an image Path, or an HTML string."""
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style></head><body><h1>{html.escape(title)}</h1>"
    ]
    for heading, content in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            parts.append(content.to_html(index=False, float_format=lambda v: f"{v:.3f}", border=0))
        elif isinstance(content, Path):
            parts.append(_img_tag(content))
        else:
            parts.append(str(content))
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
