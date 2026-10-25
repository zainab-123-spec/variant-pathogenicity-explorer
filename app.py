"""Interactive explorer.  Run:  streamlit run app.py

Pick a gene in the sidebar to see its AlphaFold confidence, UniProt domains,
ClinVar variants and model predictions. Data are fetched from UniProt /
AlphaFold DB / ClinVar on first use and cached in data/cache/.
Open with ?gene=TP53 in the URL to pre-select a gene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `vpe` importable

from vpe.cli import cmd_analyze  # noqa: E402
from vpe.genes import PANEL  # noqa: E402

RESULTS, REPORTS, FIGS = Path("results"), Path("reports"), Path("docs/figures")

st.set_page_config(page_title="Variant pathogenicity explorer", layout="wide")
st.title("🧬 Variant pathogenicity explorer")
st.caption(
    "UniProt + AlphaFold DB + ClinVar → interpretable features → baseline classifier. "
    "Research and education only, not for clinical use."
)

if not (RESULTS / "model.pkl").exists():
    st.error("No trained model yet. Run `python -m vpe build-dataset` and `python -m vpe train` first.")
    st.stop()


@st.cache_data(show_spinner=False)
def analyse(gene: str) -> pd.DataFrame:
    """Run the same code path as `python -m vpe analyze GENE` and load its table."""
    args = argparse.Namespace(gene=gene, results=str(RESULTS), out=str(REPORTS), saturation=True, offline=False)
    cmd_analyze(args)
    return pd.read_csv(REPORTS / f"{gene}_variants.csv")


# ----------------------------------------------------------------- sidebar
default = st.query_params.get("gene", "BRCA1").upper()
with st.sidebar:
    st.header("Choose a gene")
    options = sorted(set(PANEL) | {default})
    picked = st.selectbox("Training-panel gene", options, index=options.index(default))
    typed = st.text_input("…or type any human gene symbol", "").strip().upper()
    gene = typed or picked
    st.markdown(
        "Panel genes are scored **out-of-fold** (the model never saw them during training). "
        "Other genes are scored by the model trained on all 42 panel genes."
    )

# -------------------------------------------------------------- gene view
st.subheader(f"{gene}")
try:
    with st.spinner(f"Fetching and scoring {gene} …"):
        table = analyse(gene)
except Exception as exc:  # API / lookup errors are shown, not crashed on
    st.error(f"Could not analyse {gene}: {exc}")
    st.stop()

if len(table) and table["label"].nunique() == 2:
    from sklearn.metrics import average_precision_score, roc_auc_score

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ClinVar variants", len(table))
    c2.metric("Pathogenic / likely pathogenic", int(table["label"].sum()))
    c3.metric("ROC-AUC on this gene", f"{roc_auc_score(table['label'], table['score']):.3f}")
    c4.metric("PR-AUC on this gene", f"{average_precision_score(table['label'], table['score']):.3f}")

st.image(str(REPORTS / f"{gene}_track.png"), width="stretch")

if len(table):
    show = ["protein_change", "classification", "review_status", "plddt", "contacts_10A", "blosum62", "score"]
    show += ["alphamissense"] if "alphamissense" in table else []
    st.markdown("**Variants, highest predicted pathogenicity first**")
    st.dataframe(
        table.sort_values("score", ascending=False)[show].round(3), width="stretch", height=320, hide_index=True
    )

# --------------------------------------------------------- model results
with st.expander("Model performance (5-fold cross-validation)"):
    st.dataframe(pd.read_csv(RESULTS / "cv_metrics.csv").round(3), width="stretch")
    st.dataframe(pd.read_csv(RESULTS / "baselines.csv").round(3), width="stretch")
    cols = st.columns(2)
    for i, name in enumerate(["roc_curves", "split_comparison", "feature_importance", "per_gene_auc"]):
        if (FIGS / f"{name}.png").exists():
            cols[i % 2].image(str(FIGS / f"{name}.png"))
