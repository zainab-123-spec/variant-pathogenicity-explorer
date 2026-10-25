"""Command-line interface.

python -m vpe fetch TP53 BRCA1          # download + cache, print a summary
python -m vpe build-dataset             # build features for the training panel
python -m vpe train                     # cross-validate, write metrics/figures/model
python -m vpe analyze BRCA1             # per-gene HTML report
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from vpe import model as M
from vpe import report as R
from vpe.aminoacids import AMINO_ACIDS
from vpe.dataset import build_gene, build_panel
from vpe.features import FEATURE_COLUMNS
from vpe.genes import PANEL
from vpe.http import CachedSession

log = logging.getLogger("vpe")
DATA = Path("data/processed/variants.csv")
RESULTS = Path("results")
FIGURES = Path("docs/figures")


# --------------------------------------------------------------------- fetch
def cmd_fetch(args: argparse.Namespace) -> int:
    s = CachedSession(offline=args.offline)
    for gene in args.genes:
        gd = build_gene(gene, s)
        t = gd.table
        n_p = int(t["label"].sum()) if not t.empty else 0
        print(
            f"{gene}: UniProt {gd.protein.accession} ({gd.protein.length} aa), "
            f"AlphaFold {'yes' if gd.structure is not None else 'no'}, "
            f"{len(t)} variants ({n_p} pathogenic, {len(t) - n_p} benign), "
            f"{gd.n_ref_mismatch} dropped for reference mismatch"
        )
    return 0


# ------------------------------------------------------------- build-dataset
def cmd_build(args: argparse.Namespace) -> int:
    genes = args.genes or PANEL
    data, qc = build_panel(genes, CachedSession(offline=args.offline))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    qc.to_csv(Path(args.out).with_name("gene_qc.csv"), index=False)
    print(qc.to_string(index=False))
    print(f"\n{len(data)} variants from {data['gene'].nunique()} genes -> {args.out}")
    return 0


# --------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)
    # Need both classes in a gene for it to be informative in per-gene metrics,
    # but every gene is used for training/evaluation.
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    y = df["label"].to_numpy(int)
    out, figs = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)

    summaries, oof = [], {}
    for split in ["gene_held_out", "random"]:
        for name, mdl in M.make_models().items():
            res = M.cross_validate(df, mdl, name, split=split, n_splits=args.folds)
            summaries.append(res.summary(y))
            if split == "gene_held_out":
                oof[name] = res.oof_score
            log.info("%s / %s: ROC-AUC %.3f", split, name, summaries[-1]["roc_auc_mean"])
    summary = pd.DataFrame(summaries)

    baselines = pd.DataFrame(
        [
            M.score_baseline(df, "blosum62", higher_is_pathogenic=False),
            M.score_baseline(df, "plddt"),
            M.score_baseline(df, "alphamissense"),
        ]
    )
    best = "gradient_boosting"
    per_gene = M.per_gene_auc(df, oof[best])
    perm = M.grouped_permutation_importance(df, M.make_models()[best], n_splits=args.folds)
    coefs = M.logreg_coefficients(df)

    # Save artefacts
    summary.to_csv(out / "cv_metrics.csv", index=False)
    baselines.to_csv(out / "baselines.csv", index=False)
    per_gene.to_csv(out / "per_gene_auc.csv", index=False)
    perm.to_csv(out / "permutation_importance.csv", index=False)
    coefs.rename("coefficient").to_csv(out / "logreg_coefficients.csv")
    oof_df = df[["gene", "protein_change", "pos", "label"]].copy()
    for name, s in oof.items():
        oof_df[f"oof_{name}"] = s
    oof_df.to_csv(out / "oof_predictions.csv", index=False)
    meta = {
        "n_variants": int(len(df)),
        "n_genes": int(df["gene"].nunique()),
        "n_pathogenic": int(y.sum()),
        "n_benign": int(len(y) - y.sum()),
        "folds": args.folds,
        "features": FEATURE_COLUMNS,
        "cv": summaries,
        "baselines": baselines.to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    final = M.fit_final(df, best)
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(final, fh)

    scores = {
        "Gradient boosting": oof["gradient_boosting"],
        "Logistic regression": oof["logistic_regression"],
        "BLOSUM62 only (no training)": -df["blosum62"].to_numpy(float),
        "pLDDT only (no training)": df["plddt"].to_numpy(float),
    }
    R.plot_curves(y, scores, figs)
    R.plot_importance(perm, coefs, figs)
    R.plot_feature_distributions(df, figs)
    R.plot_leakage(summary, figs)
    R.plot_per_gene(per_gene, figs)

    held = summary[summary.split == "gene_held_out"]
    R.write_html(
        out / "index.html",
        "variant-pathogenicity-explorer: training report",
        [
            (
                "Dataset",
                f"<p>{meta['n_variants']} ClinVar missense variants from {meta['n_genes']} genes "
                f"({meta['n_pathogenic']} pathogenic/likely pathogenic, {meta['n_benign']} benign/likely benign).</p>",
            ),
            ("Cross-validation (5-fold, gene-held-out and random)", summary),
            ("Single-score baselines (no training)", baselines),
            ("ROC curves", figs / "roc_curves.png"),
            ("Precision-recall curves", figs / "pr_curves.png"),
            ("Why the split matters", figs / "split_comparison.png"),
            ("Feature importance", figs / "feature_importance.png"),
            ("Feature distributions", figs / "feature_distributions.png"),
            ("Per-gene ROC-AUC (held-out)", figs / "per_gene_auc.png"),
            ("Per-gene table", per_gene),
        ],
    )
    print(held.to_string(index=False))
    print(baselines.to_string(index=False))
    print(f"\nwrote {out}/metrics.json, {out}/index.html and figures in {figs}/")
    return 0


# ------------------------------------------------------------------- analyze
def saturation_scores(gd, mdl) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Mean predicted P(pathogenic) over all 19 possible swaps at each residue."""
    from vpe.clinvar import ClinVarVariant
    from vpe.features import variant_features

    contacts = gd.structure.contact_counts() if gd.structure is not None else None
    rows = []
    for pos, ref in enumerate(gd.protein.sequence, start=1):
        if ref not in AMINO_ACIDS:
            continue
        for alt in AMINO_ACIDS:
            if alt == ref:
                continue
            v = ClinVarVariant("", gd.protein.gene, "", ref, pos, alt, "", "", 0)
            row = variant_features(v, gd.protein, gd.structure, contacts)
            if row:
                rows.append(row)
    sat = pd.DataFrame(rows)
    sat["score"] = mdl.predict_proba(sat[FEATURE_COLUMNS].to_numpy(float))[:, 1]
    per_pos = sat.groupby("pos")["score"].mean()
    return per_pos.reindex(range(1, gd.protein.length + 1)).to_numpy()


def cmd_analyze(args: argparse.Namespace) -> int:
    gene = args.gene.upper()
    gd = build_gene(gene, CachedSession(offline=args.offline))
    with open(Path(args.results) / "model.pkl", "rb") as fh:
        mdl = pickle.load(fh)
    t = gd.table.copy()
    note = ""
    oof_path = Path(args.results) / "oof_predictions.csv"
    oof = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame(columns=["gene"])
    if not t.empty and gene in set(oof["gene"]):
        # Honest scores: from the CV fold where this gene was held out.
        m = oof[oof.gene == gene].set_index("protein_change")["oof_gradient_boosting"]
        t["score"] = t["protein_change"].map(m)
        note = (
            f"{gene} is in the training panel, so ClinVar variant scores shown are <b>out-of-fold</b> "
            "predictions from the cross-validation fold in which this gene was held out."
        )
    elif not t.empty:
        t["score"] = mdl.predict_proba(t[FEATURE_COLUMNS].to_numpy(float))[:, 1]
        note = f"{gene} was not in the training panel; scores come from the model trained on all panel genes."
    else:
        t = pd.DataFrame(columns=["pos", "label", "score"])
    pos_score = saturation_scores(gd, mdl) if args.saturation else None
    if pos_score is not None:
        note += " The purple line in the top panel scores every possible substitution with the final model" + (
            " (which was trained on this gene, so treat it as illustrative)." if gene in set(oof["gene"]) else "."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plddt = gd.structure.plddt if gd.structure is not None and gd.structure.sequence == gd.protein.sequence else None
    track = R.plot_gene_track(gd.protein, plddt, t, out / f"{gene}_track.png", pos_score)
    t.to_csv(out / f"{gene}_variants.csv", index=False)

    sections: list[tuple[str, object]] = [
        (
            "Summary",
            f"<p>UniProt <a href='https://www.uniprot.org/uniprotkb/{gd.protein.accession}'>"
            f"{gd.protein.accession}</a>, {gd.protein.length} aa. "
            f"{len(t)} labelled ClinVar missense variants. "
            + (
                f"Mean AlphaFold pLDDT: {np.mean(plddt):.1f}.</p>"
                if plddt is not None
                else "No single-chain AlphaFold model matched this sequence, so structure features are missing "
                "(the model falls back on sequence-only features).</p>"
            ),
        ),
        ("Note", f"<p class='note'>{note} Research/education only - not for clinical use.</p>"),
        ("Protein overview", track),
    ]
    if len(t) and t["label"].nunique() == 2 and t["score"].notna().all():
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = roc_auc_score(t["label"], t["score"])
        ap = average_precision_score(t["label"], t["score"])
        sections.append(("Performance on this gene", f"<p>ROC-AUC {auc:.3f}, PR-AUC {ap:.3f}</p>"))
        print(f"{gene}: ROC-AUC {auc:.3f}  PR-AUC {ap:.3f}  (n={len(t)})")
    if len(t):
        cols = [
            "protein_change",
            "classification",
            "review_status",
            "plddt",
            "contacts_10A",
            "blosum62",
            "in_domain",
            "score",
        ] + (["alphamissense"] if "alphamissense" in t else [])
        sections.append(("Variants (highest model score first)", t.sort_values("score", ascending=False)[cols]))
    page = R.write_html(out / f"{gene}.html", f"{gene}: variant pathogenicity explorer", sections)
    print(f"wrote {page}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vpe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--offline", action="store_true", help="use cached API responses only")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download and summarise data for genes")
    f.add_argument("genes", nargs="+")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("build-dataset", help="build the feature table for the panel")
    b.add_argument("--genes", nargs="*", help=f"defaults to the {len(PANEL)}-gene panel in vpe/genes.py")
    b.add_argument("--out", default=str(DATA))
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="cross-validate models and write results")
    t.add_argument("--data", default=str(DATA))
    t.add_argument("--out", default=str(RESULTS))
    t.add_argument("--figures", default=str(FIGURES))
    t.add_argument("--folds", type=int, default=5)
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("analyze", help="HTML report for one gene")
    a.add_argument("gene")
    a.add_argument("--results", default=str(RESULTS))
    a.add_argument("--out", default="reports")
    a.add_argument(
        "--no-saturation",
        dest="saturation",
        action="store_false",
        help="skip scoring all 19 possible substitutions per residue",
    )
    a.set_defaults(func=cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return int(args.func(args))
