"""Glue: fetch everything for a gene and assemble a feature table."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from vpe.alphafold import StructureRecord, fetch_alphamissense, fetch_structure
from vpe.clinvar import fetch_variants
from vpe.features import variant_features
from vpe.http import CachedSession
from vpe.uniprot import ProteinRecord, fetch_protein

log = logging.getLogger(__name__)


@dataclass
class GeneData:
    protein: ProteinRecord
    structure: StructureRecord | None
    table: pd.DataFrame  # one row per variant
    n_clinvar: int  # labelled missense variants returned by ClinVar
    n_ref_mismatch: int  # dropped because ref residue != UniProt sequence


def build_gene(gene: str, session: CachedSession, with_alphamissense: bool = True) -> GeneData:
    """Fetch UniProt + AlphaFold + ClinVar for one gene and build its feature table."""
    protein = fetch_protein(gene, session)
    structure = fetch_structure(protein.accession, session)
    if structure is not None and structure.sequence != protein.sequence:
        log.warning("%s: AlphaFold sequence differs from UniProt; structure features set to NaN", gene)
    contacts = structure.contact_counts() if structure is not None else None
    variants = fetch_variants(gene, session)

    rows = []
    mismatches = 0
    for v in variants:
        row = variant_features(v, protein, structure, contacts)
        if row is None:
            mismatches += 1
            continue
        rows.append(row)
    table = pd.DataFrame(rows)

    if with_alphamissense and not table.empty:
        am = fetch_alphamissense(protein.accession, session)
        table["alphamissense"] = table["protein_change"].map(am)
    return GeneData(protein, structure, table, len(variants), mismatches)


def build_panel(genes: list[str], session: CachedSession) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the combined variant table and a per-gene QC summary."""
    tables, qc = [], []
    for gene in genes:
        try:
            gd = build_gene(gene, session)
        except (LookupError, RuntimeError) as exc:
            log.error("%s: skipped (%s)", gene, exc)
            qc.append({"gene": gene, "status": f"skipped: {exc}"})
            continue
        t = gd.table
        n_path = int(t["label"].sum()) if not t.empty else 0
        qc.append(
            {
                "gene": gene,
                "uniprot": gd.protein.accession,
                "length": gd.protein.length,
                "alphafold": gd.structure is not None,
                "clinvar_labelled": gd.n_clinvar,
                "ref_mismatch_dropped": gd.n_ref_mismatch,
                "variants": len(t),
                "pathogenic": n_path,
                "benign": len(t) - n_path,
                "status": "ok",
            }
        )
        log.info("%s: %d variants (%d pathogenic)", gene, len(t), n_path)
        tables.append(t)
    data = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return data, pd.DataFrame(qc)
