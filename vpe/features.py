"""Turn (protein, structure, variant) into a row of numeric features.

Every feature is something a biologist could explain on a whiteboard.
See LEARNING_GUIDE.md for the biology behind each one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vpe.alphafold import StructureRecord
from vpe.aminoacids import CHARGE, KYTE_DOOLITTLE, VOLUME, blosum62
from vpe.clinvar import ClinVarVariant
from vpe.uniprot import ProteinRecord

#: Columns fed to the model, in a fixed order.
FEATURE_COLUMNS: list[str] = [
    "plddt",
    "plddt_window",
    "contacts_10A",
    "rel_position",
    "blosum62",
    "hydrophobicity_delta",
    "abs_hydrophobicity_delta",
    "volume_delta",
    "abs_volume_delta",
    "charge_change",
    "in_domain",
    "in_disordered_region",
    "near_functional_site",
    "involves_gly_or_pro",
    "involves_cys",
]


def substitution_features(ref: str, alt: str) -> dict[str, float]:
    """Features that depend only on the amino-acid swap, not on where it is."""
    hyd = KYTE_DOOLITTLE[alt] - KYTE_DOOLITTLE[ref]
    vol = VOLUME[alt] - VOLUME[ref]
    return {
        "blosum62": float(blosum62(ref, alt)),
        "hydrophobicity_delta": hyd,
        "abs_hydrophobicity_delta": abs(hyd),
        "volume_delta": vol,
        "abs_volume_delta": abs(vol),
        "charge_change": float(abs(CHARGE[alt] - CHARGE[ref])),
        "involves_gly_or_pro": float(ref in "GP" or alt in "GP"),
        "involves_cys": float(ref == "C" or alt == "C"),
    }


def variant_features(
    variant: ClinVarVariant,
    protein: ProteinRecord,
    structure: StructureRecord | None,
    contacts: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Full feature row for one variant, or ``None`` if it fails QC.

    QC: the reference amino acid reported by ClinVar must match the UniProt
    canonical sequence at that position. A mismatch means ClinVar used a
    different isoform numbering, and the variant would be placed at the
    wrong residue, so we drop it.
    """
    pos = variant.pos
    if not (1 <= pos <= protein.length) or protein.sequence[pos - 1] != variant.ref:
        return None

    structure_ok = structure is not None and structure.sequence == protein.sequence
    row: dict[str, Any] = {
        "gene": variant.gene,
        "uniprot": protein.accession,
        "clinvar_id": variant.clinvar_id,
        "protein_change": variant.protein_change,
        "pos": pos,
        "ref": variant.ref,
        "alt": variant.alt,
        "classification": variant.classification,
        "review_status": variant.review_status,
        "label": variant.label,
        "plddt": structure.plddt_at(pos) if structure_ok else math.nan,
        "plddt_window": structure.plddt_window(pos) if structure_ok else math.nan,
        "contacts_10A": float(contacts[pos - 1]) if (structure_ok and contacts is not None) else math.nan,
        "rel_position": pos / protein.length,
        "in_domain": float(protein.in_domain(pos)),
        "in_disordered_region": float(protein.in_disordered_region(pos)),
        "near_functional_site": float(protein.near_site(pos)),
    }
    row.update(substitution_features(variant.ref, variant.alt))
    return row
