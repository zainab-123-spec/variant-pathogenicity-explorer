"""AlphaFold Protein Structure Database: per-residue confidence and 3D coordinates.

API docs: https://alphafold.ebi.ac.uk/api-docs

Two things are used per protein:

* **pLDDT** (predicted Local Distance Difference Test, 0-100) for each
  residue. High pLDDT (>70) usually means a well-folded region; very low
  pLDDT (<50) is a good predictor of *intrinsically disordered* regions.
  Mutations in folded regions are, on average, more often damaging.
* **C-alpha coordinates** from the predicted model, used to count how many
  other residues are within 10 angstroms of each residue. A high count means
  the residue is buried in the protein core, where swaps are risky.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from vpe.http import CachedSession

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


@dataclass
class StructureRecord:
    accession: str
    model_id: str
    sequence: str
    plddt: np.ndarray  # shape (L,), index i -> residue i+1
    ca_coords: np.ndarray  # shape (L, 3); NaN rows if a residue is missing

    def plddt_at(self, position: int) -> float:
        return float(self.plddt[position - 1])

    def plddt_window(self, position: int, half_width: int = 5) -> float:
        lo = max(0, position - 1 - half_width)
        hi = min(len(self.plddt), position + half_width)
        return float(np.mean(self.plddt[lo:hi]))

    def contact_counts(self, radius: float = 10.0) -> np.ndarray:
        """Number of other C-alpha atoms within ``radius`` angstroms of each residue."""
        xyz = self.ca_coords
        diff = xyz[:, None, :] - xyz[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        within = (dist < radius) & ~np.isnan(dist)
        return within.sum(axis=1) - 1  # minus itself


def parse_ca_coords(pdb_text: str, length: int) -> np.ndarray:
    """Extract C-alpha coordinates from PDB text into an (L, 3) array."""
    coords = np.full((length, 3), np.nan)
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            resnum = int(line[22:26])
            if 1 <= resnum <= length:
                coords[resnum - 1] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return coords


def fetch_structure(accession: str, session: CachedSession) -> StructureRecord | None:
    """Fetch the AlphaFold model for a UniProt accession.

    Returns ``None`` when AlphaFold DB has no single-chain model (e.g. very
    long proteins that are split into fragments).
    """
    meta = session.get_json(AFDB_API.format(accession=accession))
    if not meta:
        return None
    entry = meta[0]
    conf_text = session.get_text(entry["plddtDocUrl"])
    pdb_text = session.get_text(entry["pdbUrl"])
    if conf_text is None or pdb_text is None:
        return None
    conf = json.loads(conf_text)
    sequence = entry.get("uniprotSequence") or entry.get("sequence", "")
    plddt = np.asarray(conf["confidenceScore"], dtype=float)
    return StructureRecord(
        accession=accession,
        model_id=entry.get("entryId", ""),
        sequence=sequence,
        plddt=plddt,
        ca_coords=parse_ca_coords(pdb_text, len(plddt)),
    )


def fetch_alphamissense(accession: str, session: CachedSession) -> dict[str, float]:
    """AlphaMissense scores (``{'R248W': 0.99, ...}``) for an accession.

    Used ONLY as an external reference point to compare against, never as
    a model feature. Returns an empty dict if unavailable.
    """
    meta = session.get_json(AFDB_API.format(accession=accession))
    if not meta or not meta[0].get("amAnnotationsUrl"):
        return {}
    text = session.get_text(meta[0]["amAnnotationsUrl"])
    if not text:
        return {}
    scores: dict[str, float] = {}
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2:
            scores[parts[0]] = float(parts[1])
    return scores
