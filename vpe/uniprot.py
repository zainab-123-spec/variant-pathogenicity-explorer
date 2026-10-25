"""UniProt REST API: gene symbol -> canonical human protein + annotated features.

API docs: https://www.uniprot.org/help/api
We search reviewed (Swiss-Prot) human entries whose gene name is an exact
match, then keep the canonical sequence and the feature table (domains,
zinc fingers, binding sites, disordered regions, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vpe.http import CachedSession

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

#: Feature types that describe a structured, functional unit of the protein.
DOMAIN_TYPES = {"Domain", "Zinc finger", "Repeat", "DNA binding", "Motif"}
#: Feature types that mark individual functionally important residues.
SITE_TYPES = {"Active site", "Binding site", "Site", "Disulfide bond"}


@dataclass
class Feature:
    """One annotated stretch of the protein (1-based, inclusive)."""

    type: str
    start: int
    end: int
    description: str = ""

    def contains(self, position: int) -> bool:
        return self.start <= position <= self.end


@dataclass
class ProteinRecord:
    """The parts of a UniProt entry we use."""

    gene: str
    accession: str
    name: str
    sequence: str
    features: list[Feature] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def domains(self) -> list[Feature]:
        return [f for f in self.features if f.type in DOMAIN_TYPES]

    def in_domain(self, position: int) -> bool:
        return any(f.contains(position) for f in self.domains())

    def in_disordered_region(self, position: int) -> bool:
        return any(
            f.type == "Region" and "disordered" in f.description.lower() and f.contains(position) for f in self.features
        )

    def near_site(self, position: int, window: int = 3) -> bool:
        """True if a functional site lies within ``window`` residues."""
        for f in self.features:
            if f.type not in SITE_TYPES:
                continue
            # A disulfide bond is annotated by its two cysteines (start/end).
            points = [f.start, f.end] if f.type == "Disulfide bond" else range(f.start, f.end + 1)
            if any(abs(p - position) <= window for p in points):
                return True
        return False


def _parse_features(raw: list[dict[str, Any]]) -> list[Feature]:
    out: list[Feature] = []
    for f in raw:
        loc = f.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None or end is None:  # unknown boundaries
            continue
        out.append(Feature(f.get("type", ""), int(start), int(end), f.get("description", "")))
    return out


def parse_entry(gene: str, entry: dict[str, Any]) -> ProteinRecord:
    """Turn one UniProt JSON entry into a :class:`ProteinRecord`."""
    desc = entry.get("proteinDescription", {}).get("recommendedName", {})
    name = desc.get("fullName", {}).get("value", "")
    return ProteinRecord(
        gene=gene.upper(),
        accession=entry["primaryAccession"],
        name=name,
        sequence=entry["sequence"]["value"],
        features=_parse_features(entry.get("features", [])),
    )


def _primary_gene_name(entry: dict[str, Any]) -> str:
    genes = entry.get("genes") or [{}]
    return genes[0].get("geneName", {}).get("value", "").upper()


def fetch_protein(gene: str, session: CachedSession) -> ProteinRecord:
    """Fetch the reviewed human UniProt entry whose *primary* gene name is ``gene``.

    ``gene_exact`` also matches synonyms: searching "CDH1" returns FZR1 (whose
    old alias is CDH1) as well as E-cadherin. We therefore ask for a few hits
    and keep the one whose primary gene name matches exactly.

    Raises ``LookupError`` if no such entry exists.
    """
    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "format": "json",
        "size": 10,
    }
    data = session.get_json(UNIPROT_SEARCH, params)
    for entry in (data or {}).get("results", []):
        if _primary_gene_name(entry) == gene.upper():
            return parse_entry(gene, entry)
    raise LookupError(f"No reviewed human UniProt entry with primary gene name {gene!r}")
