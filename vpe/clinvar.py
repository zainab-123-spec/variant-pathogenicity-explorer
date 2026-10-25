"""ClinVar via NCBI E-utilities: missense variants with clinical labels.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
ClinVar search fields: https://www.ncbi.nlm.nih.gov/clinvar/docs/help/

Two-step pattern used by almost every E-utilities client:

1. ``esearch`` - run a query, get back a list of ClinVar variation IDs.
2. ``esummary`` - fetch summary records for those IDs in batches.

Rate limit: NCBI allows 3 requests/second without an API key (10 with
``NCBI_API_KEY``). :mod:`vpe.http` enforces that and caches everything.

Labels
------
* 1 = Pathogenic / Likely pathogenic / Pathogenic/Likely pathogenic
* 0 = Benign / Likely benign / Benign/Likely benign
* Everything else (VUS, conflicting, risk factor ...) is dropped.

By default we also drop records whose review status is "no assertion
criteria provided" (0 ClinVar stars), because those labels are the least
reliable.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vpe.aminoacids import to_one_letter
from vpe.http import CachedSession

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 200

PATHOGENIC = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN = {"benign", "likely benign", "benign/likely benign"}
LOW_QUALITY_REVIEW = {"no assertion criteria provided", "no classification provided", ""}

# e.g. "NM_000546.6(TP53):c.742C>T (p.Arg248Trp)"
_PROTEIN_CHANGE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")


@dataclass
class ClinVarVariant:
    clinvar_id: str
    gene: str
    title: str
    ref: str  # one-letter amino acid
    pos: int  # 1-based protein position
    alt: str
    classification: str
    review_status: str
    label: int  # 1 pathogenic, 0 benign

    @property
    def protein_change(self) -> str:
        return f"{self.ref}{self.pos}{self.alt}"


def build_query(gene: str) -> str:
    """ClinVar search term for missense variants with a P/LP/B/LB call.

    We deliberately do NOT add ``single_gene[prop]``: genes that overlap a
    read-through transcript (e.g. RAD51D, GLA) would return zero results.
    Instead :func:`parse_summary` checks that the record lists our gene.
    """
    return (
        f'{gene}[gene] AND "missense variant"[molecular consequence] '
        "AND (clinsig_pathogenic[prop] OR clinsig_likely_pathogenic[prop] "
        "OR clinsig_benign[prop] OR clinsig_likely_benign[prop])"
    )


def label_from_classification(description: str) -> int | None:
    d = description.strip().lower()
    if d in PATHOGENIC:
        return 1
    if d in BENIGN:
        return 0
    return None


def parse_summary(gene: str, record: dict[str, Any], min_review: bool = True) -> ClinVarVariant | None:
    """Convert one esummary record into a :class:`ClinVarVariant` (or ``None`` to skip)."""
    if record.get("obj_type") != "single nucleotide variant":
        return None
    genes = {g.get("symbol", "").upper() for g in record.get("genes", [])}
    if gene.upper() not in genes:
        return None
    # The protein change in the title must be on OUR gene's transcript,
    # e.g. "NM_000546.6(TP53):c.742C>T (p.Arg248Trp)".
    if f"({gene.upper()})" not in record.get("title", "").upper():
        return None
    germ = record.get("germline_classification") or {}
    label = label_from_classification(germ.get("description", ""))
    if label is None:
        return None
    review = germ.get("review_status", "")
    if min_review and review in LOW_QUALITY_REVIEW:
        return None
    match = _PROTEIN_CHANGE.search(record.get("title", ""))
    if not match:
        return None
    try:
        ref, alt = to_one_letter(match.group(1)), to_one_letter(match.group(3))
    except KeyError:  # stop codons etc.
        return None
    if ref == alt:
        return None
    return ClinVarVariant(
        clinvar_id=str(record["uid"]),
        gene=gene.upper(),
        title=record["title"],
        ref=ref,
        pos=int(match.group(2)),
        alt=alt,
        classification=germ.get("description", ""),
        review_status=review,
        label=label,
    )


def search_ids(gene: str, session: CachedSession, retmax: int = 10000) -> list[str]:
    data = session.get_json(
        f"{EUTILS}/esearch.fcgi",
        {"db": "clinvar", "term": build_query(gene), "retmode": "json", "retmax": retmax},
    )
    return list(data["esearchresult"]["idlist"])


def iter_summaries(ids: list[str], session: CachedSession) -> Iterator[dict[str, Any]]:
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        data = session.get_json(
            f"{EUTILS}/esummary.fcgi",
            {"db": "clinvar", "id": ",".join(batch), "retmode": "json"},
        )
        result = data.get("result", {})
        for uid in result.get("uids", []):
            yield result[uid]


def fetch_variants(gene: str, session: CachedSession, min_review: bool = True) -> list[ClinVarVariant]:
    """All labelled missense variants for ``gene``, one per protein change.

    Different DNA changes can produce the same protein change (e.g. two
    codons for Trp). If those records agree on the label we keep one; if they
    disagree (one pathogenic, one benign) we drop the protein change entirely.
    """
    by_change: dict[str, list[ClinVarVariant]] = {}
    for rec in iter_summaries(search_ids(gene, session), session):
        v = parse_summary(gene, rec, min_review=min_review)
        if v is not None:
            by_change.setdefault(v.protein_change, []).append(v)
    out: list[ClinVarVariant] = []
    for variants in by_change.values():
        if len({v.label for v in variants}) == 1:
            out.append(variants[0])
    return sorted(out, key=lambda v: (v.pos, v.alt))
