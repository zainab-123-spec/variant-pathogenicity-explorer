"""Shared fixtures: a fake HTTP session so tests never touch the network."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest


class FakeSession:
    """Stands in for :class:`vpe.http.CachedSession`.

    ``routes`` maps a substring of the URL to a function ``(url, params) -> text``.
    """

    def __init__(self, routes: dict[str, Callable[[str, dict], str | None]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def get_text(self, url: str, params: dict | None = None) -> str | None:
        self.calls.append((url, dict(params or {})))
        for key, fn in self.routes.items():
            if key in url:
                return fn(url, dict(params or {}))
        raise AssertionError(f"unexpected URL in test: {url}")

    def get_json(self, url: str, params: dict | None = None) -> Any:
        text = self.get_text(url, params)
        return None if text is None else json.loads(text)


# A tiny 12-residue "protein" used across tests.
TOY_SEQ = "MKTAYIAKQRCL"


def uniprot_entry(gene: str = "TOY", accession: str = "P00001", seq: str = TOY_SEQ) -> dict:
    return {
        "primaryAccession": accession,
        "genes": [{"geneName": {"value": gene}}],
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Toy protein"}}},
        "sequence": {"value": seq},
        "features": [
            {"type": "Domain", "location": {"start": {"value": 3}, "end": {"value": 8}}, "description": "Toy domain"},
            {"type": "Region", "location": {"start": {"value": 10}, "end": {"value": 12}}, "description": "Disordered"},
            {"type": "Binding site", "location": {"start": {"value": 5}, "end": {"value": 5}}, "description": ""},
        ],
    }


def clinvar_record(
    uid: str,
    title: str,
    cls: str,
    review: str = "criteria provided, single submitter",
    gene: str = "TOY",
    obj_type: str = "single nucleotide variant",
) -> dict:
    return {
        "uid": uid,
        "obj_type": obj_type,
        "title": title,
        "genes": [{"symbol": gene}],
        "germline_classification": {"description": cls, "review_status": review},
    }


def toy_pdb(n: int) -> str:
    """PDB text with n C-alpha atoms spaced 3.8 A apart on a line."""
    lines = []
    for i in range(1, n + 1):
        x = 3.8 * (i - 1)
        lines.append(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C")
    return "\n".join(lines)


@pytest.fixture
def toy_routes() -> dict[str, Callable[[str, dict], str | None]]:
    n = len(TOY_SEQ)
    records = {
        "1": clinvar_record("1", "NM_1(TOY):c.10A>G (p.Lys2Arg)", "Benign"),
        "2": clinvar_record("2", "NM_1(TOY):c.13G>A (p.Ala4Thr)", "Pathogenic"),
        "3": clinvar_record("3", "NM_1(TOY):c.30C>T (p.Arg10Trp)", "Likely pathogenic"),
        "4": clinvar_record("4", "NM_1(TOY):c.31C>A (p.Arg10Trp)", "Likely benign"),  # conflicts with 3
        "5": clinvar_record("5", "NM_1(TOY):c.20T>G (p.Ile6Ser)", "Uncertain significance"),
        "6": clinvar_record("6", "NM_1(TOY):c.33G>A (p.Cys11Tyr)", "Pathogenic"),
        "7": clinvar_record("7", "NM_1(TOY):c.40G>A (p.Gly12Asp)", "Pathogenic"),  # ref mismatch (pos 12 is L)
    }
    return {
        "rest.uniprot.org": lambda u, p: json.dumps({"results": [uniprot_entry()]}),
        "/api/prediction/": lambda u, p: json.dumps(
            [
                {
                    "entryId": "AF-P00001-F1",
                    "uniprotSequence": TOY_SEQ,
                    "plddtDocUrl": "https://alphafold.ebi.ac.uk/files/conf.json",
                    "pdbUrl": "https://alphafold.ebi.ac.uk/files/model.pdb",
                    "amAnnotationsUrl": "https://alphafold.ebi.ac.uk/files/am.csv",
                }
            ]
        ),
        "conf.json": lambda u, p: json.dumps({"confidenceScore": [50.0 + 4 * i for i in range(n)]}),
        "model.pdb": lambda u, p: toy_pdb(n),
        "am.csv": lambda u, p: "protein_variant,am_pathogenicity,am_class\nA4T,0.91,LPath\nK2R,0.05,LBen\n",
        "esearch.fcgi": lambda u, p: json.dumps({"esearchresult": {"idlist": list(records)}}),
        "esummary.fcgi": lambda u, p: json.dumps(
            {"result": {"uids": p["id"].split(","), **{i: records[i] for i in p["id"].split(",")}}}
        ),
    }


@pytest.fixture
def fake_session(toy_routes) -> FakeSession:  # type: ignore[no-untyped-def]
    return FakeSession(toy_routes)
