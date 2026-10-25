import json

import numpy as np
import pytest
from conftest import TOY_SEQ, FakeSession, toy_pdb, uniprot_entry

from vpe.alphafold import fetch_alphamissense, fetch_structure, parse_ca_coords
from vpe.uniprot import fetch_protein, parse_entry


def test_parse_entry_and_feature_queries():
    p = parse_entry("toy", uniprot_entry())
    assert p.gene == "TOY" and p.length == len(TOY_SEQ)
    assert p.in_domain(3) and p.in_domain(8) and not p.in_domain(9)
    assert p.in_disordered_region(11) and not p.in_disordered_region(2)
    assert p.near_site(5) and p.near_site(8) and not p.near_site(9)


def test_fetch_protein_ignores_synonym_hits():
    # "CDH1" is also an old alias of FZR1; the right entry has primary name CDH1.
    wrong = uniprot_entry(gene="FZR1", accession="Q9UM11")
    right = uniprot_entry(gene="CDH1", accession="P12830")
    s = FakeSession({"rest.uniprot.org": lambda u, p: json.dumps({"results": [wrong, right]})})
    assert fetch_protein("CDH1", s).accession == "P12830"


def test_fetch_protein_raises_when_missing():
    s = FakeSession({"rest.uniprot.org": lambda u, p: json.dumps({"results": []})})
    with pytest.raises(LookupError):
        fetch_protein("NOTAGENE", s)


def test_parse_ca_coords():
    xyz = parse_ca_coords(toy_pdb(4), 5)
    assert xyz.shape == (5, 3)
    assert xyz[1, 0] == pytest.approx(3.8)
    assert np.isnan(xyz[4]).all()  # residue 5 absent


def test_fetch_structure_and_contacts(fake_session):
    s = fetch_structure("P00001", fake_session)
    assert s is not None and s.sequence == TOY_SEQ
    assert s.plddt_at(1) == 50.0 and s.plddt_at(3) == 58.0
    counts = s.contact_counts(radius=10.0)
    # On a straight line with 3.8 A spacing, neighbours within 10 A are +-1 and +-2.
    assert counts[0] == 2 and counts[5] == 4


def test_fetch_structure_none_when_absent():
    s = FakeSession({"/api/prediction/": lambda u, p: None})
    assert fetch_structure("P99999", s) is None


def test_alphamissense(fake_session):
    am = fetch_alphamissense("P00001", fake_session)
    assert am == {"A4T": 0.91, "K2R": 0.05}
