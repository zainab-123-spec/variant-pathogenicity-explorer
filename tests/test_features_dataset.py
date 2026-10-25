import math

from conftest import uniprot_entry

from vpe.clinvar import ClinVarVariant
from vpe.dataset import build_gene, build_panel
from vpe.features import FEATURE_COLUMNS, substitution_features, variant_features
from vpe.uniprot import parse_entry


def test_substitution_features():
    f = substitution_features("R", "W")  # Arg -> Trp
    assert f["blosum62"] == -3
    assert f["charge_change"] == 1.0
    assert f["hydrophobicity_delta"] == -0.9 - (-4.5)
    assert f["involves_gly_or_pro"] == 0.0
    assert substitution_features("G", "A")["involves_gly_or_pro"] == 1.0


def test_reference_mismatch_is_rejected():
    p = parse_entry("TOY", uniprot_entry())
    bad = ClinVarVariant("1", "TOY", "", "G", 2, "A", "Benign", "", 0)  # position 2 is K
    assert variant_features(bad, p, None) is None
    out_of_range = ClinVarVariant("1", "TOY", "", "G", 99, "A", "Benign", "", 0)
    assert variant_features(out_of_range, p, None) is None


def test_missing_structure_gives_nan():
    p = parse_entry("TOY", uniprot_entry())
    v = ClinVarVariant("1", "TOY", "", "K", 2, "R", "Benign", "", 0)
    row = variant_features(v, p, None)
    assert row is not None and math.isnan(row["plddt"])
    assert set(FEATURE_COLUMNS) <= set(row)


def test_build_gene_end_to_end(fake_session):
    gd = build_gene("TOY", fake_session)
    t = gd.table.set_index("protein_change")
    assert set(t.index) == {"K2R", "A4T", "C11Y"}
    assert gd.n_ref_mismatch == 1  # G12D: residue 12 is L
    assert t.loc["A4T", "in_domain"] == 1.0 and t.loc["K2R", "in_domain"] == 0.0
    assert t.loc["A4T", "near_functional_site"] == 1.0
    assert t.loc["C11Y", "in_disordered_region"] == 1.0
    assert t.loc["A4T", "plddt"] == 62.0
    assert t.loc["A4T", "alphamissense"] == 0.91
    assert math.isnan(t.loc["C11Y", "alphamissense"])


def test_build_panel_reports_skipped_genes(fake_session):
    import json

    fake_session.routes["rest.uniprot.org"] = lambda u, p: (
        json.dumps({"results": []}) if "NOPE" in p["query"] else json.dumps({"results": [uniprot_entry()]})
    )
    data, qc = build_panel(["TOY", "NOPE"], fake_session)
    assert len(data) == 3
    assert qc.set_index("gene").loc["NOPE", "status"].startswith("skipped")
