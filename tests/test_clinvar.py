from conftest import clinvar_record

from vpe.clinvar import build_query, fetch_variants, label_from_classification, parse_summary


def test_labels():
    assert label_from_classification("Pathogenic") == 1
    assert label_from_classification("Likely pathogenic") == 1
    assert label_from_classification("Pathogenic/Likely pathogenic") == 1
    assert label_from_classification("Benign/Likely benign") == 0
    assert label_from_classification("Uncertain significance") is None
    assert label_from_classification("Conflicting classifications of pathogenicity") is None


def test_query_mentions_gene_and_missense():
    q = build_query("TP53")
    assert q.startswith("TP53[gene]") and "missense variant" in q


def test_parse_valid_missense():
    v = parse_summary(
        "TP53", clinvar_record("12347", "NM_000546.6(TP53):c.742C>T (p.Arg248Trp)", "Pathogenic", gene="TP53")
    )
    assert v is not None
    assert (v.ref, v.pos, v.alt, v.label) == ("R", 248, "W", 1)
    assert v.protein_change == "R248W"


def test_parse_skips_bad_records():
    title = "NM_1(TOY):c.1A>G (p.Lys2Arg)"
    assert parse_summary("TOY", clinvar_record("1", title, "Uncertain significance")) is None
    assert parse_summary("TOY", clinvar_record("1", title, "Benign", review="no assertion criteria provided")) is None
    assert parse_summary("TOY", clinvar_record("1", title, "Benign", obj_type="Haplotype")) is None
    assert parse_summary("TOY", clinvar_record("1", "NM_1(TOY):c.1A>T (p.Lys2Ter)", "Pathogenic")) is None
    assert parse_summary("TOY", clinvar_record("1", "NM_1(TOY):c.1A>G (p.Lys2=)", "Benign")) is None
    # record lists TOY but the title describes another gene's transcript
    assert parse_summary("TOY", clinvar_record("1", "NM_9(OTHER):c.1A>G (p.Lys2Arg)", "Benign")) is None
    # low-review records are kept if the filter is switched off
    assert (
        parse_summary(
            "TOY", clinvar_record("1", title, "Benign", review="no assertion criteria provided"), min_review=False
        )
        is not None
    )


def test_fetch_variants_drops_conflicts_and_vus(fake_session):
    variants = fetch_variants("TOY", fake_session)
    changes = {v.protein_change: v.label for v in variants}
    assert "R10W" not in changes  # LP and LB for the same protein change -> dropped
    assert "I6S" not in changes  # VUS
    assert changes["K2R"] == 0 and changes["A4T"] == 1
    assert [v.pos for v in variants] == sorted(v.pos for v in variants)
