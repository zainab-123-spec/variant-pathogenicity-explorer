import pytest

from vpe.aminoacids import AMINO_ACIDS, BLOSUM62, KYTE_DOOLITTLE, VOLUME, blosum62, to_one_letter


def test_blosum62_is_symmetric_and_complete():
    assert len(BLOSUM62) == 400
    for a in AMINO_ACIDS:
        for b in AMINO_ACIDS:
            assert BLOSUM62[(a, b)] == BLOSUM62[(b, a)]


def test_blosum62_known_values():
    assert blosum62("W", "W") == 11
    assert blosum62("C", "C") == 9
    assert blosum62("I", "V") == 3  # similar hydrophobic residues: common swap
    assert blosum62("w", "g") == -2  # case-insensitive


def test_property_tables_cover_all_amino_acids():
    assert set(KYTE_DOOLITTLE) == set(VOLUME) == set(AMINO_ACIDS)


def test_three_letter_conversion():
    assert to_one_letter("Arg") == "R"
    assert to_one_letter("TRP") == "W"
    with pytest.raises(KeyError):
        to_one_letter("Ter")
