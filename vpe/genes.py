"""The training panel: well-studied disease genes with many ClinVar missense calls.

Chosen to span different biology (DNA repair, tumour suppressors, cardiac
ion channels / sarcomere, metabolism, connective tissue) so that a model
which works across held-out genes has learned something general. Very long
proteins (>2,700 aa) are avoided because AlphaFold DB splits them into
fragments.
"""

PANEL: list[str] = [
    # DNA repair / hereditary cancer
    "BRCA1",
    "PALB2",
    "CHEK2",
    "RAD51C",
    "RAD51D",
    "BRIP1",
    "BARD1",
    "MLH1",
    "MSH2",
    "MSH6",
    "PMS2",
    "MUTYH",
    # Tumour suppressors / oncogenes
    "TP53",
    "PTEN",
    "CDH1",
    "STK11",
    "VHL",
    "RET",
    "MEN1",
    "SMAD4",
    "RB1",
    "NF2",
    "TSC2",
    # Cardiac
    "MYH7",
    "MYBPC3",
    "TNNT2",
    "TNNI3",
    "KCNQ1",
    "KCNH2",
    "SCN5A",
    "LMNA",
    # Metabolic / other Mendelian
    "LDLR",
    "PAH",
    "GAA",
    "GLA",
    "CFTR",
    "ATP7B",
    "CBS",
    "G6PD",
    "OTC",
    "F9",
    "HBB",
]
