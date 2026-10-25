# Learning Guide

This guide explains the project in simple English. Read it slowly, one part at a time. Run the code while you read. At the end there are interview questions and short answers you can practise.

**Tip:** Do not memorise the answers. Understand them, then say them in your own words.

---

## Part 1. The biology in 5 minutes

* A **gene** is an instruction in DNA. The cell uses it to make a **protein**.
* A protein is a chain of **amino acids**. There are 20 kinds, and each has a one-letter code (A, R, N, D, ...).
* A **missense variant** changes one amino acid into another one. Example: `R175H` means "at position 175, Arginine (R) became Histidine (H)".
* Some variants cause disease (**pathogenic**). Some are harmless (**benign**). Many are still unknown (**VUS**, variant of uncertain significance).
* **Our question:** can we guess whether a variant is pathogenic by looking at *where* it is in the protein and *what kind* of change it is?

**Why structure helps:** a protein folds into a 3D shape. If a change happens in the tightly packed core, it often breaks the shape, and that is often pathogenic. If it happens in a floppy, unfolded tail, it is often harmless.

---

## Part 2. The three APIs

An **API** is a web address that gives data to a program instead of to a person.

| API | What we ask | What we get |
|---|---|---|
| **UniProt** | "Give me the reviewed human protein for gene TP53" | Sequence, domains, functional sites, disordered regions |
| **AlphaFold DB** | "Give me the predicted structure for P04637" | 3D coordinates and **pLDDT** (confidence from 0 to 100) for each residue |
| **ClinVar** (NCBI E-utilities) | "Give me missense variants in TP53 with a clear label" | Variant name, label (pathogenic or benign), review status |

**pLDDT** is AlphaFold's confidence. A high value (above 90) means a well-folded region. A low value (below 50) usually means a disordered, floppy region.

---

## Part 3. The code, module by module

All code is in the folder `vpe/` ("variant pathogenicity explorer").

### `http.py`: talking to the internet safely
* `CachedSession` is used by every API call.
* **Cache:** each answer is saved to a file in `data/cache/`. The second time, we read the file and do not call the internet. This is fast and kind to the servers.
* **Rate limit:** NCBI allows only about 3 requests per second without a key. We wait 0.34 seconds between calls.
* **Retry:** if the server says "too busy" (error 429) or has an error (5xx), we wait and try again.
* **404 is cached too:** if a protein has no AlphaFold model, we remember that, so we do not ask again.
* **Offline mode:** `--offline` uses only the cache. This is useful for testing.

### `aminoacids.py`: chemistry tables
* Converts three-letter codes to one letter (`Arg` → `R`).
* **BLOSUM62:** a score for how often one amino acid replaces another during evolution. A negative score means a "rare, unusual swap".
* **Hydrophobicity** (Kyte-Doolittle: does it like water or not?), **volume** (size) and **charge**.

### `uniprot.py`: the protein
* Searches UniProt for `gene_exact:TP53 AND organism_id:9606 AND reviewed:true`.
* **Important lesson:** the search can match a *synonym*. At first, "CDH1" returned the protein FZR1, because FZR1 has "CDH1" as another name. We fixed this: we now accept only an entry whose **primary** gene name is exactly the one we asked for.
* `ProteinRecord` stores the sequence and features and answers questions like "is position 175 in a domain?"

### `alphafold.py`: the structure
* Gets the pLDDT for each residue and the position (x, y, z) of each Cα atom from the PDB file.
* `contact_counts()` counts how many other residues are within 10 Å. **Many contacts = buried in the core.**
* Also downloads AlphaMissense scores, but only to **compare** with them, never as an input.
* Note: the download needs a `User-Agent` header, otherwise the server returns error 403.

### `clinvar.py`: the labels
* `esearch` gets the list of variant IDs. `esummary` gets the details in batches of 200.
* Label rule: "Pathogenic" or "Likely pathogenic" → 1. "Benign" or "Likely benign" → 0. Everything else is ignored.
* We drop variants with the review status "no assertion criteria provided" (lowest quality).
* We read the protein change from the title, for example `(p.Arg248Trp)`.
* If the same protein change has both labels, we drop it (conflict).

### `features.py`: turning a variant into numbers
A machine learning model needs numbers. For each variant we make 15 numbers:
* **Structure:** pLDDT, average pLDDT in a ±5 window, contacts within 10 Å.
* **Position:** relative position (0 = start, 1 = end), in a domain, in a disordered region, near a functional site.
* **Chemistry:** BLOSUM62, change in hydrophobicity, volume and charge, and whether Gly/Pro or Cys is involved (these amino acids are special for shape).

### `dataset.py` and `genes.py`: building the table
* `genes.py` lists 42 well-known disease genes (cancer, heart, metabolism).
* `build_gene()` calls the three APIs and builds one row per variant.
* **Quality control:** the old amino acid in ClinVar must be the same as the amino acid in the UniProt sequence at that position. If not, the variant is dropped. This usually happens because ClinVar uses a different **isoform** (another version of the protein). We dropped 299 of 8,483 variants this way.

### `model.py`: machine learning
* **Logistic regression:** a simple, linear model. We fill missing values with the median, scale features, and use `class_weight="balanced"`.
* **Gradient boosting** (`HistGradientBoostingClassifier`): many small decision trees. It can learn non-linear patterns and handles missing values. We keep the trees small so it does not memorise.
* **GroupKFold by gene:** see Part 4. This is the most important idea.
* **Permutation importance:** shuffle one feature in the test genes and see how much the score falls. A big fall means an important feature.

### `report.py` and `cli.py`: results for people
* `report.py` draws the figures and writes an HTML report.
* `cli.py` gives the commands `fetch`, `build-dataset`, `train` and `analyze`.
* `analyze BRCA1` makes a picture of the whole protein with all its variants.
* For a panel gene, `analyze` uses **out-of-fold** scores (from the model that did not see this gene), so the picture is honest.

### `app.py`: the Streamlit app
* A small web page. Choose a gene, and it shows the track picture, metrics and a table.

### `tests/`: checking the code
* 32 tests. We use a **fake session** that returns saved toy answers, so the tests never touch the internet. They are fast and always give the same result.
* GitHub Actions runs them, plus `ruff` lint, on every push.

---

## Part 4. The most important idea: gene-held-out evaluation

**Random split:** we mix all variants and put 80% in training and 20% in test. Variants from the *same* gene are on both sides. The model can learn "this gene is mostly pathogenic". That is **leakage**, which means information from the test set leaks into training.

**Gene-held-out split (GroupKFold):** all variants of a gene are either in training or in test, never both. The test genes are new to the model. This is how the model would be used in real life.

**Our real result:**
* Random split, gradient boosting: ROC-AUC **0.920**
* Gene-held-out, gradient boosting: ROC-AUC **0.859**
* The difference (0.06) is leakage. We report 0.859.

---

## Part 5. The metrics

* **ROC-AUC:** pick one random pathogenic variant and one random benign variant. ROC-AUC is the chance the model gives the pathogenic one a higher score. 0.5 = guessing, 1.0 = perfect.
* **PR-AUC (average precision):** focuses on the positive class. **Careful:** a model that guesses randomly already gets about 0.74 here, because 74% of our variants are pathogenic. So PR-AUC 0.94 is good, but not as amazing as it sounds.

**Our numbers (gene-held-out):**

| | ROC-AUC | PR-AUC |
|---|---|---|
| BLOSUM62 only | 0.697 | 0.840 |
| pLDDT only | 0.798 | 0.911 |
| Logistic regression | 0.847 | 0.931 |
| Gradient boosting | 0.859 | 0.939 |
| AlphaMissense (reference) | 0.931 | 0.972 |

---

## Part 6. Things to try yourself (make the project yours)

1. Run everything from the start: `build-dataset`, `train`, `analyze TP53`. Check that your numbers match.
2. Add a gene to `vpe/genes.py`, then retrain. Does the score change?
3. **Add conservation** (how much a position changes across species). This is the strongest signal we are missing.
4. **Fix isoforms:** map ClinVar positions to the canonical UniProt isoform, so MUTYH variants are not dropped.
5. Try another model, for example a random forest, and compare it under GroupKFold.

---

## Part 7. Interview questions and answers

**1. What does your project do, in one sentence?**
It takes data from UniProt, AlphaFold and ClinVar, makes 15 structural and chemical features for each missense variant, and trains a model that predicts pathogenicity on genes it has never seen, with ROC-AUC 0.859.

**2. Why did you use gene-held-out cross-validation?**
Variants in the same gene are similar. With a random split the model can remember gene-specific patterns. In my data, the random split gave 0.920 and the gene-held-out split gave 0.859. The real use case is a new gene, so I report the gene-held-out number.

**3. Which features were most important?**
The number of nearby residues within 10 Å (being buried), pLDDT (being in a folded region) and BLOSUM62 (an unusual swap). I measured this with permutation importance on held-out genes.

**4. Why is the logistic regression coefficient for pLDDT negative, when pLDDT is important?**
pLDDT and the pLDDT window are highly correlated. The model gives the window a large positive weight, and the single-residue value then corrects it a little in the other direction. So single coefficients are not safe to read when features are correlated. Permutation importance or removing one group of features is safer.

**5. How did you handle the API rate limits?**
One cached HTTP session. It waits 0.34 seconds between NCBI calls (0.11 s with an API key), retries on 429 and 5xx with back-off, and saves every answer to disk. After the first run, the whole dataset builds from cache in about 25 seconds.

**6. Tell me about a bug you found.**
My quality check compares the ClinVar amino acid with the UniProt sequence. For CDH1, almost every variant failed. I found that the UniProt search had returned FZR1, a different protein that has "CDH1" as a synonym. I fixed it by matching the primary gene name exactly. The lesson: write checks that fail loudly when the data is wrong.

**7. How do you test code that calls web APIs?**
I made a fake session class that returns saved JSON and a small toy protein. The tests run with no internet, in about 25 seconds, and they are the same every time. CI runs them on Python 3.10 and 3.12.

**8. Your PR-AUC is 0.94. Is that very good?**
It is good, but the baseline is high. 74% of the variants are pathogenic, so random guessing gives about 0.74. I always compare with the base rate and with simple baselines like BLOSUM62 (0.840).

**9. How do you compare with AlphaMissense?**
AlphaMissense gets 0.931 ROC-AUC on the same variants. I use it as a reference, not as a competitor. It uses evolutionary information from many species and a much bigger model. My goal was an interpretable baseline and a clean evaluation, and it shows clearly how far simple structural features can go.

**10. What are the limitations?**
ClinVar labels are biased towards well-studied genes. 3.5% of variants were dropped because of isoform numbering. Very long proteins have no AlphaFold model. There is no conservation feature. And it cannot be checked on VUS, which are the real problem cases.

**11. Why gradient boosting and not deep learning?**
I have about 8,000 rows and 15 features. For tabular data of this size, boosted trees are strong, fast (training takes seconds) and easy to explain. Deep learning would need much more data or pre-trained protein language models.

**12. How do you deal with class imbalance?**
I used `class_weight="balanced"` in both models, and I report metrics that do not depend on a threshold (ROC-AUC and PR-AUC). Some genes are almost all one class, so per-gene ROC-AUC is reported only for genes with at least 5 variants of each class (33 of 42 genes).

**13. Does the model work on genes outside the training set?**
Yes, partly. On SCN1A (1,052 variants, not in training) ROC-AUC was 0.905. On long proteins with no AlphaFold model, like ATM (0.743) and APC (0.635), it was weaker, because the structure features were missing.

**14. How would you scale this to all human genes?**
Use bulk downloads instead of one request per gene (the ClinVar VCF, AlphaFold proteome files, the UniProt FTP). Run the per-gene work in parallel. Store the features in a columnar format like Parquet, and keep the same gene-held-out evaluation.

**15. What would you do next?**
Add conservation from multiple sequence alignments, map isoforms to recover dropped variants, add protein–protein interface features, and calibrate the probabilities so a score of 0.8 really means about 80%.

---

## Small glossary

| Word | Simple meaning |
|---|---|
| Residue | One amino acid inside a protein |
| Canonical sequence | The main, reference version of the protein in UniProt |
| Isoform | Another version of the same protein (for example, longer or shorter) |
| Å (angstrom) | 0.1 nanometre. Atoms are about 1-2 Å apart |
| Out-of-fold | A prediction made by a model that did not train on that row |
| Leakage | Test information getting into training by mistake |
| Imputation | Filling in missing values (we use the median) |
