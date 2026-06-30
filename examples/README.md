# Examples

This folder contains small example inputs for testing the researcher-facing RARE-PHENIX quickstart scripts.

## Files

| File | Purpose |
|---|---|
| `sample_notes.csv` | Example clinical-note input for Module 1 phenotype extraction. |
| `sample_module1_for_module2.csv` | Example long-format phenotype input for lightweight Module 2 HPO standardization. |

## Example workflow

Run Module 1 on sample notes:

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv \
  --module2-output outputs/module1_for_module2.csv
~~~

Run lightweight Module 2 directly on the standalone sample input:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input examples/sample_module1_for_module2.csv \
  --id-column UID \
  --phenotype-column Step1_Clean_Split \
  --hpo-terms data/HPO_ID_TERM_DEFN.xlsx \
  --output outputs/sample_module2_hpo_candidates.csv \
  --top-k 5
~~~
