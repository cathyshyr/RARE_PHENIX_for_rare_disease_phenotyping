# Scripts

This folder contains RARE-PHENIX scripts.

## Researcher-facing quickstart scripts

| Script | Purpose |
|---|---|
| `run_module1_extraction.py` | Runs Module 1 phenotype extraction using the public Hugging Face LoRA adapters. |
| `run_module2_hpo_standardization.py` | Runs lightweight Module 2 HPO standardization using lexical, semantic, or combined matching. |
| `smoke_test_module1_hf.py` | Checks that the Module 1 Hugging Face adapter can be loaded and used. Requires access to the gated Llama-2 base model. |
| `smoke_test_module2_lightweight.py` | Fast local smoke test for lightweight Module 2. Does not require Llama-2 access. |
| `smoke_test_quickstart.py` | Runs both Module 1 and Module 2 smoke tests. Requires access to the gated Llama-2 base model. |

## Module 1

Run phenotype extraction on a CSV file:

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv \
  --module2-output outputs/module1_for_module2.csv
~~~

## Lightweight Module 2

Run HPO standardization on Module 1 output:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --id-column UID \
  --phenotype-column Step1_Clean_Split \
  --hpo-terms data/HPO_ID_TERM_DEFN.xlsx \
  --output outputs/module2_hpo_candidates.csv \
  --top-k 5
~~~

The default method is `combined`, which combines lexical fuzzy matching with semantic retrieval.

Other available methods:

~~~bash
--method lexical
--method semantic
--method combined
~~~

## Notes

The lightweight Module 2 script is intended to be easy to run locally. It is not a full replacement for the original RAG-based Module 2 workflow used in the RARE-PHENIX research pipeline.
