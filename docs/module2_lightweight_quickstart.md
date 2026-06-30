# RARE-PHENIX Module 2 Lightweight HPO Standardization Quickstart

This guide shows how to run a lightweight, researcher-friendly Module 2 HPO standardization workflow.

## What this runs

This script takes extracted phenotype strings from Module 1 and returns candidate Human Phenotype Ontology (HPO) terms.

Input:

~~~text
UID,Step1_Clean_Split
P001,global developmental delay
P001,hypotonia
P001,microcephaly
~~~

Output:

~~~text
UID,Step1_Clean_Split,method,rank,match_score,hpo_id,hpo_label
P001,global developmental delay,combined,1,116.888,HP:0001263,Global developmental delay
~~~

## Important scope note

This is a lightweight HPO standardization script.

It is intended to be easier for researchers to run than the original RAG-based Module 2 script.

It supports three methods:

| Method | Description |
|---|---|
| `combined` | Default. Combines lexical fuzzy matching with semantic retrieval. |
| `lexical` | Fast fuzzy matching against HPO labels. |
| `semantic` | Embedding retrieval over HPO labels and definitions. |

The original RARE-PHENIX Module 2 used retrieval-augmented generation. This lightweight script is a practical local alternative and does not require running a large LLM.

## Install

Install the researcher dependencies:

~~~bash
python -m pip install -r requirements_module1_hf.txt
~~~

## Run Module 1 first

Generate the Module 2-compatible input file:

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv \
  --module2-output outputs/module1_for_module2.csv
~~~

## Run lightweight Module 2

Default combined mode:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --id-column UID \
  --phenotype-column Step1_Clean_Split \
  --hpo-terms data/HPO_ID_TERM_DEFN.xlsx \
  --output outputs/module2_hpo_candidates.csv \
  --top-k 5
~~~

Lexical-only mode:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --output outputs/module2_hpo_candidates_lexical.csv \
  --method lexical \
  --top-k 5
~~~

Semantic-only mode:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --output outputs/module2_hpo_candidates_semantic.csv \
  --method semantic \
  --top-k 5
~~~

## Output columns

| Column | Description |
|---|---|
| `UID` | Patient or note identifier |
| `Step1_Clean_Split` | Extracted phenotype string |
| `method` | Matching method used |
| `rank` | Candidate rank for that phenotype |
| `match_score` | Matching score |
| `hpo_id` | Candidate HPO identifier |
| `hpo_label` | Candidate HPO label |
| `hpo_definition` | Candidate HPO definition |

## Notes

By default, obsolete HPO terms are excluded.

To include obsolete HPO terms:

~~~bash
--include-obsolete
~~~

The combined method prioritizes exact HPO label matches when present, while still using semantic retrieval to surface related candidates.
