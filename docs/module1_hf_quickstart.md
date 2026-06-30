# RARE-PHENIX Module 1 Hugging Face Quickstart

This guide shows how to run the public RARE-PHENIX Module 1 phenotype extraction adapter from Hugging Face.

## What this runs

This runs **Module 1 only** of RARE-PHENIX:

1. Input: clinical note text
2. Output: rare disease phenotype mentions extracted from the note

This does **not** run the full end-to-end RARE-PHENIX pipeline. Modules 2 and 3 are responsible for HPO standardization and HPO prioritization.

## Public adapter

The default adapter is:

~~~text
shyrcathy/rare-phenix-llama2-7b-raredis
~~~

This is a PEFT/LoRA adapter trained on the public RareDis corpus. It is not the full UDN-trained RARE-PHENIX model described in the paper.

Users must separately have access to the corresponding Meta Llama-2 base model:

~~~text
meta-llama/Llama-2-7b-chat-hf
~~~

## Install

Create and activate a Python environment:

~~~bash
python3 -m venv rare-phenix-env
source rare-phenix-env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_module1_hf.txt
~~~

## Hugging Face login

Log in to Hugging Face:

~~~bash
hf auth login
~~~

Your Hugging Face account must have access to the gated Meta Llama-2 model.

## Run a smoke test

~~~bash
python scripts/smoke_test_module1_hf.py
~~~

## Run Module 1 on the sample CSV

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv
~~~

## Output columns

The output CSV contains:

| Column | Description |
|---|---|
| `patient_id` | Patient or note identifier from the input CSV |
| `cleaned_annotated_note` | Note with extracted phenotype mentions wrapped in `<span class="condition">...</span>` |
| `extracted_phenotypes` | Pipe-delimited extracted phenotype mentions |

Example:

~~~text
global developmental delay|hypotonia|microcephaly|feeding difficulties|short stature
~~~

## Optional debugging output

To include the raw model completion before cleanup:

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted_with_raw.csv \
  --include-raw-output
~~~

## Clinical and privacy note

Use only synthetic or properly deidentified text unless running in an environment approved for protected health information. Outputs should be reviewed by domain experts and should not be used as autonomous clinical diagnoses.
