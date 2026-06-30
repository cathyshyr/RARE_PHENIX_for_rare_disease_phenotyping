# RARE-PHENIX Module 3 Lightweight HPO Prioritization Quickstart

This guide shows how to run a lightweight, researcher-friendly Module 3 workflow.

## What this runs

The script takes Module 2 HPO candidates and produces a patient-level prioritized HPO list.

Input:

~~~text
UID,Step1_Clean_Split,method,rank,match_score,hpo_id,hpo_label,hpo_definition
P001,microcephaly,combined,1,116.264,HP:0000252,Microcephaly,
~~~

Output:

~~~text
UID,priority_rank,ranking_score,hpo_id,hpo_label,source_phenotypes
P001,1,129.037,HP:0000252,Microcephaly,microcephaly
~~~

## Scope note

This is a lightweight public baseline.

It is not the original supervised Module 3 ranker training pipeline. The original Module 3 script trains learning-to-rank models using curated positives, generated negatives, and additional ontology/knowledgebase features.

This lightweight script is intended to make the public repository runnable end-to-end.

## Try Module 3 directly

A standalone sample Module 3 input file is included:

~~~text
examples/sample_module2_for_module3.csv
~~~

Run:

~~~bash
python scripts/run_module3_hpo_prioritization.py \
  --input examples/sample_module2_for_module3.csv \
  --output outputs/sample_module3_prioritized_hpos.csv \
  --max-candidate-rank 3 \
  --top-n 10
~~~

## Run after Module 2

~~~bash
python scripts/run_module3_hpo_prioritization.py \
  --input outputs/module2_hpo_candidates.csv \
  --output outputs/module3_prioritized_hpos.csv \
  --max-candidate-rank 5 \
  --top-n 30
~~~

## Output columns

| Column | Description |
|---|---|
| `UID` | Patient or note identifier |
| `priority_rank` | Patient-level HPO priority rank |
| `ranking_score` | Lightweight prioritization score |
| `hpo_id` | Candidate HPO identifier |
| `hpo_label` | Candidate HPO label |
| `best_candidate_rank` | Best Module 2 rank for this HPO term |
| `best_match_score` | Best Module 2 match score for this HPO term |
| `mean_match_score` | Mean Module 2 match score |
| `mention_count` | Number of candidate rows supporting this HPO term |
| `phenotype_count` | Number of unique extracted phenotypes supporting this HPO term |
| `source_phenotypes` | Extracted phenotype strings supporting the HPO term |
| `hpo_definition` | HPO definition |
