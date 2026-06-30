# RARE-PHENIX for Rare Disease Phenotyping

RARE-PHENIX is a modular pipeline for rare disease phenotyping from clinical notes.

The public version of this repository provides a runnable researcher-facing workflow:

1. **Module 1**: extract phenotype mentions from clinical notes using public RareDis-only Hugging Face adapters.
2. **Module 2**: standardize extracted phenotype strings to candidate Human Phenotype Ontology (HPO) terms using a lightweight local method.
3. **Module 3**: aggregate Module 2 candidates into patient-level prioritized HPO lists using a lightweight public baseline.

## Important release notes

The full RARE-PHENIX study used Undiagnosed Diseases Network (UDN)-derived data. These data contain sensitive patient information and are subject to controlled-access data sharing policies. For that reason, this public repository does **not** release UDN-trained checkpoints, UDN patient-level data, or the original supervised Module 3 trained rankers.

To support reproducibility and public use, we release **RareDis-only Module 1 PEFT/LoRA adapters** on Hugging Face. These adapters were fine-tuned only on the public RareDis corpus. These adapters were fine-tuned only on the publicly released RareDis corpus. RareDis is a biomedical NLP corpus annotated for rare diseases, clinical manifestations such as signs and symptoms, and relations among them. 

Reference: Martínez-de Miguel, C., Segura-Bedmar, I., Chacón-Solano, E., & Guerrero-Aspizua, S. (2022). The RareDis corpus: A corpus annotated with rare diseases, their signs and symptoms. Journal of Biomedical Informatics, 125, 103961. DOI: 10.1016/j.jbi.2021.103961.

The public Module 2 and Module 3 scripts are lightweight researcher-facing alternatives intended to make the repository runnable end-to-end. They are not the original RAG-based Module 2 and supervised learning-to-rank Module 3 workflows propsed in the manuscript.

## Public Hugging Face adapters for Module 1

The Module 1 adapters are hosted on Hugging Face, not in this GitHub repository.

| Adapter | Required base model |
|---|---|
| [`shyrcathy/rare-phenix-llama2-7b-raredis`](https://huggingface.co/shyrcathy/rare-phenix-llama2-7b-raredis) | `meta-llama/Llama-2-7b-chat-hf` |
| [`shyrcathy/rare-phenix-llama2-13b-raredis`](https://huggingface.co/shyrcathy/rare-phenix-llama2-13b-raredis) | `meta-llama/Llama-2-13b-chat-hf` |
| [`shyrcathy/rare-phenix-llama2-70b-raredis`](https://huggingface.co/shyrcathy/rare-phenix-llama2-70b-raredis) | `meta-llama/Llama-2-70b-chat-hf` |

These are adapter checkpoints, not standalone full model weights. Users must separately have access to the corresponding gated Meta Llama-2 chat base model through Hugging Face.

## Recommended public quickstart

Create an environment and install dependencies for the public workflow:

~~~bash
python3 -m venv rare-phenix-env
source rare-phenix-env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_researcher_quickstart.txt
hf auth login
~~~

Run Module 1 phenotype extraction on the sample notes:

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv \
  --module2-output outputs/module1_for_module2.csv
~~~

Run lightweight Module 2 HPO standardization:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --id-column UID \
  --phenotype-column Step1_Clean_Split \
  --hpo-terms data/HPO_ID_TERM_DEFN.xlsx \
  --output outputs/module2_hpo_candidates.csv \
  --top-k 5
~~~

Run lightweight Module 3 HPO prioritization:

~~~bash
python scripts/run_module3_hpo_prioritization.py \
  --input outputs/module2_hpo_candidates.csv \
  --output outputs/module3_prioritized_hpos.csv \
  --max-candidate-rank 5 \
  --top-n 30
~~~

The resulting file, `outputs/module3_prioritized_hpos.csv`, contains patient-level prioritized HPO terms.

## Module-specific quickstarts

More detailed instructions are available here:

| Module | Guide |
|---|---|
| Module 1 RareDis-only Hugging Face adapters | `docs/module1_hf_quickstart.md` |
| Module 2 lightweight HPO standardization | `docs/module2_lightweight_quickstart.md` |
| Module 3 lightweight HPO prioritization | `docs/module3_lightweight_quickstart.md` |

## Expected input data format

For the original clinical-note preprocessing workflow, input data generally use:

| Column | Description |
|---|---|
| `UID` | Patient identifier |
| `note_date` | Date of the clinical note |
| `note_title` | Title of the clinical note |
| `note_text` | Clinical note text |

The public quickstart uses `examples/sample_notes.csv`, which contains:

| Column | Description |
|---|---|
| `patient_id` | Example patient identifier |
| `note_text` | Example note text |

## Researcher-facing scripts

| Script | Purpose |
|---|---|
| `scripts/run_module1_extraction.py` | Runs Module 1 phenotype extraction using the public RareDis-only Hugging Face adapters. |
| `scripts/run_module2_hpo_standardization.py` | Runs lightweight Module 2 HPO standardization using lexical, semantic, or combined retrieval. |
| `scripts/run_module3_hpo_prioritization.py` | Runs lightweight Module 3 HPO prioritization by aggregating Module 2 candidates into patient-level ranked HPO lists. |

## Public workflow scope

### Module 1

The public Module 1 workflow uses RareDis-only LoRA adapters hosted on Hugging Face. These adapters are useful for phenotype extraction experiments, but they are **not** the full UDN-trained RARE-PHENIX checkpoints from the manuscript.

### Module 2

The public Module 2 script is a lightweight HPO standardization workflow. It supports:

| Method | Description |
|---|---|
| `combined` | Default. Combines lexical fuzzy matching and semantic retrieval. |
| `lexical` | Fast fuzzy matching against HPO labels. |
| `semantic` | Embedding retrieval over HPO labels and definitions. |

This public Module 2 script does not require running a large LLM. It is intended to be easier for researchers to run than the original RAG-based Module 2 workflow.

### Module 3

The public Module 3 script is a lightweight prioritization baseline. It aggregates Module 2 HPO candidates into patient-level ranked HPO lists using candidate scores, candidate ranks, and repeated evidence.

The original Module 3 learning-to-rank workflow used controlled-access patient-level data and trained supervised ranking models. Those trained models and the underlying UDN-derived training data are not publicly released.

## Optional validation

These commands are optional. They are mainly useful for checking that the public quickstart scripts are installed correctly.

Run the lightweight Module 2 smoke test:

~~~bash
python scripts/smoke_test_module2_lightweight.py
~~~

Run the lightweight Module 3 smoke test:

~~~bash
python scripts/smoke_test_module3_lightweight.py
~~~

Run the combined quickstart smoke test:

~~~bash
python scripts/smoke_test_quickstart.py
~~~

The combined smoke test runs both the Module 1 Hugging Face adapter test and the lightweight Module 2 test. It requires access to the gated Llama-2 base model on Hugging Face.

## Legacy research scripts

The following root-level scripts are retained for transparency and continuity with the original research workflow:

| Script | Purpose |
|---|---|
| `Preprocess_Clinical_Notes.py` | Prepares clinical notes for downstream processing. |
| `Instruction_Fine_Tuning.py` | Performs instruction tuning of a pre-trained LLM. |
| `RARE_PHENIX_Module1.py` | Original Module 1 inference script. |
| `RARE_PHENIX_Module1_Postprocess.py` | Original Module 1 post-processing script. |
| `RARE_PHENIX_Module2.py` | Original Module 2 RAG-based HPO standardization script. |
| `RARE_PHENIX_Module2_Postprocess.py` | Original Module 2 post-processing script. |
| `RARE_PHENIX_Module3_Preprocess.py` | Original Module 3 preprocessing script for training data construction. |
| `RARE_PHENIX_Module3.py` | Original Module 3 supervised learning-to-rank training script. |

For new users, the recommended entry point is the researcher-facing public workflow in the `scripts/`, `docs/`, and `examples/` folders.

## Data availability and privacy

The UDN-derived data used in the study contain sensitive patient information and are subject to controlled-access data sharing policies. De-identified UDN data, including phenotypic and genomic data, are deposited in the NIH database of Genotypes and Phenotypes (dbGaP). Researchers interested in UDN data access should use the appropriate dbGaP data access request process.

Do not upload protected health information or sensitive patient data to public services unless your use complies with all applicable institutional, legal, and data-use requirements.

## Citation

If you use RARE-PHENIX or the public Module 1 adapters, please cite:

~~~bibtex
@article{shyr2026artificial,
  title={An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models},
  author={Shyr, Cathy and Hu, Yan and Tinker, Rory J and Cassini, Thomas A and Byram, Kevin W and Hamid, Rizwan and Fabbri, Daniel V and Wright, Adam and Peterson, Josh F and Bastarache, Lisa and others},
  journal={arXiv preprint arXiv:2602.20324},
  year={2026}
}
~~~

## Disclaimer

RARE-PHENIX is intended for research use. It is not a medical device and should not be used as the sole basis for clinical diagnosis, treatment, or patient management decisions.
