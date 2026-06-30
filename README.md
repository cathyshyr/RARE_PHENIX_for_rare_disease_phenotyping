# RARE-PHENIX for Rare Disease Phenotyping

## Update: May 18, 2026

Because the Undiagnosed Diseases Network (UDN)-derived data used in this study contain sensitive patient information and are subject to controlled-access data sharing policies, we are unable to publicly release the RARE-PHENIX checkpoints fine-tuned on UDN data.

To support reproducibility, we now provide the final PEFT/LoRA adapter checkpoints for the **Llama-2 family of RARE-PHENIX models** fine-tuned **only on the public RareDis corpus**. These checkpoints are available under `/model_checkpoint_RareDis/`.

We release the `Llama-2-70b-chat-hf` RareDis-only checkpoint because this model achieved the best performance in the study based on ontology-based similarity, which was the primary outcome. We also release smaller `Llama-2-13b-chat-hf` and `Llama-2-7b-chat-hf` variants to support users with more limited computational resources and to enable comparison across model versions.

Please note that these are adapter checkpoints, not standalone full model weights. To use them, users must obtain access to the corresponding Llama-2 chat base model separately.

**Please Note:** The Undiagnosed Diseases Network (UDN) data used in this study contain sensitive patient information. De-identified patient data, including phenotypic and genomic data, are deposited in the [database of Genotypes and Phenotypes (dbGaP)](https://www.ncbi.nlm.nih.gov/gap/) maintained by the National Institutes of Health. To explore data available in the latest release, visit the [UDN study page in dbGaP](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001232.v7.p3). Individuals interested in accessing UDN data through dbGaP should submit a data access request. Detailed instructions for this process can be found on the NIH Scientific Data Sharing website: [How to Request and Access Datasets from dbGaP](https://sharing.nih.gov/accessing-data/accessing-genomic-data/how-to-request-and-access-datasets-from-dbgap).

# Expected Data Format
- `UID` – patient identifier  
- `note_date` - date of the clinical note 
- `note_title` - title of the clinical note
- `note_text` - content of the clinical note

---

# RARE-PHENIX Pipeline Structure

## Quickstart: public Module 1 Hugging Face adapters

A public, runnable path is available for **RARE-PHENIX Module 1 phenotype extraction** using Hugging Face PEFT/LoRA adapters.

Start here:

~~~bash
python3 -m venv rare-phenix-env
source rare-phenix-env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_module1_hf.txt

# Or install dependencies for both Module 1 and lightweight Module 2:
python -m pip install -r requirements_researcher_quickstart.txt
hf auth login
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv
~~~

See the full quickstart:

~~~text
docs/module1_hf_quickstart.md
~~~

Available public RareDis adapters:

| Adapter | Base model |
|---|---|
| [`shyrcathy/rare-phenix-llama2-7b-raredis`](https://huggingface.co/shyrcathy/rare-phenix-llama2-7b-raredis) | `meta-llama/Llama-2-7b-chat-hf` |
| [`shyrcathy/rare-phenix-llama2-13b-raredis`](https://huggingface.co/shyrcathy/rare-phenix-llama2-13b-raredis) | `meta-llama/Llama-2-13b-chat-hf` |
| [`shyrcathy/rare-phenix-llama2-70b-raredis`](https://huggingface.co/shyrcathy/rare-phenix-llama2-70b-raredis) | `meta-llama/Llama-2-70b-chat-hf` |

These adapters are trained on the public RareDis corpus and are **not** the full Undiagnosed Diseases Network (UDN)-trained RARE-PHENIX model described in the manuscript. Users must separately have access to the corresponding gated Meta Llama-2 base model through Hugging Face.

This quickstart runs **Module 1 only**. Modules 2 and 3 are responsible for HPO standardization and HPO prioritization.

## Quickstart: lightweight Module 2 HPO standardization

A lightweight, researcher-friendly Module 2 script is also available. It standardizes extracted phenotype strings to candidate Human Phenotype Ontology (HPO) terms using lexical and semantic retrieval.

Start after generating the Module 2-compatible output from Module 1:

~~~bash
python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --id-column UID \
  --phenotype-column Step1_Clean_Split \
  --hpo-terms data/HPO_ID_TERM_DEFN.xlsx \
  --output outputs/module2_hpo_candidates.csv \
  --top-k 5
~~~

See the full quickstart:

~~~text
docs/module2_lightweight_quickstart.md
~~~

This lightweight script is intended to be easier for researchers to run than the original RAG-based Module 2 script. It does not require running a large LLM.

## Minimal Module 1 → Module 2 example

Run phenotype extraction and HPO standardization on the included sample notes:

~~~bash
python scripts/run_module1_extraction.py \
  --input examples/sample_notes.csv \
  --id-column patient_id \
  --text-column note_text \
  --output outputs/module1_extracted.csv \
  --module2-output outputs/module1_for_module2.csv

python scripts/run_module2_hpo_standardization.py \
  --input outputs/module1_for_module2.csv \
  --id-column UID \
  --phenotype-column Step1_Clean_Split \
  --hpo-terms data/HPO_ID_TERM_DEFN.xlsx \
  --output outputs/module2_hpo_candidates.csv \
  --top-k 5
~~~

The first command extracts phenotype mentions from notes. The second command maps extracted phenotype strings to candidate HPO terms.

## Quickstart: lightweight Module 3 HPO prioritization

A lightweight Module 3 script is available to aggregate Module 2 candidates into patient-level prioritized HPO lists.

~~~bash
python scripts/run_module3_hpo_prioritization.py \
  --input outputs/module2_hpo_candidates.csv \
  --output outputs/module3_prioritized_hpos.csv \
  --max-candidate-rank 5 \
  --top-n 30
~~~

See the full quickstart:

~~~text
docs/module3_lightweight_quickstart.md
~~~

This lightweight script is intended to make the public repository runnable end-to-end. It is not the original supervised Module 3 learning-to-rank training pipeline.

## Optional validation

Run the lightweight Module 2 smoke test:

~~~bash
python scripts/smoke_test_module2_lightweight.py
~~~

Run the combined quickstart smoke test:

~~~bash
python scripts/smoke_test_quickstart.py
~~~

The combined smoke test runs both the Module 1 Hugging Face adapter test and the lightweight Module 2 test. It requires access to the gated Llama-2 base model on Hugging Face.

---


## Citation

If you use RARE-PHENIX or the public Module 1 adapters, please cite:

Shyr, C., Hu, Y., Tinker, R.J., Cassini, T.A., Byram, K.W., Hamid, R., Fabbri, D.V., Wright, A., Peterson, J.F., Bastarache, L., and Xu, H. 2026. *An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models*. arXiv preprint arXiv:2602.20324.

---

## Preprocessing & Training

### `Preprocess_Clinical_Notes.py`

Prepares clinical notes for downstream processing.

**Main steps:**
- Load the raw notes file  
- Filter notes by title and content  
- Remove unwanted sections (e.g., signatures, administrative text)  
- Concatenate notes by patient identifier (`UID`)  
- Split long documents into smaller chunks  
- Create multiple subsets for parallel processing  

**Output:**
- Chunked and filtered note files ready for model input  

---

### `Instruction_Fine_Tuning.py`

Performs instruction tuning of a pre-trained LLM.

**Main steps:**
- Load the pre-trained LLM
- Load training and evaluation datasets  
- Format each example into an instruction/chat template  
- Apply LoRA for parameter-efficient fine-tuning  
- Run training with the Hugging Face `Trainer`  
- Save the fine-tuned model  

**Output:**
- Fine-tuned LoRA checkpoint  

---

## RARE-PHENIX Inference Pipeline

The framework is divided into three modules.  
Each module includes a main script and a post-processing script.

---

### Module 1

#### `RARE_PHENIX_Module1.py`

Phenotype Extraction from Clinical Notes

**Typical responsibilities:**
- Load the fine-tuned model  
- Generate predictions from note chunks  
- Write raw model outputs  

#### `RARE_PHENIX_Module1_Postprocess.py`

Cleans and structures the raw outputs from Module 1.

**Typical responsibilities:**
- Parse model generations  
- Remove formatting artifacts (e.g., HTML span tags)
- Convert outputs into a structured format for the next module  

---

### Module 2

#### `RARE_PHENIX_Module2.py`

Standardization to Human Phenotype Ontology (HPO) Terms

**Typical responsibilities:**
- Load results from Module 1
- Perform entity linking of free-text strings to structured HPO terms using retrieval-augmented generation

#### `RARE_PHENIX_Module2_Postprocess.py`

Post-processes Module 2 outputs into a clean structured format for Module 3.

---

### Module 3

#### `RARE_PHENIX_Module3_Preprocess.py`

**Typical responsibilities:**
- Create negative HPO sets for training the learning-to-rank model
- Prepares data matrix for model training and validation

#### `RARE_PHENIX_Module3.py`

Prioritization of Diagnostically Informative Phenotypes

**Typical responsibilities:**
- Trains, validates, and selects the best-performing learning-to-rank model based on mean average precision
- Candidate models include: XGBoost, CatBoost, LightGBM, and logistic regression


---

