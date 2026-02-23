# RARE-PHENIX for Rare Disease Phenotyping

**Please Note:** The Undiagnosed Diseases Network (UDN) data used in this study contain sensitive patient information. De-identified patient data, including phenotypic and genomic data, are deposited in the [database of Genotypes and Phenotypes (dbGaP)](https://www.ncbi.nlm.nih.gov/gap/) maintained by the National Institutes of Health. To explore data available in the latest release, visit the [UDN study page in dbGaP](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001232.v7.p3). Individuals interested in accessing UDN data through dbGaP should submit a data access request. Detailed instructions for this process can be found on the NIH Scientific Data Sharing website: [How to Request and Access Datasets from dbGaP](https://sharing.nih.gov/accessing-data/accessing-genomic-data/how-to-request-and-access-datasets-from-dbgap).

# Expected Data Format
- `UID` – patient identifier  
- `note_date` - date of the clinical note 
- `note_title` - title of the clinical note
- `note_text` - content of the clinical note

---

# RARE-PHENIX Pipeline Structure

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

Performs supervised fine-tuning of a pre-trained LLM.

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

