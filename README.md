# RARE-PHENIX for Rare Disease Phenotyping

**Please Note:** The original dataset used in this project contains patient-sensitive data from the Undiagnosed Diseases network and is **not included**.  
The expected data format is as follows:

# Expected Data Format
- `UID` – patient identifier  
- `note_date` - date of the clinical note 
- `note_title` - title of the clinical note
- `note_text` - content of the clinical note

---

# Project Structure

## Preprocessing & Training

### `Preprocess_Clinical_Notes.py`

Prepares raw clinical notes for downstream processing.

**Main steps:**
- Load the raw notes file  
- Filter notes by title and content  
- Remove unwanted sections (e.g., signatures, administrative text)  
- Concatenate notes by patient (`UID`)  
- Split long documents into smaller chunks  
- Create multiple subsets for parallel processing  

**Output:**
- Chunked and filtered note files ready for model input  

---

### `Instruction_Fine_Tuning.py`

Performs supervised fine-tuning of a base LLM using LoRA.

**Main steps:**
- Load the base instruction model  
- Load training and evaluation datasets  
- Format each example into an instruction/chat template  
- Apply LoRA for parameter-efficient fine-tuning  
- Run training with the Hugging Face `Trainer`  
- Save the fine-tuned adapter  

**Output:**
- Fine-tuned LoRA checkpoint  

---

## RARE-PHENIX Inference Pipeline

The pipeline is divided into three modules.  
Each module includes a main script and a post-processing script.

---

### Module 1

#### `RARE_PHENIX_Module1.py`

Runs first-stage model inference on the preprocessed notes.

**Typical responsibilities:**
- Load the fine-tuned model  
- Generate predictions from note chunks  
- Write raw model outputs  

#### `RARE_PHENIX_Module1_Postprocess.py`

Cleans and structures the raw outputs from Module 1.

**Typical responsibilities:**
- Parse model generations  
- Remove formatting artifacts  
- Convert outputs into a structured format for the next module  

---

### Module 2

#### `RARE_PHENIX_Module2.py`

Runs second-stage inference using outputs from Module 1.

**Typical responsibilities:**
- Load intermediate results  
- Perform additional model prompting or aggregation  
- Produce refined predictions  

#### `RARE_PHENIX_Module2_Postprocess.py`

Post-processes Module 2 outputs into a clean structured format for Module 3.

---

### Module 3

#### `RARE_PHENIX_Module3_Preprocess.py`

Prepares inputs for the final module.

**Typical responsibilities:**
- Merge and reformat outputs from Module 2  
- Construct the final prompt/input structure  

#### `RARE_PHENIX_Module3.py`

Performs final-stage inference.

**Typical responsibilities:**
- Run the model on the fully aggregated context  
- Produce the final structured predictions  

---

