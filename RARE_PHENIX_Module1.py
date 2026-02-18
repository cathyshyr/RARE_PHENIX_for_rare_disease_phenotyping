from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import pandas as pd
from peft import AutoPeftModelForCausalLM
from accelerate import Accelerator
accelerator = Accelerator()
device_index = Accelerator().process_index
device_map = {"": device_index}
device_map = "auto"


def main(file_name, batch_size):
    # Read notes_filtered_data
    notes_data_filtered = pd.read_excel(file_name)

    finetuned_model_name = 'fine_tuned_models/llama3.1/Llama-3.1-70b-2epoch/RareDis_SyntheticNotes/final_checkpoint'
    # Define variable to hold Llama3 weights naming
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # Set auth token variable from hugging face
    token = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Create model (GPU is required for quantization)

    # model = AutoModelForCausalLM.from_pretrained(model_id,
    #      cache_dir="huggingface_models/", token=token, torch_dtype=torch.float16,
    #      rope_scaling={"type": "dynamic", "factor": 2}, load_in_70bit=True)

    model = AutoPeftModelForCausalLM.from_pretrained(f"{finetuned_model_name}", token=token, device_map=device_map,
                                                     torch_dtype=torch.bfloat16)

    # Initialize a new column for HPO_Llama31_70b if it doesn't exist
    if 'Step1_Llama31_70b' not in notes_data_filtered.columns:
        notes_data_filtered['Step1_Llama31_70b'] = pd.NA

    # Split the string on the '.' to separate the base name and the extension
    base_name, extension = file_name.rsplit('.', 1)

    # Create new filename
    new_file_name = f"{base_name}_Step1_Llama31_70b_RareDis_SyntheticNotes.{extension}"

    # Function to batch a list
    def batch_list(input_list, batch_size):
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    # List to store the concatenated prompts
    all_prompts = []
    indices = []

    for index, row in notes_data_filtered.iterrows():
        if pd.isna(row['Step1_Llama31_70b']):
            indices.append(index)
            note = row['note_text']
            system_prompt = f"### TASK: Your task is to generate an HTML version of an INPUT TEXT, using HTML <span> tags to mark up only rare disease phenotypes of a patient.

### MARKUP GUIDE: Use <span class="phenotype"></span> to identify the phenotype.

### DEFINITION: A rare disease phenotype includes any observable characteristic, congenital abnormality, structural variation, functional deficit, developmental condition, subjective symptom, or measurable physiological change in a patient’s health. This covers medical conditions (e.g., physical or mental abnormalities like "autism"), physical traits (e.g., "short stature," "high palate," or "long face"), congenital abnormalities (e.g., "intrauterine growth retardation," "congenital bilateral ptosis"), structural variations (e.g., "hypoplasia of the maxilla," "abnormal skull base morphology"), functional deficits (e.g., "speech apraxia," "poor suck"), developmental conditions (e.g., "global developmental delay," "growth delay"), measurable physiological changes (e.g., "decreased response to growth hormone stimulation test"), and subjective symptoms (e.g., "pain," "fatigue").

### EXAMPLE INPUT TEXT: The combination of global developmental delay, neonatal seizure, and cerebral visual impairment necessitates a thorough genetic evaluation alongside metabolic testing to rule out treatable conditions. Continued support with physical and occupational therapy is recommended to address developmental delays and muscle tone issues. Recent MRI findings have shown ventriculomegaly and cerebral hypomyelination, with some evidence of progressive brain atrophy.

### EXAMPLE OUTPUT TEXT: The combination of <span class=“phenotype”>global developmental delay</span>, <span class=“phenotype”>neonatal seizure</span>, and <span class=“phenotype”>cerebral visual impairment</span> necessitates a thorough genetic evaluation alongside metabolic testing to rule out treatable conditions. Continued support with physical and occupational therapy is recommended to address developmental delays and muscle tone issues. Recent MRI findings have shown <span class=“phenotype”>ventriculomegaly</span> and <span class=“phenotype”>cerebral hypomyelination</span>, with some evidence of progressive <span class=“phenotype”>brain atrophy</span>. 


### INPUT TEXT: {note}

### OUTPUT:"
            all_prompts.append(system_prompt)

    # Now, batch the prompts and keep track of indices
    prompts_list = batch_list(all_prompts, batch_size)
    indices_list = batch_list(indices, batch_size)

    # Define the complete path for the output file just once
    output_file_path = f'{new_file_name}'

    for batch_index, prompt_list in enumerate(prompts_list):
        print(f'Batch:{batch_index + 1} of Total:{len(prompts_list)}', flush=True)
        inputs = tokenizer(prompt_list, return_tensors="pt", padding=True).to(device)
        output = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"],
                                max_new_tokens=1024,
                                pad_token_id=tokenizer.eos_token_id, top_p=0.99999, temperature=1e-7)
        batch_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)

        # Update the DataFrame with the outputs
        for i, text in enumerate(batch_outputs):
            original_index = indices_list[batch_index][i]  # Use the tracked indices to update the correct row
            notes_data_filtered.at[original_index, 'Step1_Llama31_70b'] = text

        # Save the DataFrame to the same file after processing each batch
        notes_data_filtered.to_excel(output_file_path, index=False)
        print(f"Saved updated DataFrame to {output_file_path} after batch {batch_index + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an .xlsx file.')
    parser.add_argument('file_name', type=str, help='The name of the .xlsx file')
    # Add an optional argument for batch size with a default value
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing (default: 20)')
    args = parser.parse_args()
    # Pass both file_name and batch_size to the main function
    main(args.file_name, args.batch_size)




