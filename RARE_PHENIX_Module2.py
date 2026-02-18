from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pandas as pd
import torch
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext, set_global_service_context, Settings
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, download_loader

def main(file_name, record_id):
    # Read notes_filtered_data
    notes_data_filtered = pd.read_excel(file_name)

    # Define variable to hold Llama3 weights naming
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Set auth token variable from hugging face
    token = "hf_"
    device = 'auto'

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct", token=token)
    # tokenizer = AutoTokenizer.from_pretrained("meta/Meta-Llama-3-8B-Instruct-hf",
    #                                          token=token)

    # Create model (GPU is required for quantization)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct", token=token, torch_dtype=torch.float16,
         rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained("meta/Meta-Llama-3-8B-Instruct-hf", token=token, torch_dtype=torch.float16,
    #     rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)


    # Initialize a new column for HPO if it doesn't exist
    if 'Step2' not in notes_data_filtered.columns:
        notes_data_filtered['Step2'] = pd.NA

    # Split the string on the '.' to separate the base name and the extension
    base_name, extension = file_name.rsplit('.', 1)

    # Append '_HPO' to the base name, then add the extension back
    new_file_name = f"{base_name}_Step2_StudyID_{record_id}.{extension}"

    system_prompt = '''<s>[INST] <<SYS>>
    ### TASK: Your task is to search for terms that match a given H_TERM and output them in a numbered list. Output 'None' if there is no matching term. Respond ONLY based on the nodes I provided and do not rely on prior knowledge.
    <</SYS>>
    ### H_TERM: 
    '''

    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(context_window=4096,
                         max_new_tokens=56,
                         system_prompt=system_prompt,
                         query_wrapper_prompt=query_wrapper_prompt,
                         model=model,
                         tokenizer=tokenizer,
                         device_map="auto",
                         # generate_kwargs={"do_sample": False, "num_beams": 1}
                         # generate_kwargs = {"top_p": 0.5, "temperature": 0.5}
                         generate_kwargs={"top_p": 0.9999999, "temperature": 1e-7}
                         )

    # embeddings = LangchainEmbedding(
    #     HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
    #     # HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    #     # HuggingFaceEmbeddings(model_name="sentence-transformers/all-roberta-large-v1")
    # )

    embed_dir = ''
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=""))
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_dir))

    df = pd.read_excel('HPO_ID_TERM_DEFN.xlsx')
    print("Step 2: RAG query engine initializing...")
    nodes = []

    # Iterate over each row in the DataFrame and create a TextNode for each
    for index, row in df.iterrows():
        # Filter out NaN values and convert the row to a string
        row_string = ' '.join(map(str, [val for val in row.values if pd.notna(val)]))
        node = TextNode(text=row_string)
        nodes.append(node)

    # Create new service context instance
    # service_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embeddings
    # )
    Settings.llm = llm
    Settings.embed_model = embeddings
    Settings.num_output = 56
    Settings.context_window = 4096

    index = VectorStoreIndex(nodes, similarity_top_k=10, embed_model = embeddings)

    query_engine = index.as_query_engine()
    print("Step 2: RAG query engine initialized")

    # Find the first non-empty entry in 'Step1_Clean_Split'
    start_index = notes_data_filtered[~notes_data_filtered['Step1_Clean_Split'].isna()].index.min()

    # Initialize a counter for tracking the number of processed records since the last save
    records_since_last_save = 0

    # Loop through the DataFrame rows using iterrows (consider using .itertuples() for better performance)
    # Check if start_index is not NaN (i.e., there is at least one non-empty entry)
    if pd.notna(start_index):
        # Loop through the DataFrame rows starting from the first non-empty entry
        for index in range(start_index, len(notes_data_filtered)):
            row = notes_data_filtered.iloc[index]

            # Skip this iteration if 'Step1_Clean_Split' is empty
            if pd.isna(row['Step1_Clean_Split']):
                continue

            # Check if 'Step2' is NaN before proceeding
            if pd.isna(row['Step2']):
                print(f"Step 2: Processing row number: {index}")

                # Perform the query using the non-empty 'Step1_Clean_Split'
                response = query_engine.query(notes_data_filtered.at[index, 'Step1_Clean_Split'])
                notes_data_filtered.at[index, 'Step2'] = response.response

                # Increment the processed records counter
                records_since_last_save += 1

                # Check if it's time to save (every 20 records)
                if records_since_last_save >= 20:
                    notes_data_filtered.to_excel(f'{new_file_name}', index=False)
                    records_since_last_save = 0  # Reset the counter after saving

    # Ensure the DataFrame is saved after finishing all rows, to capture any remaining changes
    notes_data_filtered.to_excel(f'{new_file_name}', index=False)
    print(f"Step 2: Terms standardized to HPO terms and saved to {new_file_name}")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process an .xlsx file.')

    # Add an argument for the .xlsx file path
    parser.add_argument('file_name', type=str, help='The name of the .xlsx file')

    # Add an argument for the record ID
    parser.add_argument('record_id', type=str, help='The record ID')

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the main function
    print("Step 2: Starting...")
    print(f"Step 2: Processing file {args.file_name} for record_id {args.record_id}")
    main(args.file_name, args.record_id)


