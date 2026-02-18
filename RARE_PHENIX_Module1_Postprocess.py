import pandas as pd
import re
import math

def extract_matching_output(input_text, output_text):
    # Choose a part from the end of the input text to use for matching
    part_to_match = input_text[-10:]  # Last 10 characters from the input

    # Define the regex pattern to match the part immediately followed by </span> or not
    # The '|' operator is used for 'or' in regex
    pattern = re.escape(part_to_match) + "(</span>)?"

    # Search for this pattern in the output text
    match = re.search(pattern, output_text)

    if match:
        # Extract everything in the output text up to the end of the match
        extracted_output = output_text[:match.end()]
        return extracted_output
    else:
        return None


def convert_HTML_output(text):
    # Split the text by '### OUTPUT:' and take the first occurrence
    # first_output_section = text.split('### OUTPUT:')[1].split('###')[0]
    # Using regex to find the specific section
    # match = re.search(r'### OUTPUT:(.*?)\n', text)
    input_match = re.search(r'### INPUT TEXT: (.+?)\n### OUTPUT:', text, re.DOTALL)
    output_match = re.search(r'### OUTPUT: (.+)$', text, re.DOTALL)

    if input_match and output_match:
        input_text = input_match.group(1).strip()
        output_text = output_match.group(1).strip()
    else:
        input_text = ''
        output_text = ''

    # Extracting the matched text if it exists
    first_output_section = extract_matching_output(input_text, output_text)
    # Define the regular expression pattern for extracting terms within <span class="condition"></span>
    pattern = r'<span class=[\'"]condition[\'"]>(.*?)</span>'

    if first_output_section:
        # Find all matches of the pattern in the first output section
        conditions = re.findall(pattern, first_output_section)

        # Remove duplicates in a case-insensitive manner by using a dictionary
        # This approach also preserves the original case of the first occurrence of each condition
        unique_conditions_dict = {}
        for condition in conditions:
            unique_conditions_dict[condition.lower()] = condition

        # Extract the conditions from the dictionary, preserving the order of first occurrences
        unique_conditions = list(unique_conditions_dict.values())

        # Check if unique_conditions list is empty and set output accordingly
        if unique_conditions:
            # Change delimiter to pipe | instead of ', '
            conditions_string = '|'.join(unique_conditions)
        else:
            conditions_string = 'None'
    else:
        conditions_string = 'None'
    return conditions_string


# Function to remove case-insensitive duplicates within each group
def remove_case_insensitive_duplicates(group, model_name):
    group['temp_lower'] = group['Step1_Llama31_{0}_Clean_Split'.format(model_name)].str.lower()
    is_duplicate = group['temp_lower'].duplicated(keep='first')
    group = group[~is_duplicate]
    return group.drop('temp_lower', axis=1)

def postprocess_part1(model_name):
    orig_model_name = model_name
    for i in range(1, 41):  # Loop through the range of file indices
        model_name = orig_model_name.split("_")[0]
        if "RareDis_SyntheticNotes" in orig_model_name:
            file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_RareDis_SyntheticNotes.xlsx"
            output_file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_RareDis_SyntheticNotes.xlsx"
            split_output_file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_RareDis_SyntheticNotes.xlsx"
        else:
            file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}.xlsx"
            output_file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean.xlsx"
            split_output_file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split.xlsx"

        # Read in the current file
        notes_data_filtered_output = pd.read_excel(file_name)

        if 'Step1_Llama31_{0}_Clean'.format(model_name) not in notes_data_filtered_output.columns:
            notes_data_filtered_output['Step1_Llama31_{0}_Clean'.format(model_name)] = pd.NA

        for index, row in notes_data_filtered_output.iterrows():
            if pd.isna(row['Step1_Llama31_{0}_Clean'.format(model_name)]):
                print(f"Processing file {i}, row number: {index}")
                cleaned_output = convert_HTML_output(row['Step1_Llama31_{0}'.format(model_name)])
                notes_data_filtered_output.at[index, 'Step1_Llama31_{0}_Clean'.format(model_name)] = cleaned_output

        # Save the cleaned data to an Excel file
        notes_data_filtered_output.to_excel(output_file_name, index=False)

        # Split 'Step1_Llama31_13b_Clean' column into separate rows
        notes_data_filtered_output['Step1_Llama31_{0}_Clean_Split'.format(model_name)] = notes_data_filtered_output[
            'Step1_Llama31_{0}_Clean'.format(model_name)].str.split('|')
        notes_data_filtered_output_final = notes_data_filtered_output.explode('Step1_Llama31_{0}_Clean_Split'.format(model_name))

        # Remove rows where there were no extracted entities and also duplicate entities
        notes_data_filtered_output_final = notes_data_filtered_output_final[
            notes_data_filtered_output_final['Step1_Llama31_{0}_Clean_Split'.format(model_name)] != "None"]

        notes_data_filtered_output_final = notes_data_filtered_output_final[
            notes_data_filtered_output_final['Step1_Llama31_{0}_Clean_Split'.format(model_name)] != ""]

        # Group by 'UID' and apply the function to remove case-insensitive duplicates
        notes_data_filtered_output_final = notes_data_filtered_output_final.groupby('UID', group_keys=False).apply(
            lambda group: remove_case_insensitive_duplicates(group, model_name)
        )

        # Save the final DataFrame to an Excel file
        notes_data_filtered_output_final.to_excel(split_output_file_name, index=False)

    ################################################################################
    #
    # Create a file with only the unique entities to use RAG on
    #
    #################################################################################
    # Initialize an empty DataFrame to hold the concatenated column
    concatenated_df = pd.DataFrame()

    # Loop over the range of numbers from 1 to 30
    for i in range(1, 41):
        # Construct the file name based on the current iteration
        if "RareDis_SyntheticNotes" in orig_model_name:
            file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_RareDis_SyntheticNotes.xlsx"
        else:
            file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split.xlsx"

        # Read the current file into a DataFrame
        current_df = pd.read_excel(file_name)

        # Assuming 'Step1_Llama31_{model_name}_Clean_Split' is the column you're interested in,
        # extract this column and convert it to a DataFrame before appending
        # This step ensures that we're working with DataFrames and can use pd.concat effectively
        column_df = current_df[['Step1_Llama31_{0}_Clean_Split'.format(model_name)]].copy()

        # Concatenate the current column DataFrame to the accumulated DataFrame
        concatenated_df = pd.concat([concatenated_df, column_df], ignore_index=True)

    # Remove duplicates in the 'Step1_Llama31_{model_name}_Clean_Split' column of the concatenated DataFrame
    concatenated_df = concatenated_df.drop_duplicates(subset=['Step1_Llama31_{0}_Clean_Split'.format(model_name)])

    # Define the number of subsets
    num_subsets = 40

    # Calculate the number of rows per subset. Using math.ceil to ensure each subset has an equal number of rows
    rows_per_subset = math.ceil(len(concatenated_df) / num_subsets)

    # Divide the DataFrame into subsets and save each as an Excel file
    for i in range(num_subsets):
        start_row = i * rows_per_subset
        end_row = start_row + rows_per_subset
        subset = concatenated_df.iloc[start_row:end_row]
        # Print the subset index and the number of rows in the current subset
        i = i + 1
        print(f'Subset {i}: {len(subset)} rows')
        if "RareDis_SyntheticNotes" in orig_model_name:
            output_file_name = f'notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_Unique_RareDis_SyntheticNotes.xlsx'
        else:
            output_file_name = f'notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_Unique.xlsx'
        subset.to_excel(output_file_name, index=False)


postprocess_part1("70b_RareDis_SyntheticNotes")
