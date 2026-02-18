import pandas as pd
import re
from rapidfuzz import process, fuzz

# This function will post-process the LLM output from step 2
def extract_terms(input_string):
    # Find all occurrences of numbered terms using regular expression
    terms = re.findall(r'\d+\.\s+([^\n]+)', input_string)

    # # Remove duplicates while preserving order
    # unique_terms = []
    # [unique_terms.append(item) for item in terms if item not in unique_terms]
    # Remove duplicates while preserving order and remove [TERM_START] and [TERM_END]
    unique_terms = []
    for item in terms:
        # Remove the markers from the term
        cleaned_item = item.replace("[TERM_START]", "").replace("[TERM_END]", "")
        if cleaned_item not in unique_terms:
            unique_terms.append(cleaned_item)
    # Format the unique terms as a numbered list
    numbered_list = "\n".join(f"{index + 1}. {term}" for index, term in enumerate(unique_terms))

    return numbered_list


# This function will take the post-processed output and find specific HPO terms
HPO = pd.read_excel('''HPO_ID_TERM_DEFN.xlsx''')


def normalize_term_plural(term):
    """Simple normalization to handle plurals."""
    if term.endswith('s'):
        return term[:-1]
    return term


def find_matching_terms(input_list, HPO):
    # Check if input_list is empty or consists only of whitespace
    if not input_list.strip():
        return ""

    lines = input_list.strip().split('\n')
    output_list = []

    for line in lines:
        # Remove parentheses and content within
        term = re.sub(r'\s*\(.*?\)\s*', '', line.split('. ', 1)[1]) if '. ' in line else line

        # Normalize the term for matching
        normalized_term = normalize_term_plural(term.lower())

        # Attempt to find a matching term in the HPO DataFrame with simple plural handling
        match = HPO[HPO['lbl'].str.lower().apply(normalize_term_plural) == normalized_term]

        if not match.empty:
            for _, row in match.iterrows():
                output_list.append(f"{line.split('. ')[0]}. {row['id']} {row['lbl']}")
        else:
            # Output NO_MATCHING_HPO followed by the original item if no match is found
            output_list.append(f"{line.split('. ')[0]}. NO_MATCHING_HPO {term}")

    return '\n'.join(output_list)


def find_matching_terms(input_list, HPO, score_cutoff=90):
    if not input_list.strip():
        return ""

    # Normalize the labels in the HPO DataFrame
    HPO['normalized_lbl'] = HPO['lbl'].str.lower()

    lines = input_list.strip().split('\n')
    output_list = []

    for line in lines:
        term = line.split('. ', 1)[1] if '. ' in line else line
        normalized_term = term.lower()

        # Use extractOne with a built-in scorer, such as fuzz.token_sort_ratio
        best_match = process.extractOne(normalized_term, HPO['normalized_lbl'], scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)

        if best_match:
            match_id = HPO[HPO['normalized_lbl'] == best_match[0]]['id'].iloc[0]
            match_lbl = HPO[HPO['normalized_lbl'] == best_match[0]]['lbl'].iloc[0]
            output_list.append(f"{line.split('. ')[0]}. {match_id} {match_lbl}")
        else:
            output_list.append(f"{line.split('. ')[0]}. NO_MATCHING_HPO {term}")

    return '\n'.join(output_list)


def find_subset_match(input_list, HPO):
    if not input_list.strip():
        return ""

    # Normalize the labels in the HPO DataFrame
    HPO['normalized_lbl'] = HPO['lbl'].str.lower()

    lines = input_list.strip().split('\n')
    output_list = []
    item_number = 1  # Initialize counter for the item numbers

    for line in lines:
        term = line.split('. ', 1)[1] if '. ' in line else line
        normalized_term = term.lower()

        # Check if any HPO term is a subset of the input term
        matches = HPO[HPO['normalized_lbl'].apply(lambda x: x in normalized_term)]

        if not matches.empty:
            for index, match in matches.iterrows():
                match_id = match['id']
                match_lbl = match['lbl']
                output_list.append(f"{item_number}. {match_id} {match_lbl}")
                item_number += 1  # Increment the item number for each match
        else:
            output_list.append(f"{item_number}. NO_MATCHING_HPO {term}")
            item_number += 1  # Increment the item number even if there's no match

    return '\n'.join(output_list)


def combine_matches(score_match, subset_match):
    if not subset_match.strip():
        return score_match  # If subset_match is empty, return score_match as is.

    # Splitting the inputs into lists of strings by lines
    score_match_lines = score_match.strip().split('\n')
    subset_match_lines = subset_match.strip().split('\n')

    # To remove duplicates, we'll use a set to track seen suffixes
    seen = set()
    combined_list = []
    item_number = 1

    # First add from score_match while removing duplicates
    for line in score_match_lines:
        suffix = line.split('. ', 1)[1] if '. ' in line else line
        if suffix not in seen:
            combined_list.append(f"{item_number}. {suffix}")
            seen.add(suffix)
            item_number += 1

    # Now add from subset_match while removing duplicates
    for line in subset_match_lines:
        suffix = line.split('. ', 1)[1] if '. ' in line else line
        if suffix not in seen:
            combined_list.append(f"{item_number}. {suffix}")
            seen.add(suffix)
            item_number += 1

    # Joining all lines back into a single string
    return '\n'.join(combined_list)

def split_terms(string):
    """
    Split the string by either ', ' or '|'
    """
    if pd.isna(string) or not isinstance(string, str):
        return []
    return re.split(', | \|', string)

def filter_rows(row, acronyms):
    """
    Check if the 'Step1_Llama31_7b_Clean_Split' contains any of the identified acronyms.
    """
    if pd.isna(row) or not isinstance(row, str):
        return True  # If the row is NaN or not a string, we don't filter it out
    terms_split = split_terms(row)
    return not any(term in acronyms for term in terms_split)

def is_acronym(term, full_terms):
    """
    Check if the term is an acronym of any of the full terms.
    This function is a simple implementation and might need adjustments for complex scenarios.
    """
    term = term.replace('.', '').upper()
    for full_term in full_terms:
        words = full_term.split()
        if len(words) == len(term) and all(word[0].upper() == t for word, t in zip(words, term)):
            return True
    return False


# Handle Example 2
# '<span class="condition">negative for tachypnea</span> and <span class="condition">cyanosis</span>.  Neurologic:†<span class="condition">negative for local weakness</span> or <span class="condition">seizure</span>'
def handle_example_2(first_output_section):
    # all_conditions = re.findall(r'<span class="condition">(.*?)</span>', first_output_section, re.IGNORECASE)
    # Split the string by <span class="condition"> and </span>
    parts = re.split(r'(<span class="condition">.*?</span>)', first_output_section, flags=re.IGNORECASE | re.DOTALL)

    # Process each part separately, ignoring the content inside <span> tags
    updated_parts = []
    for part in parts:
        if '<span class="condition">' in part:
            updated_parts.append(part)  # Append without change
        else:
            updated_part = re.sub(r'(?<!\b)no|no(?! \b)', replace_no, part, flags=re.IGNORECASE)
            updated_parts.append(updated_part)

    # Join all parts back together
    first_output_section = ''.join(updated_parts)
    exclusion_pattern_preceding = r'(?i)(negative for|no evidence for|no history of|no|normal|not significant for).{0,60}<span class="condition">(.*?)</span>(?:\s*(?:and|or)\s*<span class="condition">(.*?)</span>)*'
    matches_preceding = re.finditer(exclusion_pattern_preceding, first_output_section)
    excluded_conditions_preceding = []
    for match in matches_preceding:
        negated_conditions = re.findall(r'<span class="condition">(.*?)</span>', match.group(0), re.IGNORECASE)
        excluded_conditions_preceding.extend(negated_conditions)
    return excluded_conditions_preceding

def replace_no(match):
    # If 'no' is found within the <span class="condition">...</span>, skip replacing
    if '<span class="condition">' in match.string[match.start()-80:match.end()+80]:
        return match.group(0)
    else:
        return ''

def postprocess_part2(model_name):
    # Read in the data frames for Step 2 with unique HPO queries
    Step2_Output_Unique = []
    orig_model_name = model_name
    # Loop over the range of numbers from 1 to 30
    for i in range(1, 41):
        if "RareDis_SyntheticNotes" in model_name:
            model = model_name.split("_")[0]
            file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model}_Clean_Split_Unique_RareDis_SyntheticNotes_Step2_Llama31_{model}.xlsx"
        else:
            file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_Unique_Step2_Llama31_{model_name}.xlsx"

        # Construct the file name based on the current iteration
        # file_name = f"notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_Unique_RareDis_SyntheticNotes_Step2_Llama31_{model_name}.xlsx"
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_name)
        # Append the DataFrame to the list
        Step2_Output_Unique.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    Step2_Output_Unique_All = pd.concat(Step2_Output_Unique, ignore_index=True)

    for i in range(1, 41):
        if "RareDis_SyntheticNotes" in orig_model_name:
            # This will remove 'RareDis_SyntheticNotes' from the model_name and overwrite it
            # This is fine because the rest of the code will just use the model (i.e., 8b, 13b, 70b) without the RareDis_SyntheticNotes modifier
            model_name = orig_model_name.split("_")[0]
            # Construct the file name based on the current value of i
            input_file_name = f"./notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_RareDis_SyntheticNotes.xlsx"
            output_file_name = f"./notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_RareDis_SyntheticNotes_Step2_Llama31_{model_name}_Clean.xlsx"
        else:
            input_file_name = f"./notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split.xlsx"
            output_file_name = f"./notes_data_filtered_updated{i}_Step1_Llama31_{model_name}_Clean_Split_Step2_Llama31_{model_name}_Clean.xlsx"

        # Read the current file
        df = pd.read_excel(input_file_name)
        df = pd.merge(df, Step2_Output_Unique_All, on = f'Step1_Llama31_{model_name}_Clean_Split', how='inner')

        #############################################################
        #          Handle acronyms (i.e., if OSA and obstructive    #
        #          sleep apnea both appear, then keep the full term)#
        #############################################################
        # Identify rows with acronyms
        acronyms = set()

        # for index, row in df.iterrows():
        #     terms = split_terms(row['Step1_Llama31_{0}_Clean'.format(model_name)])
        #     for term in terms:
        #         if is_acronym(term, terms):
        #             acronyms.add(term)

        for index, row in df.iterrows():
            print(index)
            # Clean HTML tags from the specified column
            if pd.isna(row['Step1_Llama31_{0}_Clean'.format(model_name)]):
                cleaned_text = "None"
            else:
                cleaned_text = re.sub(r'<span class="condition">|</span>', '',
                                      row['Step1_Llama31_{0}_Clean'.format(model_name)], flags=re.IGNORECASE)
            # Update the DataFrame directly with the cleaned text
            df.at[index, 'Step1_Llama31_{0}_Clean'.format(model_name)] = cleaned_text

            if not pd.isna(row['Step1_Llama31_{0}_Clean_Split'.format(model_name)]):
                # Perform the cleaning operation if it's not NaN
                cleaned_text_split = re.sub(r'<span class="condition">|</span>', '',
                                            row['Step1_Llama31_{0}_Clean_Split'.format(model_name)], flags=re.IGNORECASE)
                # Optionally update the DataFrame with the cleaned text
                df.at[index, 'Step1_Llama31_{0}_Clean_Split'.format(model_name)] = cleaned_text_split

        # Filter out rows based on identified acronyms
        df = df[df['Step1_Llama31_{0}_Clean_Split'.format(model_name)].apply(lambda x: filter_rows(x, acronyms))]

        ############################################################
        #                         Negation                         #
        ############################################################
        # Add 'Step1_Llama31_Xb_Clean_Split_Negated' column if it doesn't exist
        if 'Step1_Llama31_{0}_Clean_Split_Negated'.format(model_name) not in df.columns:
            df['Step1_Llama31_{0}_Clean_Split_Negated'.format(model_name)] = pd.NA

        for index, row in df.iterrows():
            # Rule-based removal of negations
            # Split the text by '### OUTPUT:' and take the first occurrence
            text = df.at[index, 'Step1_Llama31_{0}'.format(model_name)]
            # Note that this first_output_section extraction may only work for Llama31 output
            # Need different post-processing for Llama31
            first_output_section = text.split('### OUTPUT:')[1].split('###')[0]

            # Step 1: Match all conditions
            all_conditions = re.findall(r'<span class="condition">(.*?)</span>', first_output_section, re.IGNORECASE)

            # Step 2: Define exclusion patterns
            # This pattern captures conditions based on both preceding and following criteria
            # This is going to catch 'negative for <span class="condition">goiter</span>'
            exclusion_pattern_preceding = r'(?i)(negative for|no evidence for|no history of|no|normal|not significant for|not noted)\s+(<span class="condition">.*?</span>(?:\s*(?:,|\s|or|and)?\s*<span class="condition">.*?</span>)*)'
            exclusion_pattern_following = r'<span class="condition">(.*?)</span>\s*(Neg Hx|Neg Dx)'

            # Find all matches for the preceding pattern
            matches_preceding = re.finditer(exclusion_pattern_preceding, first_output_section)
            excluded_conditions_preceding = []
            # for match in matches_preceding:
            #     excluded_conditions_preceding.extend(
            #         re.findall(r'<span class="condition">(.*?)</span>', match.group(2), re.IGNORECASE))
            for match in matches_preceding:
                # Extract conditions from the matched group, handling each match
                # Consider match.group() since it captures the full pattern including negation phrases and multiple conditions
                all_matched_text = match.group()
                negated_conditions = re.findall(r'<span class="condition">(.*?)</span>', all_matched_text,
                                                re.IGNORECASE)

                # Extend the list of excluded conditions with the found conditions
                excluded_conditions_preceding.extend(negated_conditions)

            # Find all matches for the following pattern
            excluded_conditions_following = re.findall(exclusion_pattern_following, first_output_section, re.IGNORECASE)
            excluded_conditions_following = [match[0] for match in excluded_conditions_following]

            # Combine excluded conditions from both patterns
            excluded_conditions = excluded_conditions_preceding + excluded_conditions_following

            # excluded_conditions2 = handle_example_2(first_output_section)
            # excluded_conditions = excluded_conditions + excluded_conditions2
            # # Remove duplicates
            # excluded_conditions = list(set(excluded_conditions))

            # Step 3: Filter out excluded conditions from all_conditions
            valid_conditions = [cond for cond in all_conditions if cond not in excluded_conditions]

            # Remove duplicates while preserving order
            unique_valid_conditions = list(dict.fromkeys(valid_conditions))

            # Check if any negated terms are included in Step1_Llama31_Xb_Clean_Split, if so, set Step1_Llama31_Xb_Clean_Split_Negated to "Negated"
            # Since text_to_check is now a list, we can directly use it after converting each condition to lowercase
            conditions = [condition.strip().lower() for condition in unique_valid_conditions]
            conditions = [re.sub(r'<span class="condition">|</span>', '', condition) for condition in conditions]

            # Retrieve the DataFrame entry, split by comma, and convert to lowercase for case-insensitive comparison
            if pd.isna(df.at[index, 'Step1_Llama31_{0}_Clean_Split'.format(model_name)]):
                df.at[index, 'Step1_Llama31_{0}_Clean_Split'.format(model_name)] = "NA"
                df_conditions = "na"
            else:
                df_conditions = [condition.strip().lower() for condition in
                             df.at[index, 'Step1_Llama31_{0}_Clean_Split'.format(model_name)].split(',')]

            # Check if any condition from the conditions list matches the conditions in the DataFrame entry
            if any(condition.strip().lower() in conditions for condition in df_conditions):
                df.at[index, 'Step1_Llama31_{0}_Clean_Split_Negated'.format(model_name)] = df.at[index, 'Step1_Llama31_{0}_Clean_Split'.format(model_name)]
            else:
                df.at[index, 'Step1_Llama31_{0}_Clean_Split_Negated'.format(model_name)] = "Negated"
                df.at[index, 'Step2_Llama31_{0}'.format(model_name)] = "None"

            # This will take care of cases where "negative for" is wrapped in the <span class="condition"></span> tags
            if 'negative' in df.at[index, f'Step1_Llama31_{model_name}_Clean_Split'].lower() \
                    or 'negative for' in df.at[index, f'Step1_Llama31_{model_name}_Clean_Split'].lower() \
                    or 'no ' in df.at[index, f'Step1_Llama31_{model_name}_Clean_Split'].lower():
                df.at[index, f'Step1_Llama31_{model_name}_Clean_Split_Negated'] = "Negated"
                df.at[index, f'Step2_Llama31_{model_name}'] = "None"

        # Add 'Step1_Llama31_{0}_Clean' column if it doesn't exist
        if 'Step1_Llama31_{0}_Clean_HPO'.format(model_name) not in df.columns:
            df['Step1_Llama31_{0}_Clean_HPO'.format(model_name)] = pd.NA

        # Add 'Step2_Llama31_{0}_Clean' column if it doesn't exist
        if 'Step2_Llama31_{0}_Clean'.format(model_name) not in df.columns:
            df['Step2_Llama31_{0}_Clean'.format(model_name)] = pd.NA

        # Process each row for 'Step2_Llama31_Xb_Clean'
        for index, row in df.iterrows():
            print(f"Processing file {i}, row number: {index}")
            if pd.isna(row['Step1_Llama31_{0}_Clean_HPO'.format(model_name)]):
                if df.at[index, 'Step1_Llama31_{0}_Clean_Split_Negated'.format(model_name)] != "Negated":
                    text = df.at[index, 'Step1_Llama31_{0}_Clean_Split'.format(model_name)]
                    df.at[index, 'Step1_Llama31_{0}_Clean_HPO'.format(model_name)] = f'1. {text.capitalize()}'
                else:
                    df.at[index, 'Step1_Llama31_{0}_Clean_HPO'.format(model_name)] = ""
            if pd.isna(row['Step2_Llama31_{0}_Clean'.format(model_name)]):
                if not extract_terms(df.at[index, 'Step2_Llama31_{0}'.format(model_name)]):  # Check if extracted_terms is empty
                    df.at[index, 'Step2_Llama31_{0}_Clean'.format(model_name)] = "None"
                else:
                    df.at[index, 'Step2_Llama31_{0}_Clean'.format(model_name)] = extract_terms(df.at[index, 'Step2_Llama31_{0}'.format(model_name)])

        # Add 'Step2_Llama31_Xb_Clean_HPO' column if it doesn't exist (This is Tier 2 HPO - i.e., HPO terms linked using RAG)
        if 'Step2_Llama31_{0}_Clean_HPO'.format(model_name) not in df.columns:
            df['Step2_Llama31_{0}_Clean_HPO'.format(model_name)] = pd.NA

        if 'Step1_Llama31_{0}_Clean_HPO_Subset'.format(model_name) not in df.columns:
            df['Step1_Llama31_{0}_Clean_HPO_Subset'.format(model_name)] = pd.NA

        for index, row in df.iterrows():
            if pd.isna(row['Step2_Llama31_{0}_Clean_HPO'.format(model_name)]):
                print(f"Processing file {i}, row number: {index}")
                df.at[index, 'Step1_Llama31_{0}_Clean_HPO_Subset'.format(model_name)] = find_subset_match(df.at[index, 'Step1_Llama31_{0}_Clean_HPO'.format(model_name)], HPO)
                # Tier 1 HPO match - exact mentions of HPO terms detected by Step 1
                df.at[index, 'Step1_Llama31_{0}_Clean_HPO'.format(model_name)] = find_matching_terms(df.at[index, 'Step1_Llama31_{0}_Clean_HPO'.format(model_name)], HPO)
                # Tier 2 HPO match - RAG-linked HPO terms from Step 2
                # score_match = find_matching_terms(df.at[index, 'Step2_Llama31_{0}_Clean'.format(model_name)], HPO)
                df.at[index, 'Step2_Llama31_{0}_Clean_HPO'.format(model_name)] = find_matching_terms(df.at[index, 'Step2_Llama31_{0}_Clean'.format(model_name)], HPO)
                df.at[index, 'Step2_Llama31_{0}_Clean_HPO'.format(model_name)] = combine_matches(df.at[index, 'Step2_Llama31_{0}_Clean_HPO'.format(model_name)], df.at[index, 'Step1_Llama31_{0}_Clean_HPO_Subset'.format(model_name)])

        # Save the processed DataFrame to a new Excel file
        df.to_excel(output_file_name, index=False)

postprocess_part2("70b_RareDis_SyntheticNotes")