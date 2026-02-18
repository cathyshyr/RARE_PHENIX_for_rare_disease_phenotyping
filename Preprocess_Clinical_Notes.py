import json
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import re
from textwrap import dedent
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import random
import math

########################################################################
########################################################################
# Remove irrelevant notes
#
# The goal is to remove certain notes that
#   1. Contain a pdf image (i.e., "This report is a PDF image file")
#   2. Contain non-sensical information (i.e., "038075297_dqpqvcl00_1437402240000" or "041110701_fjsta3to0_1497274560000")
#   3. Contain information that probably does not pertain to patient symptoms
#       -Specific note types: Instructions, Scheduling/Appointment, Rx/Medication related, Clinical Communication, etc.
#   4. Only keep notes from individuals we have the HPO_gold for
#
########################################################################
########################################################################

notes_data = pd.read_excel("...", engine='openpyxl')
notes_data = notes_data[notes_data['UID'].isin(unique_UIDs)]
# Add a note_id column
notes_data['note_id'] = [str(i).zfill(6) for i in range(1, len(notes_data) + 1)]
# Re-order the columns
notes_data = notes_data[['note_id', 'UID', 'note_date', 'note_title', 'note_text']]
# notes_data.to_excel('notes_data.xlsx', index=False)

#1. Remove "This report is a PDF image file"
# Create a mask where note_text column contains the exact phrase
mask1 = notes_data['note_text'].str.contains("This report is a PDF image file", na=False)
notes_data_filtered = notes_data[~mask1]

#2. Remove notes that contain irrelevant code:
# Regex pattern: starts with 9 digits followed by an underscore
pattern = r'^\d{9}_'
mask2 = notes_data_filtered['note_text'].str.contains(pattern, regex = True, na = False)
notes_data_filtered = notes_data_filtered[~mask2]

#3. Remove certain notes that do not contain patient symptoms
pattern = r'(?i)(?:instruction(?:s)?|schedule(?:d)?|rx|administrative|social history|appointment|medication(?:s)?|scheduling|.*communication.*)'
mask3 = notes_data_filtered['note_title'].str.contains(pattern, regex=True, na = False)
notes_data_filtered = notes_data_filtered[~mask3]

# Remove duplicate rows after grouping by id
notes_data_filtered = notes_data_filtered.drop_duplicates(subset=['UID', 'note_text'], keep='first')

# Subset by note type
notes_data_filtered = notes_data_filtered[notes_data_filtered['note_title'].str.contains("IMAGING|Consults|Anesthesia|NEUROLOGY|Respiratory|Operative|Pathology|ECHOCARDIOGRAPHY|Rehab|ED Notes|Letter|MRI|EMG|Nutrition|Echocardiogram|UDN", na=False)]

########################################################################
########################################################################
# Chunk notes
#
# The goal is to chunk notes into smaller parts that are <= 4096 characters (i.e., context window for Llama2)
# Also, remove specific sections from the notes that are irrelevant for phenotypes: medications, social history
#
########################################################################
########################################################################
def remove_sections(my_string):
    # Pattern to find "Medications:", "Social History:", or "Current Outpatient Prescriptions:" (case-insensitive)
    # up to the next occurrence of a title followed by ":"
    # The title is assumed to be any sequence of letters (and spaces) ending with a colon
    # Use non-greedy matching (.*?) to ensure it stops at the first occurrence of a title followed by ":"
    pattern = re.compile(r'(Medications:|Social History:|Current Outpatient Prescriptions:).*?([A-Za-z ]+):',
                         re.IGNORECASE | re.DOTALL)

    # Substitute the found pattern with the title and colon (captured group) to not remove the title part
    result_string = re.sub(pattern, r'\2:', my_string)

    return result_string


def is_potential_sentence_end(char, next_char):
    """Check if a character is a potential sentence end, avoiding decimals."""
    return char in ('.', '?', '!') and not next_char.isdigit()


def find_sentence_end_index(string, start_index, end_index):
    """Find the nearest sentence end before end_index, avoiding decimals."""
    for i in range(end_index, start_index, -1):
        if is_potential_sentence_end(string[i], string[i + 1] if i + 1 < len(string) else ' '):
            return i + 1  # Include the punctuation
    return -1  # No sentence end found


def split_into_similar_length_chunks_respecting_sentences_and_decimals(my_string, max_length=800):
    if len(my_string) <= max_length:
        return [my_string]

    chunks = []
    start_index = 0
    while start_index < len(my_string):
        # Determine provisional end_index considering max_length
        end_index = min(start_index + max_length, len(my_string))

        if end_index < len(my_string):  # If not at the end of the string
            sentence_end_index = find_sentence_end_index(my_string, start_index, end_index)
            if sentence_end_index != -1 and sentence_end_index - start_index <= max_length:
                end_index = sentence_end_index
            else:
                # Walk back to find a space or a line break if no suitable sentence end is found
                while end_index > start_index and my_string[end_index] not in (' ', '\n'):
                    end_index -= 1
                if end_index == start_index:  # No space found, forced to cut at max_length
                    end_index = start_index + max_length
        chunk = my_string[start_index:end_index].strip()
        chunks.append(chunk)
        start_index = end_index

    return chunks


# Initialize a list to hold the new or unchanged rows
new_rows = []

for index, row in notes_data_filtered.iterrows():
    text = str(row['note_text'])
    # Remove irrelevant sections and update the row
    text = remove_sections(text)
    notes_data_filtered.loc[index, 'note_text'] = text
    updated_row = notes_data_filtered.loc[index]

    # Split the text if it's not <= max_length of 3500 characters
    new_text = split_into_similar_length_chunks_respecting_sentences_and_decimals(text)

    if len(new_text) == 1:
        # If the text doesn't need splitting, keep the row unchanged
        new_rows.append(updated_row)
    else:
        # If the text is split, create new rows for each part
        for i, part in enumerate(new_text, start=1):
            new_row = row.copy()
            new_row['note_text'] = part
            new_row['note_id'] = f"{row['note_id']}_string{i}"
            new_rows.append(new_row)

# Create a new DataFrame from the new_rows list
notes_data_filtered_updated_new = pd.DataFrame(new_rows).reset_index(drop=True)

########################################################################
########################################################################
# Divide data set into subsets for parallelization
#
# The goal is to divide the data set into approximately equal number of rows
#
########################################################################
########################################################################
# Define the number of subsets
num_subsets = 10

# Calculate the number of rows per subset. Using math.ceil to ensure each subset has an equal number of rows
rows_per_subset = math.ceil(len(notes_data_filtered_updated_new) / num_subsets)

# Divide the DataFrame into subsets and save each as an Excel file
for i in range(num_subsets):
    start_row = i * rows_per_subset
    end_row = start_row + rows_per_subset
    subset = notes_data_filtered_updated_new.iloc[start_row:end_row]
    # Print the subset index and the number of rows in the current subset
    i = i + 1
    print(f'Subset {i}: {len(subset)} rows')
    file_name = f'notes_data_filtered_updated{i + 30}.xlsx'
    subset.to_excel(file_name, index=False)

len(notes_data_filtered_updated_new["UID"].unique())