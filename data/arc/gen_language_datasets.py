"""
gen_language_datasets.py | Author: Catherine Wong

This generates the separated test and train splits containing the language annotations for each CLEVR task, along with a vocabulary file, in
the standardized dataset format used in the other DreamCoder + Language domains.

For now, we only take the OUTPUT description from the language data.

It expects CSVs named {train, test} with the following rows (generated from an accompanying human experiment): task_id,description,num_successful_builds,total_num_builds

It will then generate a directory and files of the form:
    {arc-language-directory}/
        train/ -> language.json and vocab.json
        test/ ..
for each of the question classes.
Each language file is a dictionary in the form:
    {
    "arc_file.json" : [array of space tokenized question text],
    }
where the question texts are sorted in order of their success rate.

Example usage: python3 generate_language_datasets.py --questions_dir clevr_dreams/questions
    --input_dir data/arc/language/raw
    --output_dir data/arc/language
"""
import os, argparse, re, json
import pathlib
import pandas as pd
from collections import defaultdict

LANGUAGE_FILENAME = 'language.json'
VOCAB_FILENAME = 'vocab.json'
DATASET_SPLIT_NAMES = ['train', 'test']
TASK_ID = 'task_id'
DESCRIPTION = 'description'
NUM_SUCCESSFUL = 'num_successful_builds'
NUM_TOTAL = 'total_num_builds'

# File handling.
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True,
    help="Directory containing raw CSV files to extract train and test language for.")
parser.add_argument('--output_dir', required=True,
    help="Directory to write out the processed language files.")
    
def get_input_files(args):
    """Gets any valid input files from the directory, based on the split.
    :ret: {split : [full_input_filepath]}
    """
    candidate_input_files = [file for file in os.listdir(args.input_dir)]
    valid_input_files = {
        split : [os.path.join(args.input_dir, split)]
        for split in candidate_input_files if split in DATASET_SPLIT_NAMES
    }
    return valid_input_files

def create_output_dirs(args, input_files):
    """Creates the {output_dir}/ -> test/ train/ directories for the given question files, and stores these directories with the question_files for writing output.
    Returns a dict of the form: { split : [full_input_filepath, full_output_dir] }"""
    for split in input_files:
        output_directory = os.path.join(args.output_dir, split)
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        input_files[split].append(output_directory)
    return input_files

def iteratively_write_out_processed_language_dataset(args, input_files_and_output_dirs):
    for split in input_files_and_output_dirs:
        input_file, output_dir = input_files_and_output_dirs[split]
        print(f"...writing language dataset for {input_file} --> {output_dir}.")
        with open(input_file, 'r') as f:
            input_data = pd.read_csv(input_file).to_dict()
            processed_language, vocab = get_processed_language_and_vocab(input_data)  
            
            # Write out the processed langage object.
            output_filename = os.path.join(output_dir, LANGUAGE_FILENAME)
            print(f"Writing question text for [{len(processed_language)}] tasks to {output_filename}")
            with open(output_filename, 'w') as f:
                json.dump(processed_language, f)
            
            # Write out the vocabulary object.
            output_filename= os.path.join(output_dir, VOCAB_FILENAME)
            print(f"Writing vocab of [{len(vocab)}] words to {output_filename}")
            with open(output_filename, 'w') as f:
                json.dump(vocab, f)     

def get_processed_language_and_vocab(input_data):
    """
    Generates the processed_language and vocab objects from the raw language data.
    task names of the form: {ARC_JSON_ID.json}    
    Returns:
        processed_language:  { task_name :  [array of processed_text sentences sorted by build success rate]}
        vocab = [vocabulary_tokens]
    """
    descriptions_to_success_rate = defaultdict(list)
    processed_language = defaultdict(list)
    vocab = set()
    for task_idx in input_data[TASK_ID]:
        task_name, description, num_successful, num_total = input_data[TASK_ID][task_idx], input_data[DESCRIPTION][task_idx], input_data[NUM_SUCCESSFUL][task_idx], input_data[NUM_TOTAL][task_idx]
        
        num_total += 1 # Pad with a pseudo observation.
        success_rate = float(num_successful) / float(num_total)
        descriptions_to_success_rate[task_name].append(success_rate)
        
        processed = process_description_text(description)
        vocab.update(processed.split())
        processed_language[task_name].append(processed)
    vocab = list(vocab)
    
    for task_name in processed_language:
        processed_language[task_name] = [description for description, success_rate in sorted(zip(processed_language[task_name], descriptions_to_success_rate[task_name]))]
    return processed_language, vocab  

def process_description_text(description):
    """
    Processing to tokenize the question text into a standarized format.
    We only take the 'output' description from the language data.
    We remove punctuation and capitalization, and split plural objects using a heuristic. We also turn "NUMxNUM" into "NUM BY NUM".
    """
    output_delimiter = 'To make the output, you have to...'
    if output_delimiter in description:
        description = description.split(output_delimiter)[-1]
    description = description.lower()
    description = str(re.sub('(\d)(x)(\d)', r'\1 by \3', description))
    punctuation = ["?", ".", ",", ";", "(", ")", "\\", "/", '\"']
    description = "".join([c if (c.isalnum() or c. isspace()) else " " for c in description ])
    known_not_plurals = ["is", "this", "as"]
    plurals = [word for word in description.split() if word.endswith('s') and word not in known_not_plurals]
    for p in plurals:
        description = description.replace(p, f'{p[:-1]} s')
    description = " ".join(description.split())
    description = description.strip()
    return description

def main(args):
    input_files = get_input_files(args)
    input_files_and_output_dirs = create_output_dirs(args, input_files)
    iteratively_write_out_processed_language_dataset(args, input_files_and_output_dirs)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)  