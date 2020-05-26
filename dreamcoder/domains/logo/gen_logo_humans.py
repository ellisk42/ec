"""
Generates human language dataset for LOGO.
"""

import argparse, json, os, itertools, random, shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_file', default='/Users/catwong/Desktop/zyzzyva/code/ec_language_exp/psiturk/experiments/logo_unlimited_200_humans/logo_reference_all_stimuli.json',
    help="JSON file containing raw MTurk data.")
parser.add_argument('--input_synthetic_dir', default='/Users/catwong/Desktop/zyzzyva/code/ec/data/logo/language/logo_unlimited_200/synthetic',
    help="Directory containing the synthetic version of all tasks.")
parser.add_argument('--output_dir', default='/Users/catwong/Desktop/zyzzyva/code/ec/data/logo/language/logo_unlimited_200/humans',
    help="Directory where we will write the final processed human data.")

def get_task_name(img_name):
    img_name = img_name.split("/")[-1].split("_name_")[-1].split(".png")[0]
    if "copy" in img_name:
        img_head = img_name.split("_copy")[0]
        img_head = img_head.replace("_", " ")
        img_name = img_head + "_copy" + img_name.split("_copy")[-1]
        return img_name
    else:
        img_name = img_name.replace("_", " ")
        return img_name

numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
def preprocess_text(sentence):
    sentence = sentence.lower()
    # Remove punctuation
    sentence = sentence.replace("-", " ")
    sentence = "".join([c for c in sentence if (c.isalpha() or c.isdigit() or c.isspace())])
    sentence = sentence.replace("gons", "gon s")
    sentence = sentence.replace("squares", "square s")
    sentence = sentence.replace("circles", "circle s")
    # Canonicalize all of the numbers
    for i, num in enumerate(numbers):
        sentence = sentence.replace(num, str(i))
    # Add a space after every number
    for i in range(10):
        sentence = sentence.replace(str(i), str(i)+ " ")
        sentence = " ".join(sentence.split())
    return sentence
    

def build_vocab(task_data):
    vocab = set()
    for t in task_data:
        for s in task_data[t]:
            vocab.update(s.split())
    return sorted(list(vocab))

def main(args):
    # Read in all of the original task names.
    task_names = {}
    vocab = set()
    for split in ('train', 'test'):
        synthetic_fn = os.path.join(args.input_synthetic_dir, split, 'language.json')
        with open(synthetic_fn, 'r') as f:
            synthetic_data = json.load(f)
            task_names[split] = {name : [] for name in synthetic_data}
        print(f"Found n={len(task_names[split])} for split={split}")
    
        synthetic_vocab_fn = os.path.join(args.input_synthetic_dir, split, 'vocab.json')
        with open(synthetic_vocab_fn, 'r') as f:
            synthetic_vocab = json.load(f)
            vocab.update(synthetic_vocab)
    # Read in all of the raw human data
    with open(args.input_data_file, 'r') as f:
        human_data = json.load(f)
        human_data = {
            get_task_name(img_name) : human_data[img_name]
            for img_name in human_data
        }
    
    
    # Preprocess the human data.
    for split in ('train', 'test'):
        for task_name in task_names[split]:
            if task_name in human_data:
                print(f"Task name: {task_name}: n= {len(human_data[task_name])}")
                task_names[split][task_name] = [preprocess_text(s) for s in human_data[task_name]]
                for s in task_names[split][task_name]:
                    print(f"\t{s}")

        num_non_empty = len([t for t in task_names[split] if len(task_names[split][t]) > 0])
        print(f"Found n={num_non_empty}/{len(task_names[split])} for split={split}")
        
        # Get the vocabulary
        vocab.update(build_vocab(task_names[split]))
        print(f"Vocab is now: n=[{len(vocab)}]")
    
        # Write human data
        split_dir = os.path.join(args.output_dir, split)
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        out_fn = os.path.join(split_dir, "language.json")
        with open(out_fn, 'w') as f:
            json.dump(task_names[split], f)
        
    # Write the same vocab for both.
    for split in ('train', 'test'):
        split_dir = os.path.join(args.output_dir, split)
        out_fn = os.path.join(split_dir, "vocab.json")
        with open(out_fn, 'w') as f:
            json.dump(sorted(list(vocab)), f)
    
        
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)