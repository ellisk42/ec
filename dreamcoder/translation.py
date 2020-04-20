#### Statistical machine translation
import itertools
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from dreamcoder.utilities import *

# Relative to moses_dir
MOSES_SCRIPT = 'scripts/training/train-model.perl'
MOSES_TOOLS = 'training-tools'
LM_TOOL = 'bin/lmplz' 
DECODER_TOOL = "bin/moses"

def frontier_to_tokens(frontier, grammar):
    """:ret List of numeric token lists for each program 'sentence'."""
    human_readable = [grammar.escape_tokens(entry.tokens) for entry in frontier.entries]
    numeric = [[grammar.escaped_vocab[t] for t in grammar.escape_tokens(tokens)] for tokens in human_readable]
    return human_readable, numeric

def write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts):
    eprint(f"Writing ML and NL vocabs to {corpus_dir}.")
    ml_vocab = grammar.escaped_vocab.items()
    nl_vocab = language_encoder.symbolToIndex.items()
    
    # Format: INDEX      TOKEN     COUNT    
    for (vocab, type) in zip([ml_vocab, nl_vocab], ["ml", "nl"]):
        with open(os.path.join(corpus_dir, f'{type}.vcb'), 'w') as f:
            sorted_vocab = sorted(list(vocab), key=lambda k: k[-1])
            for line, (word, idx) in enumerate(sorted_vocab):
                count = count_dicts[type][word]
                f.write(f"{idx}\t{word}\t{count}\n")
                
def write_sentence_aligned(tasks, frontiers, grammar, language_encoder, corpus_dir, moses_dir):
    print(f"Writing ML and NL corpora to {corpus_dir}.")
    human_readable = ""
    nl_ml_numeric_bitext = ""
    ml_nl_numeric_bitext = ""
    ml_corpus = ""
    nl_corpus = ""
    count_dicts = {"ml": Counter(), "nl" : Counter()}
    encountered_nl = {"readable": set(), "numeric": set()} # NL words we have encountered in the solved tasks.

    for f in frontiers:
        if f.task in tasks and not f.empty:
            nl_readable, nl_numeric = language_encoder.tokenize_for_smt(f.task)
            ml_readable, ml_numeric = frontier_to_tokens(f, grammar)
            count_dicts["ml"].update(itertools.chain.from_iterable(ml_readable))
            count_dicts["nl"].update(itertools.chain.from_iterable(nl_readable))
            encountered_nl["readable"].update(itertools.chain.from_iterable(nl_readable))
            encountered_nl["numeric"].update(itertools.chain.from_iterable(nl_numeric))
            
            nl = "\n".join([" ".join(t) for t in nl_readable])
            ml = "\n".join([" ".join(t) for t in ml_readable])
            human_readable += f"###{f.task.name}\n{nl}\n{ml}\n"
            for (nl, ml) in itertools.product(nl_numeric, ml_numeric):
                nl = " ".join([str(t) for t in nl])
                ml = " ".join([str(t) for t in ml])
                nl_ml_numeric_bitext += f"1\n{ml}\n{nl}\n"
                ml_nl_numeric_bitext += f"1\n{nl}\n{ml}\n"
            # Redundant! See if we can remove this.
            for (nl, ml) in itertools.product(nl_readable, ml_readable):
                nl = " ".join([str(t) for t in nl])
                ml = " ".join([str(t) for t in ml])
                nl_corpus += f"{nl}\n"
                ml_corpus += f"{ml}\n"
        
    with open(os.path.join(corpus_dir, 'nl-ml-readable-train.snt'), 'w') as f:
        f.write(human_readable)
    with open(os.path.join(corpus_dir, 'nl-ml-int-train.snt'), 'w') as f:
        f.write(nl_ml_numeric_bitext)
    with open(os.path.join(corpus_dir, 'ml-nl-int-train.snt'), 'w') as f:
        f.write(ml_nl_numeric_bitext)
    
    # Write out corpora
    corpora_names = []
    for type, corpus in zip(('nl', 'ml'), (nl_corpus, ml_corpus)):
        name = os.path.join(corpus_dir, f'corpus.{type}')
        corpora_names += [(type, name)]
        with open(name, 'w') as f:
            f.write(corpus)
    
    eprint("Building word classes using Moses mkcls.")
    for type, corpus in corpora_names:
        mkcls_loc = os.path.join(moses_dir, MOSES_TOOLS, "mkcls")
        output_loc = os.path.join(corpus_dir, f"{type}.vcb.classes")
        mkcls_cmd = [f"{mkcls_loc}", "-c50", "-n2", f"-p{corpus}",  f"-V{output_loc}"]
        subprocess.run(mkcls_cmd, check=True)
        
    return count_dicts, encountered_nl

def count_unaligned_words(tasks, tasks_attempted, frontiers, grammar, language_encoder, encountered_nl):
    # Dirty hack! If we forgot to track what tasks we actually attempted, then look at the whole training set.
    if len(tasks_attempted) == 0: 
        eprint("[Warning]: setting the attempted tasks to the training set.")
        tasks_attempted = tasks
    # Count words that appear in tasks we have not solved so far.
    unsolved_nl = {"readable": Counter(), "numeric": Counter()}
    solved_tasks = set([f.task for f in frontiers if f])
    unsolved_tasks = set(tasks_attempted).difference(solved_tasks)
    for task in unsolved_tasks:
        nl_readable, nl_numeric = language_encoder.tokenize_for_smt(task)
        unsolved_nl["readable"].update(itertools.chain.from_iterable(nl_readable))
        unsolved_nl["numeric"].update(itertools.chain.from_iterable(nl_numeric))
    # Only use tokens we haven't seen.

    unencountered_nl = {
        token_type : {token : count for (token, count) in unsolved_nl[token_type].items() if token not in encountered_nl[token_type]} for token_type in encountered_nl
    }
    eprint(f"Found n=[{len(unencountered_nl['readable'])}] distinct unencountered tokens that appear in n=[{len(unsolved_tasks)}] unsolved tasks.")
    return unencountered_nl

def initial_ibm_alignment(corpus_dir, moses_dir, output_dir=None, max_ibm_model=4):
    eprint(f"Running IBM alignment using model {max_ibm_model}")
    if max_ibm_model != 4: 
        raise Exception('Unimplemented: IBM alignment with model != 4')
    # Remove giza files if already there. TODO: probably want to cache this in outputs instead.
    print("Removing any existing GIZA files from previous translation runs.")
    for giza_out in ("giza.ml-nl", "giza.nl-ml"):
        out_loc = os.path.join(output_dir, giza_out)
        subprocess.run(["rm", "-r", out_loc])
    
    # Build Moses command for joint alignment.
    # Note that foreign = ml and english = nl
    moses_script = os.path.join(moses_dir, MOSES_SCRIPT)
    tools_loc = os.path.join(moses_dir, MOSES_TOOLS)
    moses_cmd = f"{moses_script} --f ml --e nl --mgiza --external-bin-dir {tools_loc} --corpus-dir {corpus_dir} --first-step 2 --last-step 3 --root-dir {output_dir}".split()
    subprocess.run(moses_cmd, check=True)

def add_pseudoalignments(corpus_dir, n_pseudo, unaligned_counts, grammar, output_dir=None):
    # TODO: add pseudoalignments for the rebalancing case, or define how to handle OOV.
    raise Exception('Unimplemented: Pseudo alignments.')
    ml_vocab = grammar.escaped_vocab.values()
    align_file = os.path.join(output_dir, "model/aligned.grow-diag-final")
    with open(align_file, "a+") as f:
        f.write(pseudoalignments)

def moses_translation_tables(corpus_dir, moses_dir, output_dir=None, phrase_length=None):
    corpus = os.path.join(corpus_dir, "corpus")
    # Build Moses command for compiling translation tables and phrase tables.
    # Note that foreign = ml and english = nl
    moses_script = os.path.join(moses_dir, MOSES_SCRIPT)
    tools_loc = os.path.join(moses_dir, MOSES_TOOLS)
    moses_cmd = f"{moses_script} --f ml --e nl --mgiza --root-dir {output_dir} --external-bin-dir {tools_loc} --corpus-dir {corpus_dir} --corpus {corpus} --do-steps 4 --no-lexical-weighting".split()
    subprocess.run(moses_cmd, check=True)
    
    # IBM Model 1 lexical decoding -- directly convert this into a phrase table.
    translation_table_loc = os.path.join(output_dir, 'model', 'lex.e2f')
    phrase_table_loc = os.path.abspath(os.path.join(output_dir, 'model', 'phrase-table'))
    with open(phrase_table_loc, 'w') as f:
        delimiter_cmd = ["sed", "s/ / ||| /g", f"{translation_table_loc}"]
        subprocess.run(delimiter_cmd, check=True, stdout=f)
    return phrase_table_loc

def train_natural_language_model(tasks, language_encoder, corpus_dir, moses_dir, output_dir=None, n_grams=3):
    # Write the full set of task to train the language model.
    nl_all_task_corpus = ""
    for task in tasks:
        nl_readable, nl_numeric = language_encoder.tokenize_for_smt(task)
        nl = "\n".join([" ".join(t) for t in nl_readable])
        nl_all_task_corpus += f"{nl}\n"
    all_task_corpus_loc = os.path.join(corpus_dir, 'all_task_corpus.nl')
    with open(all_task_corpus_loc, 'w') as f:
        f.write(nl_all_task_corpus)
    
    # Train the language model on the full training data.
    lm_filename = os.path.abspath(os.path.join(output_dir, f"lm-nl-o-{n_grams}.arpa"))
    lm_tool = os.path.join(moses_dir, LM_TOOL)

    lm_cmd = f"{lm_tool} -o {n_grams} --discount_fallback --text {all_task_corpus_loc} --arpa {lm_filename}"
    subprocess.check_output(lm_cmd.split())
    return {"factor" : 0,
            "filename": lm_filename,
            "order" : n_grams}
            
def generate_decoder_config(corpus_dir, moses_dir, output_dir, lm_config, phrase_table, phrase_length):
    corpus = os.path.join(corpus_dir, "corpus")
    lm_cmd = f"{lm_config['factor']}:{lm_config['order']}:{lm_config['filename']}"
    moses_script = os.path.join(moses_dir, MOSES_SCRIPT)
    tools_loc = os.path.join(moses_dir, MOSES_TOOLS)
    if phrase_length > 1:
        eprint(f"Training Moses phrase table with max_length = {phrase_length}")
        moses_cmd = f"{moses_script} --f ml --e nl --mgiza --root-dir {output_dir} --external-bin-dir {tools_loc} --corpus-dir {corpus_dir} --corpus {corpus} --lm {lm_cmd} --first-step 5 --last-step 9 --max-phrase-length {phrase_length}".split()
        subprocess.run(moses_cmd, check=True)
    else:
        eprint(f"Manually writing Moses file with phrase max_length = {phrase_length}")
        decoder_config = os.path.abspath(os.path.join(corpus_dir, "model", "moses.ini"))
        lm_loc = os.path.abspath(lm_config['filename'])
        # Write the moses.ini file directly 
        moses_config_text = f"[input-factors]\n0\n[mapping]\n0 T 0\n[distortion-limit]\n6\n"
        moses_config_text += f"[feature]\nUnknownWordPenalty\nWordPenalty\nPhrasePenalty\nDistortion\n"
        moses_config_text += f"PhraseDictionaryMemory name=TranslationModel0 num-features=1 path={phrase_table} input-factor=0 output-factor=0\n"
        moses_config_text += f"KENLM name=LM0 factor=0 path={lm_loc} order=3\n"
        # Weights -- considering retuning.
        moses_config_text += "[weight]\nUnknownWordPenalty0= 1\nWordPenalty0= -1\nPhrasePenalty0= 0.0\nTranslationModel0= 0.2\nDistortion0= 0.3\nLM0= 0.5\n"
        with open(decoder_config, 'w') as f:
            f.write(moses_config_text)

def translate_frontiers_to_nl(frontiers, grammar, translation_info, n_best, verbose):
    # Frontiers contain multiple programs
    if len(frontiers) < 1: return []
    program_idx_to_task = {}
    program_tokens = []
    isHelmholtz = frontiers[0].__class__.__name__ == 'HelmholtzEntry' 
    idx = 0
    for f in frontiers:
        if isHelmholtz:
            for t in f.program_tokens:
                program_idx_to_task[idx] = f.task
                program_tokens.append(t)
                idx += 1
        else:
            for e in f.entries:
                program_idx_to_task[idx] = f.task
                program_tokens.append(e.tokens)
                idx += 1
    idx_to_translations = decode_programs(program_tokens, grammar, translation_info["output_dir"], "helmholtz_translations",  translation_info['corpus_dir'], translation_info['moses_dir'], n_best)
    task_to_tokens = defaultdict(list)
    for idx in idx_to_translations:
        task_to_tokens[program_idx_to_task[idx]] += idx_to_translations[idx]
    return task_to_tokens

def decode_programs(program_tokens,  grammar, output_dir, output_suffix, corpus_dir, moses_dir, n_best):
    """:ret: idx_to_translations: {index of program : [list of translation tokens]}"""
    eprint(f"Translating n=[{len(program_tokens)}] using n_best={n_best}")
    tmp_program_file = os.path.abspath(os.path.join(output_dir, 'tmp_program_file'))
    with open(tmp_program_file, 'w') as f:
        for tokens in program_tokens:
            for t in tokens:
                if t not in grammar.escaped_vocab:
                    import pdb; pdb.set_trace()
            f.write(' '.join(grammar.escape_tokens(tokens)) + "\n")

    corpus = os.path.join(corpus_dir, "corpus")
    tmp_nbest_file = os.path.abspath(os.path.join(output_dir, 'tmp_nbest_file'))
    decoder_config = os.path.abspath(os.path.join(corpus_dir, "model", "moses.ini"))
    decoder_loc = os.path.join(moses_dir, DECODER_TOOL)
    decoder_cmd = f"{decoder_loc} --config {decoder_config} --input-file {tmp_program_file} "
    decoder_cmd += f"--drop-unknown --phrase-drop-allowed -v 0 --n-best-list {tmp_nbest_file} {n_best}"
 
    subprocess.check_output(decoder_cmd.split()) # Suppress output

    delimiter = ' ||| '
    with open(tmp_nbest_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if delimiter in l]
    idx_to_translations = defaultdict(list)
    for l in lines:
        split_l = l.split(delimiter)
        idx, tokens = int(split_l[0]), split_l[1].split()
        idx_to_translations[idx].append(tokens)
        
    if output_suffix is not None:
        output = os.path.join(output_dir, output_suffix)
        mode = 'a+' if os.path.exists(output) else 'w+'
        with open(output, mode) as f:
            for idx, p in enumerate(program_tokens):
                f.write(f"{idx} ||| {' '.join(p)}\n")
                for tokens in idx_to_translations[idx]:
                    f.write(f"{idx} ||| {' '.join(tokens)}\n")
    os.remove(tmp_nbest_file)
    os.remove(tmp_program_file)
    return idx_to_translations

def smt_alignment(tasks, tasks_attempted, frontiers, grammar, language_encoder, corpus_dir, moses_dir, n_pseudo=0, output_dir=None, phrase_length=1):
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    if output_dir is None: output_dir = corpus_dir
    
    count_dicts, encountered_nl = write_sentence_aligned(tasks, frontiers, grammar, language_encoder, corpus_dir, moses_dir)
    unaligned_counts = count_unaligned_words(tasks, tasks_attempted, frontiers, grammar, language_encoder, encountered_nl)
    write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts)
    initial_ibm_alignment(corpus_dir, moses_dir, output_dir=output_dir, max_ibm_model=4)
    if n_pseudo > 0:
        add_pseudoalignments(corpus_dir, n_pseudo, unaligned_counts, grammar, output_dir=None)
    phrase_table_loc = moses_translation_tables(corpus_dir, moses_dir, output_dir=output_dir)
    lm_config = train_natural_language_model(tasks, language_encoder, corpus_dir, moses_dir, output_dir=output_dir, n_grams=3)
    generate_decoder_config(corpus_dir, moses_dir, output_dir=output_dir, lm_config=lm_config, phrase_table=phrase_table_loc, phrase_length=phrase_length)

    # Return the appropriate table locations, or read into memory.
    return {
        "corpus_dir" : corpus_dir,
        "moses_dir" : moses_dir,
        "output_dir" : output_dir
    }
    
    