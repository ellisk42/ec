#### Statistical machine translation
import itertools
import gc
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from dreamcoder.utilities import *
import numpy as np

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
    corpora = defaultdict(list)
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
                # Also update our corpus store
                corpora["nl"].append([str(t) for t in nl])
                corpora["ml"].append([str(t) for t in ml])
                
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
        
    return count_dicts, encountered_nl, corpora

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
    
    # Count tokens that have not been aligned to anything.
    unencountered_nl = {
        token_type : {token : count for (token, count) in unsolved_nl[token_type].items() if token not in encountered_nl[token_type]} for token_type in encountered_nl
    }
    # Count tokens that have appeared in unsolved tasks.
    encountered_unsolved_nl = {
        token_type : {token : count for (token, count) in unsolved_nl[token_type].items() if token in encountered_nl[token_type]} for token_type in encountered_nl
    }
    unaligned_counts = {
        'unencountered' : unencountered_nl,
        'encountered' : encountered_unsolved_nl
    }
    eprint(f"Found n=[{len(unencountered_nl['readable'])}] distinct unencountered tokens that appear in n=[{len(unsolved_tasks)}] unsolved tasks, and n=[{len(encountered_unsolved_nl['readable'])} distinct encountered tokens.]")
    return unaligned_counts

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

def extract_align_indices(line):
    import re
    matches = re.findall(r'(.+?) \({(.+?)}\)', line)
    matches = [(t.strip(), a.strip().split()) for (t, a) in matches]
    matches = [(t, [int(align_index) - 1 for align_index in a]) for (t, a) in matches] # 1 indexed
    return matches

def merge_alignments(old, new):
    for k in new:
        if k not in old:
            old[k] = new[k]
        else:
            old[k] += new[k]
    return old

def merge_align_counts(old, new):
    for word in new:
        for k in new[word]:
            old[word][k] += new[word][k]
    return old

def alignments_for_sentence(source_tokens, align_indices, grammar):
    # Calculates alignments over the canonical tokens, unpacking primitives
    alignments = {}
    align_counts = defaultdict(lambda: defaultdict(float))
    for word, idxs in align_indices: # One word -> many indexes
        if word == 'NULL': continue
        word_primitive_counts = Counter() # Primitive -> # of appearances
        # For now we count the exact alignment, in sorted order, as a rule
        aligned_raw_tokens = [source_tokens[idx] for idx in idxs]
        align_key = (word, f"{word}||{'|'.join(aligned_raw_tokens)}", len(idxs))
        for aligned_raw in aligned_raw_tokens:
            primitive_counts = grammar.escaped_to_primitive_counts[aligned_raw]
            word_primitive_counts += primitive_counts
        if len(word_primitive_counts) > 0:
            # Add the full word <-> bag of primitives alignment
            alignments = merge_alignments(alignments, {
                align_key : word_primitive_counts
            })
            align_counts[word][align_key] += 1
    return alignments, align_counts

def get_alignments_and_scores(alignments, align_counts):
    total_alignments = sum (
        [sum(align_counts[word][k] for k in align_counts[word]) for word in align_counts]
    )
    alignments_scores = {}
    for word in align_counts:
        total_for_word = sum(align_counts[word][k] for k in align_counts[word])
        for k in align_counts[word]:
            (nl_word, align_key, num_prims) = k
            if num_prims < 2: continue # Only care about n > 1 primitives aligned to a word
            word_score = align_counts[word][k] / float(total_for_word)
            global_score = align_counts[word][k] / float(total_alignments)
            align_score = np.log(word_score) + np.log(global_score)
            alignments_scores[align_key] = (alignments[k], align_score)
    return alignments_scores
            
def get_alignments(grammar, output_dir=None):
    # Unzip the file
    align_file = "NONE"
    try:
        align_file = os.path.join(output_dir, "giza.ml-nl/ml-nl.A3.final.gz")
        unzip_cmd = f"gunzip {align_file}".split()
        subprocess.run(unzip_cmd, check=True)
    except:
        print(f"Can't gunzip {align_file}")
    
    try:
        align_file = os.path.join(output_dir, "giza.ml-nl/ml-nl.A3.final")
        # Extract alignments -- should be in (Comment, ML, NL form)
        all_alignments,  all_align_counts = {}, defaultdict(lambda: defaultdict(float)) # Word -> alignment -> count
        with open(align_file, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        for idx, line in enumerate(lines):
            if idx % 3 == 1:
                ml_tokens = line.strip().split()
                nl_to_indices = extract_align_indices(lines[idx+1])
                sentence_alignments, align_counts = alignments_for_sentence(ml_tokens, nl_to_indices, grammar)
                all_alignments = merge_alignments(all_alignments, sentence_alignments)
                all_align_counts = merge_align_counts(all_align_counts, align_counts)
        alignments_scores =  get_alignments_and_scores(all_alignments, all_align_counts)
        print("Sending the following alignments for compression")
        print_alignments(alignments_scores, top_n=20)
    except:
        print("Error finding alignments, returning empty alignments.")
        alignments_scores = {}
    return alignments_scores

def print_alignments(alignments_scores, top_n=20):
    for align_key in sorted(alignments_scores, key=lambda align_key: alignments_scores[align_key][-1], reverse=True):
        print(f"Alignment: {align_key} | Score: {alignments_scores[align_key][0]}")
        print(f"Bag of tokens: {alignments_scores[align_key][-1]}")
        print("\n")

def serialize_language_alignments(alignments_scores):
    if len(alignments_scores) == 0: return []
    serialized = [
        {
            'key': str(align_key),
            "score" : float(align_score),
            "primitive_counts" : [ {"prim": p, "count": c} for (p, c) in align_counts.items()]
        }
        for (align_key, (align_counts, align_score)) in alignments_scores.items()
    ]
    return serialized

def add_pseudoalignments(corpus_dir, n_pseudo, n_me, unaligned_counts, grammar, corpora, output_dir=None):
    eprint(f"Writing pseudoalignments with n_me = {n_me} and n_pseudo = {n_pseudo}")
    # Adds n_pseudo counts for all tokens and n_me counts for unused tokens.
    ml_vocab = [v for v in grammar.escaped_vocab if v is not 'VAR']
    align_file = os.path.join(output_dir, "model/aligned.grow-diag-final")
    global_used_ml = set() # All ML tokens currently aligned.
    token_alignments = defaultdict(set) # Tokens used for any one primitive:
    with open(align_file, 'r') as f: # Alignments are (ml, nl) locations in the corpus files.
        for idx, line in enumerate(f.readlines()):
            alignments = [[int(t) for t in a.split("-")] for a in line.split()]
            global_used_ml.update([corpora['ml'][idx][a[0]] for a in alignments])
            for a in alignments:
                token_alignments[corpora['nl'][idx][a[1]]].add(corpora['ml'][idx][a[0]])
    unused_ml = set(ml_vocab) - global_used_ml
    
    def get_pseudotokens(type, nl_token):
        if type == 'unencountered':
            me = unused_ml
        elif type == 'encountered':
            unused_for_nl = set(ml_vocab) - set(token_alignments[nl_token])
            me = unused_for_nl
        pseudotokens = ml_vocab
        return pseudotokens, me
    
    # We need to augment our corpus files as well as the alignment files.
    num_align = 0
    ml_corpus = ""
    nl_corpus = ""
    for type in 'unencountered', 'encountered':
        for nl_token in unaligned_counts[type]['readable']:
            pseudotokens, me = get_pseudotokens(type, nl_token)
            for _ in range(int(unaligned_counts[type]['readable'][nl_token] * n_pseudo)):
                for pseudotoken in pseudotokens:
                    nl_corpus +=f"{nl_token}\n"
                    ml_corpus +=f"{pseudotoken}\n"
                    num_align += 1
            for _ in range(int(unaligned_counts[type]['readable'][nl_token] * n_me)):
                for pseudotoken in me:
                    nl_corpus +=f"{nl_token}\n"
                    ml_corpus +=f"{pseudotoken}\n"
                    num_align += 1
    corpus_augment = {'nl' : nl_corpus, 'ml' : ml_corpus}
    # Write out the corpora.
    for corpus_type in ('nl', 'ml'):
        name = os.path.join(corpus_dir, f'corpus.{corpus_type}')
        with open(name, 'a+') as f:
            f.write(corpus_augment[corpus_type])
    
    pseudoalignments = "\n".join(["0-0"] * num_align) # Single token per line.
    with open(align_file, "a+") as f:
        f.write(pseudoalignments)
    
def moses_translation_tables(corpus_dir, moses_dir, output_dir=None, phrase_length=None):
    corpus = os.path.join(corpus_dir, "corpus")
    # Remove any lex files if they exist.
    for ext in ('e2f', 'f2e'):
        lex_loc = os.path.abspath(os.path.join(output_dir, 'model', f'lex.{ext}'))
        subprocess.run(["rm", lex_loc])
        
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

    lm_cmd = f"{lm_tool} -o {n_grams} --discount_fallback --text {all_task_corpus_loc} --arpa {lm_filename} --memory 64G"
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
    gc.collect()
    count_dicts, encountered_nl, corpora = write_sentence_aligned(tasks, frontiers, grammar, language_encoder, corpus_dir, moses_dir)
    unaligned_counts = count_unaligned_words(tasks, tasks_attempted, frontiers, grammar, language_encoder, encountered_nl)
    write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts)
    initial_ibm_alignment(corpus_dir, moses_dir, output_dir=output_dir, max_ibm_model=4)
    gc.collect()
    
    if n_pseudo > 0 and phrase_length == 1:
        add_pseudoalignments(corpus_dir, n_pseudo=n_pseudo, n_me=n_pseudo, unaligned_counts=unaligned_counts, grammar=grammar, corpora=corpora, output_dir=output_dir)
    phrase_table_loc = moses_translation_tables(corpus_dir, moses_dir, output_dir=output_dir)
    lm_config = train_natural_language_model(tasks, language_encoder, corpus_dir, moses_dir, output_dir=output_dir, n_grams=3)
    generate_decoder_config(corpus_dir, moses_dir, output_dir=output_dir, lm_config=lm_config, phrase_table=phrase_table_loc, phrase_length=phrase_length)
    print("Finished translation, returning.")
    # Return the appropriate table locations, or read into memory.
    return {
        "corpus_dir" : corpus_dir,
        "moses_dir" : moses_dir,
        "output_dir" : output_dir,
    }
    
    
    