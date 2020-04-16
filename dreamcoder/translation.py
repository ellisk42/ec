#### Statistical machine translation
import itertools
import os
import subprocess
from collections import Counter
from pathlib import Path
from dreamcoder.utilities import *


MOSES_SCRIPT = 'moses/scripts/training/train-model.perl'
MOSES_SCRIPT_OLD = 'moses/scripts/training/train-model-2-1.perl'
MOSES_TOOLS = 'bin/training-tools'

def frontier_to_tokens(frontier, grammar):
    """:ret List of numeric token lists for each program 'sentence'."""
    human_readable = [entry.tokens for entry in frontier.entries]
    numeric = [[grammar.vocab[t] for t in tokens] for tokens in human_readable]
    return human_readable, numeric

def write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts):
    eprint(f"Writing ML and NL vocabs to {corpus_dir}.")
    ml_vocab = grammar.vocab.items()
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
    ml_vocab = grammar.vocab.values()
    align_file = os.path.join(output_dir, "model/aligned.grow-diag-final")
    with open(align_file, "a+") as f:
        f.write(pseudoalignments)

def moses_translation_tables(corpus_dir, moses_dir, output_dir=None, phrase_length=None):
    corpus = os.path.join(corpus_dir, "corpus")
    # Build Moses command for compiling translation tables and phrase tables.
    # Note that foreign = ml and english = nl
    moses_script = os.path.join(moses_dir, MOSES_SCRIPT)
    tools_loc = os.path.join(moses_dir, MOSES_TOOLS)
    moses_cmd = f"{moses_script} --f ml --e nl --mgiza --root-dir {output_dir} --external-bin-dir {tools_loc} --corpus-dir {corpus_dir} --corpus {corpus} --do-steps 4 --max-phrase-length {phrase_length} --no-lexical-weighting".split()
    subprocess.run(moses_cmd, check=True)
    
    # Try max 
    phrase_length=7
    moses_script_old = os.path.join(moses_dir, MOSES_SCRIPT_OLD)
    eprint("Building phrase translation tables....")
    moses_cmd = f"{moses_script_old} --f ml --e nl --mgiza --root-dir {output_dir} --external-bin-dir {tools_loc} --corpus-dir {corpus_dir} --corpus {corpus} --do-steps 5 --max-phrase-length {phrase_length} --no-lexical-weighting".split()
    subprocess.run(moses_cmd, check=True)
    # If all else fails: manually write phrase tables?

    
def smt_alignment(tasks, tasks_attempted, frontiers, grammar, language_encoder, output_prefix, moses_dir, n_pseudo=0, output_dir=None, phrase_length=1):
    corpus_dir = os.path.join(output_prefix, "corpus_tmp")
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    if output_dir is None: output_dir = corpus_dir
    
    count_dicts, encountered_nl = write_sentence_aligned(tasks, frontiers, grammar, language_encoder, corpus_dir, moses_dir)
    unaligned_counts = count_unaligned_words(tasks, tasks_attempted, frontiers, grammar, language_encoder, encountered_nl)
    write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts)
    initial_ibm_alignment(corpus_dir, moses_dir, output_dir=output_dir, max_ibm_model=4)
    
    if n_pseudo > 0:
        add_pseudoalignments(corpus_dir, n_pseudo, unaligned_counts, grammar, output_dir=None)
    moses_translation_tables(corpus_dir, moses_dir, output_dir=output_dir, phrase_length=phrase_length)

    assert False
    # Return the appropriate table locations, or read into memory.
    