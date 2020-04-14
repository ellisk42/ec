#### Statistical machine translation
import itertools
import os
import subprocess
from pathlib import Path

MOSES_SCRIPT = 'moses/scripts/training/train-model.perl'
MOSES_TOOLS = 'bin/training-tools'

def frontier_to_tokens(frontier, grammar):
    """:ret List of numeric token lists for each program 'sentence'."""
    human_readable = [entry.tokens for entry in frontier.entries]
    numeric = [[grammar.vocab[t] for t in tokens] for tokens in human_readable]
    return human_readable, numeric

def write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts):
    print(f"Writing ML and NL vocabs to {corpus_dir}.")
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
    for f in frontiers:
        if f.task in tasks and not f.empty:
            nl_readable, nl_numeric = language_encoder.tokenize_for_smt(f.task)
            ml_readable, ml_numeric = frontier_to_tokens(f, grammar)
            
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

def run_smt_alignment(corpus_dir, moses_dir, output_dir=None):
    eprint("Running SMT alignment using GIZA")
    # Remove giza files if already there. TODO: probably want to cache this in outputs instead.
    if output_dir is None: output_dir = corpus_dir
    print("Removing GIZA files if already there.")
    for giza_out in ("giza.ml-nl", "giza.nl-ml"):
        out_loc = os.path.join(output_dir, giza_out)
        subprocess.run(["rm", "-r", out_loc])
    # Build the overly long Moses command
    moses_script = os.path.join(moses_dir, MOSES_SCRIPT)
    tools_loc = os.path.join(moses_dir, MOSES_TOOLS)
    moses_cmd = f"{moses_script} --f ml --e nl --mgiza --external-bin-dir {tools_loc} --corpus-dir {corpus_dir} --first-step 2 --last-step 2 --root-dir {output_dir}".split()
    
    subprocess.run(moses_cmd, check=True)
    

def smt_alignment(tasks, frontiers, grammar, language_encoder, output_prefix, moses_dir):
    corpus_dir = os.path.join(output_prefix, "corpus_tmp")
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    
    count_dicts = write_sentence_aligned(tasks, frontiers, grammar, language_encoder, corpus_dir, moses_dir)
    write_smt_vocab(grammar, language_encoder, corpus_dir, count_dicts)
    import pdb; pdb.set_trace()
    
    run_smt_alignment(corpus_dir, moses_dir, output_dir=None)
    
    
    
    