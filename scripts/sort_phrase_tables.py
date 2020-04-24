phrase_table = '/Users/catwong/Desktop/zyzzyva/code/ec/experimentOutputs/logo/2020-04-21T15-41-24-093831/moses_corpus_12/model/phrase-table'


with open(phrase_table, 'r') as f:
    lines = [l.strip().split(' ||| ') for l in f.readlines()]

all_alignments = [(program, word, float(p)) for (program, word, p) in lines]
# Print top-ranked distributions.
global_sorted = sorted(all_alignments, key=lambda v: v[-1], reverse = True)
for (program, word, p) in global_sorted:
    print(f"p({program} | '{word}') = {p}")

from collections import defaultdict
alignments_per_word = defaultdict(list)
for (program, word, p) in global_sorted:
    alignments_per_word[word].append((program, p))

def max_p(word):
    return max([p for (program, p) in alignments_per_word[word]])
for word in sorted(alignments_per_word, key = max_p, reverse=True):
    for (program, p) in alignments_per_word[word]:
        print(f"p({program} | '{word}') = {p}")
    print("\n")
    
    