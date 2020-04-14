#### Statistical machine translation
def frontier_to_tokens(frontier, grammar):
    """:ret List of numeric token lists for each program 'sentence'."""
    return [
            [grammar.vocab[token] for token in entry.tokens]
            for entry in frontier.entries
            ]


def smt_alignment(tasks, frontiers, grammar, language_encoder):
    frontier_tokens = {
        f.task : frontier_to_tokens(f, grammar)
        for f in frontiers
    }
    language_tokens = {
        task: language_encoder.numeric_tokenize(task)
        for task in tasks
    }
    