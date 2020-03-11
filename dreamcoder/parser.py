from dreamcoder.grammar import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
import num2words

from sklearn.feature_extraction import DictVectorizer

class NgramFeaturizer():
    """
    Tokenizes and extracts n-gram and skip-gram features from text.
    n: maximum length of any n-grams, including skips.
    skip_n: maximum skip distance, which will be calculated on bigrams only.
    Default is: unigrams, bigrams, trigrams, skip-trigrams.
    """
    def __init__(self, max_n=3, skip_n=1, canonicalize_numbers=True, tokenizer_fn=None):
        self.max_n = max_n
        self.skip_n = skip_n
        self.canonicalize_numbers = True
        self.tokenizer_fn = tokenizer_fn
        if self.tokenizer_fn is None:
            from nltk.tokenize import word_tokenize
            self.tokenizer_fn = word_tokenize
    
    def canonicalize_number_token(self, t):
        from num2words import num2words
        try:
            t = num2words(int(t), to='cardinal')
        except:
            pass
        return t 
    
    def ngrams_skipgrams(self, tokens):
        features = list()
        
        for n in range(1, self.max_n+1):
            ngram_tokens = tokens
            for i in range(len(ngram_tokens) - (n-1)):
                ngram = tuple(ngram_tokens[i:i+n])
                features.append(ngram)
            if n==2:
                for i in range(len(tokens) - (n+1)):
                    skip = tuple([tokens[i]]+['*']*self.skip_n+[tokens[i+self.skip_n+1]])
                    features.append(skip)
            
        return features
    
    def featurize(self, sentence):
        """:ret: Counter containing ngram and skipgram features."""
        from collections import Counter
        tokens = self.tokenizer_fn(sentence)
        if self.canonicalize_numbers:
            tokens = [self.canonicalize_number_token(t) for t in tokens]
        features = Counter(self.ngrams_skipgrams(tokens))
        return features
    
    def featurize_sentences(self, sentences):
        """:ret: set containing ngram and skipgram features."""
        from collections import Counter
        features = Counter()
        for s in sentences:
            features += self.featurize(s) 
        return features
        
class LogLinearTreegramParser():
    """
    Log-linear tree-gram prediction model.
    Predicts P(z | language) by indirectly learning a log-linear mapping from language features to  program treegrams, allowing for fast forward-enumeration of programs | language.
    """
    
    def __init__(self, grammar, 
                language_data,
                frontiers,
                language_featurizer):
        self.language_data = language_data
        self.featurized_language_data = None
        self.frontiers = frontiers
        self.frontier_likelihood_summaries = None
        self.language_featurizer = language_featurizer
        if self.language_featurizer is None:
            self.language_featurizer = NgramFeaturizer()
        self.contextual_grammar = None
        self.language_dict_vectorizer = None
        
        # TODO: just initialize a contextual recognition model to calculate the likelihoods?
        # TODO: consider if we can just hijack more of the recognition model
        # (including training) with just a different feature extractor middle.
        self.grammar = ContextualGrammar.fromGrammar(grammar)
        self.program_feature_library = None, None
    
    def fit_language_features(self):
        """
        Featurizes self.language_data and fits the dict vectorizer.
        
        self.language_dict_vectorizer: dict vectorizer over self.language_data
        self.featurized_language_data: {t : sparse feature vector}
        """
        sorted_tasks = sorted(self.language_data.keys(), key=lambda t:t.name)
        task_features = [self.language_featurizer.featurize_sentences(self.language_data[t]) for t in sorted_tasks]
        
        self.language_dict_vectorizer = DictVectorizer()
        vectorized_features = self.language_dict_vectorizer.fit_transform(task_features)
        self.featurized_language_data = {
            t : vectorized_features[i]
            for (i, t) in enumerate(sorted_tasks)
        }
    
    def replaceProgramsWithLikelihoodSummaries(self, frontier):
        return Frontier(
            [FrontierEntry(
                program=self.grammar.closedLikelihoodSummary(frontier.task.request, e.program),
                logLikelihood=e.logLikelihood,
                logPrior=e.logPrior) for e in frontier],
            task=frontier.task)
            
    def fit_program_features(self):
        # Calculate likelihood summaries.
        self.frontier_likelihood_summaries = [self.replaceProgramsWithLikelihoodSummaries(f).normalize()
                     for f in self.frontiers]
                     
        # Initialize the 
        
        # Replace solutions with their likelihood summaries.
        # TODO: use the ContextualGrammarNetwork to produce the batched / vectorized log likelihoods for a set of programs.
    
        
    def train(self):
        # Featurize the language for every task.
        self.fit_language_features()
        import pdb; pdb.set_trace()
        self.fit_program_features()
        
        # Fit a linear 
    
    def enumerate(self):
        """:ret: bigram grammar for contextual enumeration."""
        pass