from dreamcoder.enumeration import *
from dreamcoder.grammar import *
# luke


import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
# luke
import json


def variable(x, volatile=False, cuda=False):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def maybe_cuda(x, use_cuda):
    if use_cuda:
        return x.cuda()
    else:
        return x


def is_torch_not_a_number(v):
    """checks whether a tortured variable is nan"""
    v = v.data
    if not ((v == v).item()):
        return True
    return False

def is_torch_invalid(v):
    """checks whether a torch variable is nan or inf"""
    if is_torch_not_a_number(v):
        return True
    a = v - v
    if is_torch_not_a_number(a):
        return True
    return False

def _relu(x): return x.clamp(min=0)

class Entropy(nn.Module):
    """Calculates the entropy of logits"""
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=0) * F.log_softmax(x, dim=0)
        b = -1.0 * b.sum()
        return b

class GrammarNetwork(nn.Module):
    """Neural network that outputs a grammar"""
    def __init__(self, inputDimensionality, grammar):
        super(GrammarNetwork, self).__init__()
        self.logProductions = nn.Linear(inputDimensionality, len(grammar)+1)
        self.grammar = grammar
        
    def forward(self, x):
        """Takes as input inputDimensionality-dimensional vector and returns Grammar
        Tensor-valued probabilities"""
        logProductions = self.logProductions(x)
        return Grammar(logProductions[-1].view(1), #logVariable
                       [(logProductions[k].view(1), t, program)
                        for k, (_, t, program) in enumerate(self.grammar.productions)],
                       continuationType=self.grammar.continuationType)

    def batchedLogLikelihoods(self, xs, summaries):
        """Takes as input BxinputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary"""
        use_cuda = xs.device.type == 'cuda'

        B = xs.size(0)
        assert len(summaries) == B
        logProductions = self.logProductions(xs)

        # uses[b][p] is # uses of primitive p by summary b
        uses = np.zeros((B,len(self.grammar) + 1))
        for b,summary in enumerate(summaries):
            for p, production in enumerate(self.grammar.primitives):
                uses[b,p] = summary.uses.get(production, 0.)
            uses[b,len(self.grammar)] = summary.uses.get(Index(0), 0)

        numerator = (logProductions * maybe_cuda(torch.from_numpy(uses).float(),use_cuda)).sum(1)
        numerator += maybe_cuda(torch.tensor([summary.constant for summary in summaries ]).float(), use_cuda)

        alternativeSet = {normalizer
                          for s in summaries
                          for normalizer in s.normalizers }
        alternativeSet = list(alternativeSet)

        mask = np.zeros((len(alternativeSet), len(self.grammar) + 1))
        for tau in range(len(alternativeSet)):
            for p, production in enumerate(self.grammar.primitives):
                mask[tau,p] = 0. if production in alternativeSet[tau] else NEGATIVEINFINITY
            mask[tau,len(self.grammar)] = 0. if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
        mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

        # mask: Rx|G|
        # logProductions: Bx|G|
        # Want: mask + logProductions : BxRx|G| = z
        z = mask.repeat(B,1,1) + logProductions.repeat(len(alternativeSet),1,1).transpose(1,0)
        # z: BxR
        z = torch.logsumexp(z, 2) # pytorch 1.0 dependency

        # Calculate how many times each normalizer was used
        N = np.zeros((B, len(alternativeSet)))
        for b, summary in enumerate(summaries):
            for tau, alternatives in enumerate(alternativeSet):
                N[b, tau] = summary.normalizers.get(alternatives,0.)

        denominator = (maybe_cuda(torch.tensor(N).float(),use_cuda) * z).sum(1)
        return numerator - denominator

        

class ContextualGrammarNetwork_LowRank(nn.Module):
    def __init__(self, inputDimensionality, grammar, R=16):
        """Low-rank approximation to bigram model. Parameters is linear in number of primitives.
        R: maximum rank"""
        
        super(ContextualGrammarNetwork_LowRank, self).__init__()

        self.grammar = grammar

        self.R = R # embedding size

        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        self.n_grammars = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(self.n_grammars, self.n_grammars+numberOfArguments))
            self.library[prim] = idx_list
            self.n_grammars += numberOfArguments
        
        # We had an extra grammar for when there is no parent and for when the parent is a variable
        self.n_grammars += 2
        self.transitionMatrix = LowRank(inputDimensionality, self.n_grammars, len(grammar) + 1, R)
        
    def grammarFromVector(self, logProductions):
        return Grammar(logProductions[-1].view(1),
                       [(logProductions[k].view(1), t, program)
                        for k, (_, t, program) in enumerate(self.grammar.productions)],
                       continuationType=self.grammar.continuationType)

    def forward(self, x):
        assert len(x.size()) == 1, "contextual grammar doesn't currently support batching"

        transitionMatrix = self.transitionMatrix(x)
        
        return ContextualGrammar(self.grammarFromVector(transitionMatrix[-1]), self.grammarFromVector(transitionMatrix[-2]),
                {prim: [self.grammarFromVector(transitionMatrix[j]) for j in js]
                 for prim, js in self.library.items()} )
        
    def vectorizedLogLikelihoods(self, x, summaries):
        B = len(summaries)
        G = len(self.grammar) + 1

        # Which column of the transition matrix corresponds to which primitive
        primitiveColumn = {p: c
                           for c, (_1,_2,p) in enumerate(self.grammar.productions) }
        primitiveColumn[Index(0)] = G - 1
        # Which row of the transition matrix corresponds to which context
        contextRow = {(parent, index): r
                      for parent, indices in self.library.items()
                      for index, r in enumerate(indices) }
        contextRow[(None,None)] = self.n_grammars - 1
        contextRow[(Index(0),None)] = self.n_grammars - 2

        transitionMatrix = self.transitionMatrix(x)

        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B,self.n_grammars,len(self.grammar)+1))
        for b,summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g,s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b,g,p] = s.uses.get(production, 0.)
                    uses[b,g,len(self.grammar)] = s.uses.get(Index(0), 0)
                    
            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(Index(0), 0.)

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(Index(0), 0.)

        uses = maybe_cuda(torch.tensor(uses).float(),use_cuda)
        numerator = uses.view(B, -1) @ transitionMatrix.view(-1)
        
        constant = np.zeros(B)
        for b,summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant
            
        numerator = numerator + maybe_cuda(torch.tensor(constant).float(),use_cuda)

        # Calculate the god-awful denominator
        # Map from (parent, index, {set-of-alternatives}) to [occurrences-in-summary-zero, occurrences-in-summary-one, ...]
        alternativeSet = {}
        for b,summary in enumerate(summaries):
            for normalizer, frequency in summary.noParent.normalizers.items():
                k = (None,None,normalizer)
                alternativeSet[k] = alternativeSet.get(k, np.zeros(B))
                alternativeSet[k][b] += frequency
            for normalizer, frequency in summary.variableParent.normalizers.items():
                k = (Index(0),None,normalizer)
                alternativeSet[k] = alternativeSet.get(k, np.zeros(B))
                alternativeSet[k][b] += frequency
            for parent, ss in summary.library.items():
                for argumentIndex, s in enumerate(ss):
                    for normalizer, frequency in s.normalizers.items():
                        k = (parent, argumentIndex, normalizer)
                        alternativeSet[k] = alternativeSet.get(k, zeros(B))
                        alternativeSet[k][b] += frequency

        # Calculate each distinct normalizing constant
        alternativeNormalizer = {}
        for parent, index, alternatives in alternativeSet:
            r = transitionMatrix[contextRow[(parent, index)]]
            entries = r[ [primitiveColumn[alternative] for alternative in alternatives ]]
            alternativeNormalizer[(parent, index, alternatives)] = torch.logsumexp(entries, dim=0)

        # Concatenate the normalizers into a vector
        normalizerKeys = list(alternativeSet.keys())
        normalizerVector = torch.cat([ alternativeNormalizer[k] for k in normalizerKeys])

        assert False, "This function is still in progress."
        

    def batchedLogLikelihoods(self, xs, summaries):
        """Takes as input BxinputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary"""
        use_cuda = xs.device.type == 'cuda'
        
        B = xs.shape[0]
        G = len(self.grammar) + 1
        assert len(summaries) == B

        # logProductions: Bx n_grammars x G
        logProductions = self.transitionMatrix(xs)
        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B,self.n_grammars,len(self.grammar)+1))
        for b,summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g,s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b,g,p] = s.uses.get(production, 0.)
                    uses[b,g,len(self.grammar)] = s.uses.get(Index(0), 0)
                    
            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(Index(0), 0.)

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(Index(0), 0.)
            
        numerator = (logProductions*maybe_cuda(torch.tensor(uses).float(),use_cuda)).view(B,-1).sum(1)

        constant = np.zeros(B)
        for b,summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant
            
        numerator += maybe_cuda(torch.tensor(constant).float(),use_cuda)
        
        if True:

            # Calculate the god-awful denominator
            alternativeSet = set()
            for summary in summaries:
                for normalizer in summary.noParent.normalizers: alternativeSet.add(normalizer)
                for normalizer in summary.variableParent.normalizers: alternativeSet.add(normalizer)
                for ss in summary.library.values():
                    for s in ss:
                        for normalizer in s.normalizers: alternativeSet.add(normalizer)
            alternativeSet = list(alternativeSet)

            mask = np.zeros((len(alternativeSet), G))
            for tau in range(len(alternativeSet)):
                for p, production in enumerate(self.grammar.primitives):
                    mask[tau,p] = 0. if production in alternativeSet[tau] else NEGATIVEINFINITY
                mask[tau, G - 1] = 0. if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
            mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

            z = mask.repeat(self.n_grammars,1,1).repeat(B,1,1,1) + \
                logProductions.repeat(len(alternativeSet),1,1,1).transpose(0,1).transpose(1,2)
            z = torch.logsumexp(z, 3) # pytorch 1.0 dependency

            N = np.zeros((B, self.n_grammars, len(alternativeSet)))
            for b, summary in enumerate(summaries):
                for e, ss in summary.library.items():
                    for g,s in zip(self.library[e], ss):
                        assert g < self.n_grammars - 2
                        for r, alternatives in enumerate(alternativeSet):                
                            N[b,g,r] = s.normalizers.get(alternatives, 0.)
                # noParent: this is the last network output
                for r, alternatives in enumerate(alternativeSet):
                    N[b,self.n_grammars - 1,r] = summary.noParent.normalizers.get(alternatives, 0.)
                # variableParent: this is the penultimate network output
                for r, alternatives in enumerate(alternativeSet):
                    N[b,self.n_grammars - 2,r] = summary.variableParent.normalizers.get(alternatives, 0.)
            N = maybe_cuda(torch.tensor(N).float(), use_cuda)
            denominator = (N*z).sum(1).sum(1)
        else:
            gs = [ self(xs[b]) for b in range(B) ]
            denominator = torch.cat([ summary.denominator(g) for summary,g in zip(summaries, gs) ])
            
            

        
        
        ll = numerator - denominator 

        if False: # verifying that batching works correctly
            gs = [ self(xs[b]) for b in range(B) ]
            _l = torch.cat([ summary.logLikelihood(g) for summary,g in zip(summaries, gs) ])
            assert torch.all((ll - _l).abs() < 0.0001)
        return ll
    
class ContextualGrammarNetwork_Mask(nn.Module):
    def __init__(self, inputDimensionality, grammar):
        """Bigram model, but where the bigram transitions are unconditional.
        Individual primitive probabilities are still conditional (predicted by neural network)
        """
        
        super(ContextualGrammarNetwork_Mask, self).__init__()

        self.grammar = grammar

        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        self.n_grammars = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(self.n_grammars, self.n_grammars+numberOfArguments))
            self.library[prim] = idx_list
            self.n_grammars += numberOfArguments
        
        # We had an extra grammar for when there is no parent and for when the parent is a variable
        self.n_grammars += 2
        self._transitionMatrix = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.n_grammars, len(grammar) + 1)))
        self._logProductions = nn.Linear(inputDimensionality, len(grammar)+1)

    def transitionMatrix(self, x):
        if len(x.shape) == 1: # not batched
            return self._logProductions(x) + self._transitionMatrix # will broadcast
        elif len(x.shape) == 2: # batched
            return self._logProductions(x).unsqueeze(1).repeat(1,self.n_grammars,1) + \
                self._transitionMatrix.unsqueeze(0).repeat(x.size(0),1,1)
        else:
            assert False, "unknown shape for transition matrix input"
        
    def grammarFromVector(self, logProductions):
        return Grammar(logProductions[-1].view(1),
                       [(logProductions[k].view(1), t, program)
                        for k, (_, t, program) in enumerate(self.grammar.productions)],
                       continuationType=self.grammar.continuationType)

    def forward(self, x):
        assert len(x.size()) == 1, "contextual grammar doesn't currently support batching"

        transitionMatrix = self.transitionMatrix(x)
        
        return ContextualGrammar(self.grammarFromVector(transitionMatrix[-1]), self.grammarFromVector(transitionMatrix[-2]),
                {prim: [self.grammarFromVector(transitionMatrix[j]) for j in js]
                 for prim, js in self.library.items()} )
        
    def batchedLogLikelihoods(self, xs, summaries):
        """Takes as input BxinputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary"""
        use_cuda = xs.device.type == 'cuda'
        
        B = xs.shape[0]
        G = len(self.grammar) + 1
        assert len(summaries) == B

        # logProductions: Bx n_grammars x G
        logProductions = self.transitionMatrix(xs)
        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B,self.n_grammars,len(self.grammar)+1))
        for b,summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g,s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b,g,p] = s.uses.get(production, 0.)
                    uses[b,g,len(self.grammar)] = s.uses.get(Index(0), 0)
                    
            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(Index(0), 0.)

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(Index(0), 0.)
            
        numerator = (logProductions*maybe_cuda(torch.tensor(uses).float(),use_cuda)).view(B,-1).sum(1)

        constant = np.zeros(B)
        for b,summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant
            
        numerator += maybe_cuda(torch.tensor(constant).float(),use_cuda)
        
        if True:

            # Calculate the god-awful denominator
            alternativeSet = set()
            for summary in summaries:
                for normalizer in summary.noParent.normalizers: alternativeSet.add(normalizer)
                for normalizer in summary.variableParent.normalizers: alternativeSet.add(normalizer)
                for ss in summary.library.values():
                    for s in ss:
                        for normalizer in s.normalizers: alternativeSet.add(normalizer)
            alternativeSet = list(alternativeSet)

            mask = np.zeros((len(alternativeSet), G))
            for tau in range(len(alternativeSet)):
                for p, production in enumerate(self.grammar.primitives):
                    mask[tau,p] = 0. if production in alternativeSet[tau] else NEGATIVEINFINITY
                mask[tau, G - 1] = 0. if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
            mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

            z = mask.repeat(self.n_grammars,1,1).repeat(B,1,1,1) + \
                logProductions.repeat(len(alternativeSet),1,1,1).transpose(0,1).transpose(1,2)
            z = torch.logsumexp(z, 3) # pytorch 1.0 dependency

            N = np.zeros((B, self.n_grammars, len(alternativeSet)))
            for b, summary in enumerate(summaries):
                for e, ss in summary.library.items():
                    for g,s in zip(self.library[e], ss):
                        assert g < self.n_grammars - 2
                        for r, alternatives in enumerate(alternativeSet):                
                            N[b,g,r] = s.normalizers.get(alternatives, 0.)
                # noParent: this is the last network output
                for r, alternatives in enumerate(alternativeSet):
                    N[b,self.n_grammars - 1,r] = summary.noParent.normalizers.get(alternatives, 0.)
                # variableParent: this is the penultimate network output
                for r, alternatives in enumerate(alternativeSet):
                    N[b,self.n_grammars - 2,r] = summary.variableParent.normalizers.get(alternatives, 0.)
            N = maybe_cuda(torch.tensor(N).float(), use_cuda)
            denominator = (N*z).sum(1).sum(1)
        else:
            gs = [ self(xs[b]) for b in range(B) ]
            denominator = torch.cat([ summary.denominator(g) for summary,g in zip(summaries, gs) ])
            
            

        
        
        ll = numerator - denominator

        if False: # verifying that batching works correctly
            gs = [ self(xs[b]) for b in range(B) ]
            _l = torch.cat([ summary.logLikelihood(g) for summary,g in zip(summaries, gs) ])
            assert torch.all((ll - _l).abs() < 0.0001)
        return ll
        
                

class ContextualGrammarNetwork(nn.Module):
    """Like GrammarNetwork but ~contextual~"""
    def __init__(self, inputDimensionality, grammar):
        super(ContextualGrammarNetwork, self).__init__()
        
        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        self.n_grammars = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(self.n_grammars, self.n_grammars+numberOfArguments))
            self.library[prim] = idx_list
            self.n_grammars += numberOfArguments
        
        # We had an extra grammar for when there is no parent and for when the parent is a variable
        self.n_grammars += 2
        self.network = nn.Linear(inputDimensionality, (self.n_grammars)*(len(grammar) + 1))


    def grammarFromVector(self, logProductions):
        return Grammar(logProductions[-1].view(1),
                       [(logProductions[k].view(1), t, program)
                        for k, (_, t, program) in enumerate(self.grammar.productions)],
                       continuationType=self.grammar.continuationType)

    def forward(self, x):
        assert len(x.size()) == 1, "contextual grammar doesn't currently support batching"

        allVars = self.network(x).view(self.n_grammars, -1)
        return ContextualGrammar(self.grammarFromVector(allVars[-1]), self.grammarFromVector(allVars[-2]),
                {prim: [self.grammarFromVector(allVars[j]) for j in js]
                 for prim, js in self.library.items()} )

    def batchedLogLikelihoods(self, xs, summaries):
        use_cuda = xs.device.type == 'cuda'
        """Takes as input BxinputDimensionality vector & B likelihood summaries;
        returns B-dimensional vector containing log likelihood of each summary"""

        B = xs.shape[0]
        G = len(self.grammar) + 1
        assert len(summaries) == B

        # logProductions: Bx n_grammars x G
        logProductions = self.network(xs).view(B, self.n_grammars, G)
        # uses[b][g][p] is # uses of primitive p by summary b for parent g
        uses = np.zeros((B,self.n_grammars,len(self.grammar)+1))
        for b,summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g,s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for p, production in enumerate(self.grammar.primitives):
                        uses[b,g,p] = s.uses.get(production, 0.)
                    uses[b,g,len(self.grammar)] = s.uses.get(Index(0), 0)
                    
            # noParent: this is the last network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 1, p] = summary.noParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 1, G - 1] = summary.noParent.uses.get(Index(0), 0.)

            # variableParent: this is the penultimate network output
            for p, production in enumerate(self.grammar.primitives):            
                uses[b, self.n_grammars - 2, p] = summary.variableParent.uses.get(production, 0.)
            uses[b, self.n_grammars - 2, G - 1] = summary.variableParent.uses.get(Index(0), 0.)
            
        numerator = (logProductions*maybe_cuda(torch.tensor(uses).float(),use_cuda)).view(B,-1).sum(1)

        constant = np.zeros(B)
        for b,summary in enumerate(summaries):
            constant[b] += summary.noParent.constant + summary.variableParent.constant
            for ss in summary.library.values():
                for s in ss:
                    constant[b] += s.constant
            
        numerator += maybe_cuda(torch.tensor(constant).float(),use_cuda)

        # Calculate the god-awful denominator
        alternativeSet = set()
        for summary in summaries:
            for normalizer in summary.noParent.normalizers: alternativeSet.add(normalizer)
            for normalizer in summary.variableParent.normalizers: alternativeSet.add(normalizer)
            for ss in summary.library.values():
                for s in ss:
                    for normalizer in s.normalizers: alternativeSet.add(normalizer)
        alternativeSet = list(alternativeSet)

        mask = np.zeros((len(alternativeSet), G))
        for tau in range(len(alternativeSet)):
            for p, production in enumerate(self.grammar.primitives):
                mask[tau,p] = 0. if production in alternativeSet[tau] else NEGATIVEINFINITY
            mask[tau, G - 1] = 0. if Index(0) in alternativeSet[tau] else NEGATIVEINFINITY
        mask = maybe_cuda(torch.tensor(mask).float(), use_cuda)

        z = mask.repeat(self.n_grammars,1,1).repeat(B,1,1,1) + \
            logProductions.repeat(len(alternativeSet),1,1,1).transpose(0,1).transpose(1,2)
        z = torch.logsumexp(z, 3) # pytorch 1.0 dependency

        N = np.zeros((B, self.n_grammars, len(alternativeSet)))
        for b, summary in enumerate(summaries):
            for e, ss in summary.library.items():
                for g,s in zip(self.library[e], ss):
                    assert g < self.n_grammars - 2
                    for r, alternatives in enumerate(alternativeSet):                
                        N[b,g,r] = s.normalizers.get(alternatives, 0.)
            # noParent: this is the last network output
            for r, alternatives in enumerate(alternativeSet):
                N[b,self.n_grammars - 1,r] = summary.noParent.normalizers.get(alternatives, 0.)
            # variableParent: this is the penultimate network output
            for r, alternatives in enumerate(alternativeSet):
                N[b,self.n_grammars - 2,r] = summary.variableParent.normalizers.get(alternatives, 0.)
        N = maybe_cuda(torch.tensor(N).float(), use_cuda)
        

        
        denominator = (N*z).sum(1).sum(1)
        ll = numerator - denominator

        if False: # verifying that batching works correctly
            gs = [ self(xs[b]) for b in range(B) ]
            _l = torch.cat([ summary.logLikelihood(g) for summary,g in zip(summaries, gs) ])
            assert torch.all((ll - _l).abs() < 0.0001)

        return ll
        

class RecognitionModel(nn.Module):
    def __init__(self,featureExtractor,grammar,hidden=[64],activation="tanh",
                 rank=None,contextual=False,mask=False,
                 cuda=False,
                 previousRecognitionModel=None,
                 id=0):
        super(RecognitionModel, self).__init__()
        self.id = id
        self.trained=False
        self.use_cuda = cuda

        self.featureExtractor = featureExtractor
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(myParameter is parameter for myParameter in self.parameters())

        # Build the multilayer perceptron that is sandwiched between the feature extractor and the grammar
        if activation == "sigmoid":
            activation = nn.Sigmoid
        elif activation == "relu":
            activation = nn.ReLU
        elif activation == "tanh":
            activation = nn.Tanh
        else:
            raise Exception('Unknown activation function ' + str(activation))
        self._MLP = nn.Sequential(*[ layer
                                     for j in range(len(hidden))
                                     for layer in [
                                             nn.Linear(([featureExtractor.outputDimensionality] + hidden)[j],
                                                       hidden[j]),
                                             activation()]])

        self.entropy = Entropy()

        if len(hidden) > 0:
            self.outputDimensionality = self._MLP[-2].out_features
            assert self.outputDimensionality == hidden[-1]
        else:
            self.outputDimensionality = self.featureExtractor.outputDimensionality

        self.contextual = contextual
        if self.contextual:
            if mask:
                self.grammarBuilder = ContextualGrammarNetwork_Mask(self.outputDimensionality, grammar)
            else:
                self.grammarBuilder = ContextualGrammarNetwork_LowRank(self.outputDimensionality, grammar, rank)
        else:
            self.grammarBuilder = GrammarNetwork(self.outputDimensionality, grammar)
        
        self.grammar = ContextualGrammar.fromGrammar(grammar) if contextual else grammar
        self.generativeModel = grammar
        
        self._auxiliaryPrediction = nn.Linear(self.featureExtractor.outputDimensionality, 
                                              len(self.grammar.primitives))
        self._auxiliaryLoss = nn.BCEWithLogitsLoss()

        if cuda: self.cuda()

        if previousRecognitionModel:
            self._MLP.load_state_dict(previousRecognitionModel._MLP.state_dict())
            self.featureExtractor.load_state_dict(previousRecognitionModel.featureExtractor.state_dict())
            
    def auxiliaryLoss(self, frontier, features):
        # Compute a vector of uses
        ls = frontier.bestPosterior.program
        def uses(summary):
            if hasattr(summary, 'uses'): 
                return torch.tensor([ float(int(p in summary.uses))
                                      for p in self.generativeModel.primitives ])
            assert hasattr(summary, 'noParent')
            u = uses(summary.noParent) + uses(summary.variableParent)
            for ss in summary.library.values():
                for s in ss:
                    u += uses(s)
            return u
        u = uses(ls)
        u[u > 1.] = 1.
        if self.use_cuda: u = u.cuda()
        al = self._auxiliaryLoss(self._auxiliaryPrediction(features), u)
        return al
            
    def taskEmbeddings(self, tasks):
        return {task: self.featureExtractor.featuresOfTask(task).data.cpu().numpy()
                for task in tasks}

    def forward(self, features):
        """returns either a Grammar or a ContextualGrammar
        Takes as input the output of featureExtractor.featuresOfTask"""
        features = self._MLP(features)
        return self.grammarBuilder(features)

    def auxiliaryPrimitiveEmbeddings(self):
        """Returns the actual outputDimensionality weight vectors for each of the primitives."""
        auxiliaryWeights = self._auxiliaryPrediction.weight.data.cpu().numpy()
        primitivesDict =  {self.grammar.primitives[i] : auxiliaryWeights[i, :] for i in range(len(self.grammar.primitives))}
        return primitivesDict

    def grammarOfTask(self, task):
        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None
        return self(features)

    def grammarLogProductionsOfTask(self, task):
        """Returns the grammar logits from non-contextual models."""

        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None

        if hasattr(self, 'hiddenLayers'):
            # Backward compatability with old checkpoints.
            for layer in self.hiddenLayers:
                features = self.activation(layer(features))
            # return features
            return self.noParent[1](features)
        else:
            features = self._MLP(features)

        if self.contextual:
            if hasattr(self.grammarBuilder, 'variableParent'):
                return self.grammarBuilder.variableParent.logProductions(features)
            elif hasattr(self.grammarBuilder, 'network'):
                return self.grammarBuilder.network(features).view(-1)
            elif hasattr(self.grammarBuilder, 'transitionMatrix'):
                return self.grammarBuilder.transitionMatrix(features).view(-1)
            else:
                assert False
        else:
            return self.grammarBuilder.logProductions(features)

    def grammarFeatureLogProductionsOfTask(self, task):
        return torch.tensor(self.grammarOfTask(task).untorch().featureVector())

    def grammarLogProductionDistanceToTask(self, task, tasks):
        """Returns the cosine similarity of all other tasks to a given task."""
        taskLogits = self.grammarLogProductionsOfTask(task).unsqueeze(0) # Change to [1, D]
        assert taskLogits is not None, 'Grammar log productions are not defined for this task.'
        otherTasks = [t for t in tasks if t is not task] # [nTasks -1 , D]

        # Build matrix of all other tasks.
        otherLogits = torch.stack([self.grammarLogProductionsOfTask(t) for t in otherTasks])
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosMatrix = cos(taskLogits, otherLogits)
        return cosMatrix.data.cpu().numpy()

    def grammarEntropyOfTask(self, task):
        """Returns the entropy of the grammar distribution from non-contextual models for a task."""
        grammarLogProductionsOfTask = self.grammarLogProductionsOfTask(task)

        if grammarLogProductionsOfTask is None: return None

        if hasattr(self, 'entropy'):
            return self.entropy(grammarLogProductionsOfTask)
        else:
            e = Entropy()
            return e(grammarLogProductionsOfTask)

    def taskAuxiliaryLossLayer(self, tasks):
        return {task: self._auxiliaryPrediction(self.featureExtractor.featuresOfTask(task)).view(-1).data.cpu().numpy()
                for task in tasks}
                
    def taskGrammarFeatureLogProductions(self, tasks):
        return {task: self.grammarFeatureLogProductionsOfTask(task).data.cpu().numpy()
                for task in tasks}

    def taskGrammarLogProductions(self, tasks):
        return {task: self.grammarLogProductionsOfTask(task).data.cpu().numpy()
                for task in tasks}

    def taskGrammarStartProductions(self, tasks):
        return {task: np.array([l for l,_1,_2 in g.productions ])
                for task in tasks
                for g in [self.grammarOfTask(task).untorch().noParent] }

    def taskHiddenStates(self, tasks):
        return {task: self._MLP(self.featureExtractor.featuresOfTask(task)).view(-1).data.cpu().numpy()
                for task in tasks}

    def taskGrammarEntropies(self, tasks):
        return {task: self.grammarEntropyOfTask(task).data.cpu().numpy()
                for task in tasks}

    def frontierKL(self, frontier, auxiliary=False, vectorized=True):
        features = self.featureExtractor.featuresOfTask(frontier.task)
        if features is None: return None, None
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()

        al = self.auxiliaryLoss(frontier, features if auxiliary else features.detach())

        if not vectorized:
            g = self(features)
            return - entry.program.logLikelihood(g), al
        else:
            features = self._MLP(features).expand(1, features.size(-1))
            ll = self.grammarBuilder.batchedLogLikelihoods(features, [entry.program]).view(-1)
            return -ll, al
            

    def frontierBiasOptimal(self, frontier, auxiliary=False, vectorized=True):
        if not vectorized:
            features = self.featureExtractor.featuresOfTask(frontier.task)
            if features is None: return None, None
            al = self.auxiliaryLoss(frontier, features if auxiliary else features.detach())
            g = self(features)
            summaries = [entry.program for entry in frontier]
            likelihoods = torch.cat([entry.program.logLikelihood(g) + entry.logLikelihood
                                     for entry in frontier ])
            best = likelihoods.max()
            return -best, al
            
        batchSize = len(frontier.entries)
        features = self.featureExtractor.featuresOfTask(frontier.task)
        if features is None: return None, None
        al = self.auxiliaryLoss(frontier, features if auxiliary else features.detach())
        features = self._MLP(features)
        features = features.expand(batchSize, features.size(-1))  # TODO
        lls = self.grammarBuilder.batchedLogLikelihoods(features, [entry.program for entry in frontier])
        actual_ll = torch.Tensor([ entry.logLikelihood for entry in frontier])
        lls = lls + (actual_ll.cuda() if self.use_cuda else actual_ll)
        ml = -lls.max() #Beware that inputs to max change output type
        return ml, al

    def replaceProgramsWithLikelihoodSummaries(self, frontier):
        return Frontier(
            [FrontierEntry(
                program=self.grammar.closedLikelihoodSummary(frontier.task.request, e.program),
                logLikelihood=e.logLikelihood,
                logPrior=e.logPrior) for e in frontier],
            task=frontier.task)

    def train(self, frontiers, _=None, steps=None, lr=0.001, topK=5, CPUs=1,
              timeout=None, evaluationTimeout=0.001,
              helmholtzFrontiers=[], helmholtzRatio=0., helmholtzBatch=500,
              biasOptimal=None, defaultRequest=None, auxLoss=False, vectorized=True):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        helmholtzFrontiers: Frontiers from programs enumerated from generative model (optional)
        If helmholtzFrontiers is not provided then we will sample programs during training
        """
        assert (steps is not None) or (timeout is not None), \
            "Cannot train recognition model without either a bound on the number of gradient steps or bound on the training time"
        if steps is None: steps = 9999999
        if biasOptimal is None: biasOptimal = len(helmholtzFrontiers) > 0
        
        requests = [frontier.task.request for frontier in frontiers]
        if len(requests) == 0 and helmholtzRatio > 0 and len(helmholtzFrontiers) == 0:
            assert defaultRequest is not None, "You are trying to random Helmholtz training, but don't have any frontiers. Therefore we would not know the type of the program to sample. Try specifying defaultRequest=..."
            requests = [defaultRequest]
        frontiers = [frontier.topK(topK).normalize()
                     for frontier in frontiers if not frontier.empty]
        if len(frontiers) == 0:
            eprint("You didn't give me any nonempty replay frontiers to learn from. Going to learn from 100% Helmholtz samples")
            helmholtzRatio = 1.

        # Should we sample programs or use the enumerated programs?
        randomHelmholtz = len(helmholtzFrontiers) == 0
        
        class HelmholtzEntry:
            def __init__(self, frontier, owner):
                self.request = frontier.task.request
                self.task = None
                self.programs = [e.program for e in frontier]
                self.frontier = Thunk(lambda: owner.replaceProgramsWithLikelihoodSummaries(frontier))
                self.owner = owner

            def clear(self): self.task = None

            def calculateTask(self):
                assert self.task is None
                p = random.choice(self.programs)
                return self.owner.featureExtractor.taskOfProgram(p, self.request)

            def makeFrontier(self):
                assert self.task is not None
                f = Frontier(self.frontier.force().entries,
                             task=self.task)
                return f
        
            
            

        # Should we recompute tasks on the fly from Helmholtz?  This
        # should be done if the task is stochastic, or if there are
        # different kinds of inputs on which it could be run. For
        # example, lists and strings need this; towers and graphics do
        # not. There is no harm in recomputed the tasks, it just
        # wastes time.
        if not hasattr(self.featureExtractor, 'recomputeTasks'):
            self.featureExtractor.recomputeTasks = True
        helmholtzFrontiers = [HelmholtzEntry(f, self)
                              for f in helmholtzFrontiers]
        random.shuffle(helmholtzFrontiers)
        
        helmholtzIndex = [0]
        def getHelmholtz():
            if randomHelmholtz:
                if helmholtzIndex[0] >= len(helmholtzFrontiers):
                    updateHelmholtzTasks()
                    helmholtzIndex[0] = 0
                    return getHelmholtz()
                helmholtzIndex[0] += 1
                return helmholtzFrontiers[helmholtzIndex[0] - 1].makeFrontier()

            f = helmholtzFrontiers[helmholtzIndex[0]]
            if f.task is None:
                with timing("Evaluated another batch of Helmholtz tasks"):
                    updateHelmholtzTasks()
                return getHelmholtz()

            helmholtzIndex[0] += 1
            if helmholtzIndex[0] >= len(helmholtzFrontiers):
                helmholtzIndex[0] = 0
                random.shuffle(helmholtzFrontiers)
                if self.featureExtractor.recomputeTasks:
                    for fp in helmholtzFrontiers:
                        fp.clear()
                    return getHelmholtz() # because we just cleared everything
            assert f.task is not None
            return f.makeFrontier()
            
        def updateHelmholtzTasks():
            updateCPUs = CPUs if hasattr(self.featureExtractor, 'parallelTaskOfProgram') and self.featureExtractor.parallelTaskOfProgram else 1
            if updateCPUs > 1: eprint("Updating Helmholtz tasks with",updateCPUs,"CPUs",
                                      "while using",getThisMemoryUsage(),"memory")
            
            if randomHelmholtz:
                newFrontiers = self.sampleManyHelmholtz(requests, helmholtzBatch, CPUs)
                newEntries = []
                for f in newFrontiers:
                    e = HelmholtzEntry(f,self)
                    e.task = f.task
                    newEntries.append(e)
                helmholtzFrontiers.clear()
                helmholtzFrontiers.extend(newEntries)
                return 

            # Save some memory by freeing up the tasks as we go through them
            if self.featureExtractor.recomputeTasks:
                for hi in range(max(0, helmholtzIndex[0] - helmholtzBatch,
                                    min(helmholtzIndex[0], len(helmholtzFrontiers)))):
                    helmholtzFrontiers[hi].clear()

            if hasattr(self.featureExtractor, 'tasksOfPrograms'):
                eprint("batching task calculation")
                newTasks = self.featureExtractor.tasksOfPrograms(
                    [random.choice(hf.programs)
                     for hf in helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + helmholtzBatch] ],
                    [hf.request
                     for hf in helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + helmholtzBatch] ])
            else:
                newTasks = [hf.calculateTask() 
                            for hf in helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + helmholtzBatch]]

                """
                # catwong: Disabled for ensemble training.
                newTasks = \
                           parallelMap(updateCPUs,
                                       lambda f: f.calculateTask(),
                                       helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + helmholtzBatch],
                                       seedRandom=True)
                """
            badIndices = []
            endingIndex = min(helmholtzIndex[0] + helmholtzBatch, len(helmholtzFrontiers))
            for i in range(helmholtzIndex[0], endingIndex):
                helmholtzFrontiers[i].task = newTasks[i - helmholtzIndex[0]]
                if helmholtzFrontiers[i].task is None: badIndices.append(i)
            # Permanently kill anything which failed to give a task
            for i in reversed(badIndices):
                assert helmholtzFrontiers[i].task is None
                del helmholtzFrontiers[i]


        # We replace each program in the frontier with its likelihoodSummary
        # This is because calculating likelihood summaries requires juggling types
        # And type stuff is expensive!
        frontiers = [self.replaceProgramsWithLikelihoodSummaries(f).normalize()
                     for f in frontiers]

        eprint("(ID=%d): Training a recognition model from %d frontiers, %d%% Helmholtz, feature extractor %s." % (
            self.id, len(frontiers), int(helmholtzRatio * 100), self.featureExtractor.__class__.__name__))
        eprint("(ID=%d): Got %d Helmholtz frontiers - random Helmholtz training? : %s"%(
            self.id, len(helmholtzFrontiers), len(helmholtzFrontiers) == 0))
        eprint("(ID=%d): Contextual? %s" % (self.id, str(self.contextual)))
        eprint("(ID=%d): Bias optimal? %s" % (self.id, str(biasOptimal)))
        eprint(f"(ID={self.id}): Aux loss? {auxLoss} (n.b. we train a 'auxiliary' classifier anyway - this controls if gradients propagate back to the future extractor)")

        # The number of Helmholtz samples that we generate at once
        # Should only affect performance and shouldn't affect anything else
        helmholtzSamples = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        start = time.time()
        losses, descriptionLengths, realLosses, dreamLosses, realMDL, dreamMDL = [], [], [], [], [], []
        classificationLosses = []
        totalGradientSteps = 0
        epochs = 9999999
        for i in range(1, epochs + 1):
            if timeout and time.time() - start > timeout:
                break

            if totalGradientSteps > steps:
                break

            if helmholtzRatio < 1.:
                permutedFrontiers = list(frontiers)
                random.shuffle(permutedFrontiers)
            else:
                permutedFrontiers = [None]

            finishedSteps = False
            for frontier in permutedFrontiers:
                # Randomly decide whether to sample from the generative model
                dreaming = random.random() < helmholtzRatio
                if dreaming: frontier = getHelmholtz()
                self.zero_grad()
                loss, classificationLoss = \
                        self.frontierBiasOptimal(frontier, auxiliary=auxLoss, vectorized=vectorized) if biasOptimal \
                        else self.frontierKL(frontier, auxiliary=auxLoss, vectorized=vectorized)
                if loss is None:
                    if not dreaming:
                        eprint("ERROR: Could not extract features during experience replay.")
                        eprint("Task is:",frontier.task)
                        eprint("Aborting - we need to be able to extract features of every actual task.")
                        assert False
                    else:
                        continue
                if is_torch_invalid(loss):
                    eprint("Invalid real-data loss!")
                else:
                    (loss + classificationLoss).backward()
                    classificationLosses.append(classificationLoss.data.item())
                    optimizer.step()
                    totalGradientSteps += 1
                    losses.append(loss.data.item())
                    descriptionLengths.append(min(-e.logPrior for e in frontier))
                    if dreaming:
                        dreamLosses.append(losses[-1])
                        dreamMDL.append(descriptionLengths[-1])
                    else:
                        realLosses.append(losses[-1])
                        realMDL.append(descriptionLengths[-1])
                    if totalGradientSteps > steps:
                        break # Stop iterating, then print epoch and loss, then break to finish.
                        
            if (i == 1 or i % 10 == 0) and losses:
                eprint("(ID=%d): " % self.id, "Epoch", i, "Loss", mean(losses))
                if realLosses and dreamLosses:
                    eprint("(ID=%d): " % self.id, "\t\t(real loss): ", mean(realLosses), "\t(dream loss):", mean(dreamLosses))
                eprint("(ID=%d): " % self.id, "\tvs MDL (w/o neural net)", mean(descriptionLengths))
                if realMDL and dreamMDL:
                    eprint("\t\t(real MDL): ", mean(realMDL), "\t(dream MDL):", mean(dreamMDL))
                eprint("(ID=%d): " % self.id, "\t%d cumulative gradient steps. %f steps/sec"%(totalGradientSteps,
                                                                       totalGradientSteps/(time.time() - start)))
                eprint("(ID=%d): " % self.id, "\t%d-way auxiliary classification loss"%len(self.grammar.primitives),sum(classificationLosses)/len(classificationLosses))
                losses, descriptionLengths, realLosses, dreamLosses, realMDL, dreamMDL = [], [], [], [], [], []
                classificationLosses = []
                gc.collect()
        
        eprint("(ID=%d): " % self.id, " Trained recognition model in",time.time() - start,"seconds")
        self.trained=True
        return self

    def sampleHelmholtz(self, requests, statusUpdate=None, seed=None):
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        program = self.generativeModel.sample(request, maximumDepth=6, maxAttempts=100)
        if program is None:
            return None
        task = self.featureExtractor.taskOfProgram(program, request)

        if statusUpdate is not None:
            flushEverything()
        if task is None:
            return None

        if hasattr(self.featureExtractor, 'lexicon'):
            if self.featureExtractor.tokenize(task.examples) is None:
                return None
        
        ll = self.generativeModel.logLikelihood(request, program)
        frontier = Frontier([FrontierEntry(program=program,
                                           logLikelihood=0., logPrior=ll)],
                            task=task)
        return frontier

    def sampleManyHelmholtz(self, requests, N, CPUs):
        eprint("Sampling %d programs from the prior on %d CPUs..." % (N, CPUs))
        flushEverything()
        frequency = N / 50
        startingSeed = random.random()

        # Sequentially for ensemble training.
        samples = [self.sampleHelmholtz(requests,
                                           statusUpdate='.' if n % frequency == 0 else None,
                                           seed=startingSeed + n) for n in range(N)]

        # (cathywong) Disabled for ensemble training. 
        # samples = parallelMap(
        #     1,
        #     lambda n: self.sampleHelmholtz(requests,
        #                                    statusUpdate='.' if n % frequency == 0 else None,
        #                                    seed=startingSeed + n),
        #     range(N))
        eprint()
        flushEverything()
        samples = [z for z in samples if z is not None]
        eprint()
        eprint("Got %d/%d valid samples." % (len(samples), N))
        flushEverything()

        return samples

    def enumerateFrontiers(self,
                           tasks,
                           enumerationTimeout=None,
                           testing=False,
                           solver=None,
                           CPUs=1,
                           frontierSize=None,
                           maximumFrontier=None,
                           evaluationTimeout=None):
        with timing("Evaluated recognition model"):
            grammars = {task: self.grammarOfTask(task)
                        for task in tasks}
            #untorch seperately to make sure you filter out None grammars
            grammars = {task: grammar.untorch() for task, grammar in grammars.items() if grammar is not None}

        return multicoreEnumeration(grammars, tasks,
                                    testing=testing,
                                    solver=solver,
                                    enumerationTimeout=enumerationTimeout,
                                    CPUs=CPUs, maximumFrontier=maximumFrontier,
                                    evaluationTimeout=evaluationTimeout)


class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, _=None,
                 tasks=None,
                 cuda=False,
                 # what are the symbols that can occur in the inputs and
                 # outputs
                 lexicon=None,
                 # how many hidden units
                 H=32,
                 # Should the recurrent units be bidirectional?
                 bidirectional=False,
                 # What should be the timeout for trying to construct Helmholtz tasks?
                 helmholtzTimeout=0.25,
                 # What should be the timeout for running a Helmholtz program?
                 helmholtzEvaluationTimeout=0.01):
        super(RecurrentFeatureExtractor, self).__init__()

        assert tasks is not None, "You must provide a list of all of the tasks, both those that have been hit and those that have not been hit. Input examples are sampled from these tasks."

        # maps from a requesting type to all of the inputs that we ever saw with that request
        self.requestToInputs = {
            tp: [list(map(fst, t.examples)) for t in tasks if t.request == tp ]
            for tp in {t.request for t in tasks}
        }

        inputTypes = {t
                      for task in tasks
                      for t in task.request.functionArguments()}
        # maps from a type to all of the inputs that we ever saw having that type
        self.argumentsWithType = {
            tp: [ x
                  for t in tasks
                  for xs,_ in t.examples
                  for tpp, x in zip(t.request.functionArguments(), xs)
                  if tpp == tp]
            for tp in inputTypes
        }
        self.requestToNumberOfExamples = {
            tp: [ len(t.examples)
                  for t in tasks if t.request == tp ]
            for tp in {t.request for t in tasks}
        }
        self.helmholtzTimeout = helmholtzTimeout
        self.helmholtzEvaluationTimeout = helmholtzEvaluationTimeout
        self.parallelTaskOfProgram = True
        
        assert lexicon
        self.specialSymbols = [
            "STARTING",  # start of entire sequence
            "ENDING",  # ending of entire sequence
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDOFINPUT"  # delimits the ending of an input - we might have multiple inputs
        ]
        lexicon += self.specialSymbols
        encoder = nn.Embedding(len(lexicon), H)
        self.encoder = encoder

        self.H = H
        self.bidirectional = bidirectional

        layers = 1

        model = nn.GRU(H, H, layers, bidirectional=bidirectional)
        self.model = model

        self.use_cuda = cuda
        self.lexicon = lexicon
        self.symbolToIndex = {
            symbol: index for index,
            symbol in enumerate(lexicon)}
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endingIndex = self.symbolToIndex["ENDING"]
        self.startOfOutputIndex = self.symbolToIndex["STARTOFOUTPUT"]
        self.endOfInputIndex = self.symbolToIndex["ENDOFINPUT"]

        # Maximum number of inputs/outputs we will run the recognition
        # model on per task
        # This is an optimization hack
        self.MAXINPUTS = 100

        if cuda: self.cuda()

    @property
    def outputDimensionality(self): return self.H

    # modify examples before forward (to turn them into iterables of lexicon)
    # you should override this if needed
    def tokenize(self, x): return x

    def symbolEmbeddings(self):
        return {s: self.encoder(variable([self.symbolToIndex[s]])).squeeze(
            0).data.cpu().numpy() for s in self.lexicon if not (s in self.specialSymbols)}

    def packExamples(self, examples):
        """IMPORTANT! xs must be sorted in decreasing order of size because pytorch is stupid"""
        es = []
        sizes = []
        for xs, y in examples:
            e = [self.startingIndex]
            for x in xs:
                for s in x:
                    e.append(self.symbolToIndex[s])
                e.append(self.endOfInputIndex)
            e.append(self.startOfOutputIndex)
            for s in y:
                e.append(self.symbolToIndex[s])
            e.append(self.endingIndex)
            if es != []:
                assert len(e) <= len(es[-1]), \
                    "Examples must be sorted in decreasing order of their tokenized size. This should be transparently handled in recognition.py, so if this assertion fails it isn't your fault as a user of EC but instead is a bug inside of EC."
            es.append(e)
            sizes.append(len(e))

        m = max(sizes)
        # padding
        for j, e in enumerate(es):
            es[j] += [self.endingIndex] * (m - len(e))

        x = variable(es, cuda=self.use_cuda)
        x = self.encoder(x)
        # x: (batch size, maximum length, E)
        x = x.permute(1, 0, 2)
        # x: TxBxE
        x = pack_padded_sequence(x, sizes)
        return x, sizes

    def examplesEncoding(self, examples):
        examples = sorted(examples, key=lambda xs_y: sum(
            len(z) + 1 for z in xs_y[0]) + len(xs_y[1]), reverse=True)
        x, sizes = self.packExamples(examples)
        outputs, hidden = self.model(x)
        # outputs, sizes = pad_packed_sequence(outputs)
        # I don't know whether to return the final output or the final hidden
        # activations...
        return hidden[0, :, :] + hidden[1, :, :]

    def forward(self, examples):
        tokenized = self.tokenize(examples)
        if not tokenized:
            return None

        if hasattr(self, 'MAXINPUTS') and len(tokenized) > self.MAXINPUTS:
            tokenized = list(tokenized)
            random.shuffle(tokenized)
            tokenized = tokenized[:self.MAXINPUTS]
        e = self.examplesEncoding(tokenized)
        # max pool
        # e,_ = e.max(dim = 0)

        # take the average activations across all of the examples
        # I think this might be better because we might be testing on data
        # which has far more o far fewer examples then training
        e = e.mean(dim=0)
        return e

    def featuresOfTask(self, t):
        if hasattr(self, 'useFeatures'):
            f = self(t.features)
        else:
            # Featurize the examples directly.
            f = self(t.examples)
        return f

    def taskOfProgram(self, p, tp):
        # half of the time we randomly mix together inputs
        # this gives better generalization on held out tasks
        # the other half of the time we train on sets of inputs in the training data
        # this gives better generalization on unsolved training tasks
        if random.random() < 0.5:
            def randomInput(t): return random.choice(self.argumentsWithType[t])
            # Loop over the inputs in a random order and pick the first ones that
            # doesn't generate an exception

            startTime = time.time()
            examples = []
            while True:
                # TIMEOUT! this must not be a very good program
                if time.time() - startTime > self.helmholtzTimeout: return None

                # Grab some random inputs
                xs = [randomInput(t) for t in tp.functionArguments()]
                try:
                    y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                    examples.append((tuple(xs),y))
                    if len(examples) >= random.choice(self.requestToNumberOfExamples[tp]):
                        return Task("Helmholtz", tp, examples)
                except: continue

        else:
            candidateInputs = list(self.requestToInputs[tp])
            random.shuffle(candidateInputs)
            for xss in candidateInputs:
                ys = []
                for xs in xss:
                    try: y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                    except: break
                    ys.append(y)
                if len(ys) == len(xss):
                    return Task("Helmholtz", tp, list(zip(xss, ys)))
            return None
                
            
    
class LowRank(nn.Module):
    """
    Module that outputs a rank R matrix of size m by n from input of size i.
    """
    def __init__(self, i, m, n, r):
        """
        i: input dimension
        m: output rows
        n: output columns
        r: maximum rank. if this is None then the output will be full-rank
        """
        super(LowRank, self).__init__()

        self.m = m
        self.n = n
        
        maximumPossibleRank = min(m, n)
        if r is None: r = maximumPossibleRank
        
        if r < maximumPossibleRank:
            self.factored = True
            self.A = nn.Linear(i, m*r)
            self.B = nn.Linear(i, n*r)
            self.r = r
        else:
            self.factored = False
            self.M = nn.Linear(i, m*n)

    def forward(self, x):
        sz = x.size()
        if len(sz) == 1:
            B = 1
            x = x.unsqueeze(0)
            needToSqueeze = True
        elif len(sz) == 2:
            B = sz[0]
            needToSqueeze = False
        else:
            assert False, "LowRank expects either a 1-dimensional tensor or a 2-dimensional tensor"

        if self.factored:
            a = self.A(x).view(B, self.m, self.r)
            b = self.B(x).view(B, self.r, self.n)
            y = a @ b
        else:
            y = self.M(x).view(B, self.m, self.n)
        if needToSqueeze:
            y = y.squeeze(0)
        return y
            
            
            

class DummyFeatureExtractor(nn.Module):
    def __init__(self, tasks, testingTasks=[], cuda=False):
        super(DummyFeatureExtractor, self).__init__()
        self.outputDimensionality = 1
        self.recomputeTasks = False
    def featuresOfTask(self, t):
        return variable([0.]).float()
    def featuresOfTasks(self, ts):
        return variable([[0.]]*len(ts)).float()
    def taskOfProgram(self, p, t):
        return Task("dummy task", t, [])

class RandomFeatureExtractor(nn.Module):
    def __init__(self, tasks):
        super(RandomFeatureExtractor, self).__init__()
        self.outputDimensionality = 1
        self.recomputeTasks = False
    def featuresOfTask(self, t):
        return variable([random.random()]).float()
    def featuresOfTasks(self, ts):
        return variable([[random.random()] for _ in range(len(ts)) ]).float()
    def taskOfProgram(self, p, t):
        return Task("dummy task", t, [])

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ImageFeatureExtractor(nn.Module):
    def __init__(self, inputImageDimension, resizedDimension=None,
                 channels=1):
        super(ImageFeatureExtractor, self).__init__()
        
        self.resizedDimension = resizedDimension or inputImageDimension
        self.inputImageDimension = inputImageDimension
        self.channels = channels

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(channels, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )
        
        # Each layer of the encoder halves the dimension, except for the last layer which flattens
        outputImageDimensionality = self.resizedDimension/(2**(len(self.encoder) - 1))
        self.outputDimensionality = int(z_dim*outputImageDimensionality*outputImageDimensionality)

    def forward(self, v):
        """1 channel: v: BxWxW or v:WxW
        > 1 channel: v: BxCxWxW or v:CxWxW"""

        insertBatch = False
        variabled = variable(v).float()
        if self.channels == 1: # insert channel dimension
            if len(variabled.shape) == 3: # batching
                variabled = variabled[:,None,:,:]
            elif len(variabled.shape) == 2: # no batching
                variabled = variabled[None,:,:]
                insertBatch = True
            else: assert False
        else: # expect to have a channel dimension
            if len(variabled.shape) == 4:
                pass
            elif len(variabled.shape) == 3:
                insertBatch = True
            else: assert False                

        if insertBatch: variabled = torch.unsqueeze(variabled, 0)
        
        y = self.encoder(variabled)
        if insertBatch: y = y[0,:]
        return y

class JSONFeatureExtractor(object):
    def __init__(self, tasks, cudaFalse):
        # self.averages, self.deviations = Task.featureMeanAndStandardDeviation(tasks)
        # self.outputDimensionality = len(self.averages)
        self.cuda = cuda
        self.tasks = tasks

    def stringify(self, x):
        # No whitespace #maybe kill the seperators
        return json.dumps(x, separators=(',', ':'))

    def featuresOfTask(self, t):
        # >>> t.request to get the type
        # >>> t.examples to get input/output examples
        # this might actually be okay, because the input should just be nothing
        #return [(self.stringify(inputs), self.stringify(output))
        #        for (inputs, output) in t.examples]
        return [(list(output),) for (inputs, output) in t.examples]
