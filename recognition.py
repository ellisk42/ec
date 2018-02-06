from enumeration import *
from fragmentGrammar import *
from grammar import *
from utilities import eprint

import gc

import torch
import torch.nn as nn
import torch.optim as optimization
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

def variable(x, volatile=False, cuda=False):
    if isinstance(x,list): x = np.array(x)
    if isinstance(x,(np.ndarray,np.generic)): x = torch.from_numpy(x)
    if cuda: x = x.cuda()
    return Variable(x, volatile=volatile)

class RecognitionModel(nn.Module):
    def __init__(self, featureExtractor, grammar, hidden=[16], activation="relu", cuda=False):
        super(RecognitionModel, self).__init__()
        self.grammar = grammar
        self.use_cuda = cuda
        if cuda:
            self.cuda()
        else:
            # Torch sometimes segfaults in multithreaded mode...
            pass
            # torch.set_num_threads(1)

        self.featureExtractor = featureExtractor
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(myParameter is parameter for myParameter in self.parameters())
        
        self.hiddenLayers = []
        inputDimensionality = featureExtractor.outputDimensionality
        for h in hidden:
            layer = nn.Linear(inputDimensionality, h)
            if cuda:
                layer = layer.cuda()
            self.hiddenLayers.append(layer)
            inputDimensionality = h

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "relu":
            self.activation = lambda x: x.clamp(min = 0)
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise Exception('Unknown activation function '+str(activation))

        self.logVariable = nn.Linear(inputDimensionality,1)
        self.logProductions = nn.Linear(inputDimensionality, len(self.grammar))
        if cuda:
            self.logVariable = self.logVariable.cuda()
            self.logProductions = self.logProductions.cuda()

    def productionEmbedding(self):
        # Calculates the embedding 's of the primitives based on the last weight layer
        w = self.logProductions._parameters['weight']
        # w: len(self.grammar) x E
        if self.use_cuda:
            w = w.cpu()
        e = dict({p: w[j,:].data.numpy() for j,(t,l,p) in enumerate(self.grammar.productions) })
        e[Index(0)] = w[0,:].data.numpy()
        return e

    def taskEmbeddings(self, tasks):
        return {task: self.featureExtractor.featuresOfTask(task).data.numpy()
                for task in tasks }
        
    def forward(self, features):
        for layer in self.hiddenLayers:
            features = self.activation(layer(features))
        h = features
        return self.logVariable(h), self.logProductions(h)

    def frontierKL(self, frontier):
        features = self.featureExtractor.featuresOfTask(frontier.task)
        variables, productions = self(features)
        g = Grammar(variables, [(productions[k],t,program)
                                for k,(_,t,program) in enumerate(self.grammar.productions) ])
        kl = 0.
        for entry in frontier:
            kl -= math.exp(entry.logPosterior) * g.closedLogLikelihood(frontier.task.request, entry.program)
        return kl
    def HelmholtzKL(self, features, sample, tp):
        variables, productions = self(features)
        g = Grammar(variables, [(productions[k],t,program)
                                for k,(_,t,program) in enumerate(self.grammar.productions) ])
        return - g.closedLogLikelihood(tp, sample)

    def train(self, frontiers, _=None, steps=250, lr=0.001, topK=1, CPUs=1,
              helmholtzRatio = 0.):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        """
        requests = [ frontier.task.request for frontier in frontiers ]
        frontiers = [ frontier.topK(topK).normalize() for frontier in frontiers if not frontier.empty ]

        # Not sure why this ever happens
        if helmholtzRatio is None: helmholtzRatio = 0.

        eprint("Training a recognition model from %d frontiers, %d%% Helmholtz, feature extractor %s."%(
            len(frontiers),
            int(helmholtzRatio*100),
            self.featureExtractor.__class__.__name__))
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        with timing("Trained recognition model"):
            for i in range(1,steps + 1):
                losses = []
                
                permutedFrontiers = list(frontiers)
                random.shuffle(permutedFrontiers)
                for frontier in permutedFrontiers:
                    self.zero_grad()

                    # Randomly decide whether to sample from the generative model
                    doingHelmholtz = random.random() < helmholtzRatio
                    if doingHelmholtz:
                        attempt = self.sampleHelmholtz(requests)
                        if attempt is not None:
                            program, request, features = attempt
                            loss = self.HelmholtzKL(features, program, request)
                        else: doingHelmholtz = False
                    if not doingHelmholtz:
                        loss = self.frontierKL(frontier)
                    
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data[0])
                if i%50 == 0:
                    eprint("Epoch",i,"Loss",sum(losses)/len(losses))
                    gc.collect()

    def sampleHelmholtz(self, requests):
           request = random.choice(requests)
           program = self.grammar.sample(request)
           features = self.featureExtractor.featuresOfProgram(program, request)
           # Feature extractor failure
           if features is None: return None
           else: return program, request, features

    def enumerateFrontiers(self, frontierSize, tasks,
                           CPUs=1, maximumFrontier=None):
        with timing("Evaluated recognition model"):
            grammars = {}
            for task in tasks:
                features = self.featureExtractor.featuresOfTask(task)
                variables, productions = self(features)
                grammars[task] = Grammar(variables.data[0],
                                         [ (productions.data[k],t,p)
                                           for k,(_,t,p) in enumerate(self.grammar.productions) ])
        
        return callCompiled(enumerateFrontiers,
                            grammars, frontierSize, tasks,
                            CPUs = CPUs, maximumFrontier = maximumFrontier)

class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, _=None,
                 tasks=None,
                 cuda=False,
                 # what are the symbols that can occur in the inputs and outputs
                 lexicon=None,
                 # how many hidden units
                 H=32,
                 # dimensionality of the output
                 #O=32,
                 # Should the recurrent units be bidirectional?
                 bidirectional=False,
                 # modify examples before forward (to turn them into iterables of lexicon)
                 tokenize=lambda x,l:x):
        super(RecurrentFeatureExtractor, self).__init__()

        assert tasks is not None, "You must provide a list of all of the tasks, both those that have been hit and those that have not been hit. Input examples are sampled from these tasks."

        # maps from a requesting type to all of the inputs that we ever saw that request
        self.requestToInputs = {tp: [ map(fst, t.examples) for t in tasks if t.request == tp ]
                                for tp in {t.request for t in tasks } }

        assert lexicon
        lexicon += ["STARTING","ENDING","MIDDLE"]
        encoder = nn.Embedding(len(lexicon), H)
        if cuda:
            encoder = encoder.cuda()
        self.encoder = encoder

        self.H = H
        #self.O = O
        self.bidirectional = bidirectional
        self.tokenize = tokenize

        layers = 1

        # self.inputModel = nn.GRU(H, H, layers, bidirectional = bidirectional)
        # self.outputModel = nn.GRU(H, H, layers, bidirectional = bidirectional)

        model = nn.GRU(H, H, layers, bidirectional = bidirectional)
        if cuda:
            model = model.cuda()
        self.model = model

        #self.outputLayer = nn.Linear(H,O)

        self.use_cuda = cuda
        self.lexicon = lexicon
        self.symbolToIndex = {symbol: index for index, symbol in enumerate(lexicon) }
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endingIndex = self.symbolToIndex["ENDING"]
        self.middleIndex = self.symbolToIndex["MIDDLE"]

    @property
    def outputDimensionality(self): return self.H

    def observationEmbedding(self, x):
        x = [self.startingIndex] + [ self.symbolToIndex[s] for s in x ] + [self.endingIndex]
        x = variable(x, cuda=self.use_cuda)
        x = self.encoder(x)
        return x

    def packExamples(self, examples):
        """IMPORTANT! xs must be sorted in decreasing order of size because pytorch is stupid"""
        sizes = [ len(x) + len(y) + 3 for (x,),y in examples ]
        maximumSize = max(sizes)
        xs = [ [self.startingIndex] + \
               [ self.symbolToIndex[s] for s in x ] + \
               [self.middleIndex] + \
               [ self.symbolToIndex[s] for s in y ] + \
               [self.endingIndex]*(maximumSize - len(y) - len(x) + 1 - 3)
               for ((x,),y) in examples ]

        x = variable(xs, cuda=self.use_cuda)
        x = self.encoder(x)
        # x: (batch size, maximum length, E)
        x = x.permute(1,0,2)
        # x: TxBxE
        x = pack_padded_sequence(x, sizes)
        return x, sizes

    # def readInput(self, x):
    #     x = self.observationEmbedding(x)
    #     # x: (size of input)x(size of encoding)
    #     output, hidden = self.inputModel(x.unsqueeze(1))
    #     return hidden
        
    # def readOutput(self, y, hiddenStates):
    #     y = self.observationEmbedding(y)
    #     output, hidden = self.outputModel(y.unsqueeze(1),hiddenStates)
    #     if self.bidirectional:
    #         hidden,_ = hidden.max(dim = 0)
    #     else: hidden = hidden.squeeze(0)
    #     hidden = hidden.squeeze(0)
    #     return hidden            

    def examplesEncoding(self, examples):
        examples = sorted(examples, key = lambda ((x,),y): len(x) + len(y), reverse = True)
        x,sizes = self.packExamples(examples)
        outputs, hidden = self.model(x)
        #outputs, sizes = pad_packed_sequence(outputs)
        # I don't know whether to return the final output or the final hidden activations...
        return hidden[0,:,:] + hidden[1,:,:]
        
    def forward(self, examples):
        tokenized = self.tokenize(examples, self.lexicon)
        e = self.examplesEncoding(tokenized)
        # max pool
        e,_ = e.max(dim = 0)
        return e
        #return self.outputLayer(e).clamp(min = 0)

    def featuresOfTask(self, t): return self(t.examples)
    def featuresOfProgram(self, p, tp):
        candidateInputs = list(self.requestToInputs[tp])
        # Loop over the inputs in a random order and pick the first one that doesn't generate an exception
        random.shuffle(candidateInputs)
        for xss in candidateInputs:
            try:
                ys = [ p.runWithArguments(xs) for xs in xss ]
            except: continue
            return self(zip(xss,ys))
        return None
        
class MLPFeatureExtractor(nn.Module):
    def __init__(self, tasks, cuda=False, H=16):
        super(MLPFeatureExtractor, self).__init__()

        self.averages, self.deviations = RegressionTask.featureMeanAndStandardDeviation(tasks)
        self.tasks = tasks
        self.use_cuda = cuda

        self.outputDimensionality = H
        hidden = nn.Linear(len(self.averages), H)
        if cuda:
            hidden = hidden.cuda()
        self.hidden = hidden

    def featuresOfTask(self, t):
        f = variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(t.features) ], cuda=self.use_cuda).float()
        return self.hidden(f).clamp(min = 0)
    def featuresOfProgram(self, p, t):
        features = self._featuresOfProgram(p,t)
        if features is None: return None
        f = variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(features) ], cuda=self.use_cuda).float()
        return self.hidden(f).clamp(min = 0)
    
        

class HandCodedFeatureExtractor(object):
    def __init__(self, tasks, cuda=False):
        self.averages, self.deviations = RegressionTask.featureMeanAndStandardDeviation(tasks)
        self.outputDimensionality = len(self.averages)
        self.cuda = cuda
        self.tasks = tasks
    def featuresOfTask(self, t):
        return variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(t.features) ], cuda=self.cuda).float()
    def featuresOfProgram(self, p, t):
        features = self._featuresOfProgram(p,t)
        if features is None: return None
        return variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(features) ], cuda=self.cuda).float()
    
                

