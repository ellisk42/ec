from enumeration import *
from fragmentGrammar import *
from grammar import *
from heapq import *
from utilities import eprint

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np


def variable(x, volatile=False, cuda=False):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)


class DRNN(nn.Module):
    def __init__(self, grammar, featureExtractor, hidden=64, cuda=False):
        super(DRNN, self).__init__()

        self.featureExtractor = featureExtractor
        # Converts the output of the feature extractor into the
        # initial hidden state of the parent rnn
        self.featureExtractor2parentH = \
            nn.Linear(featureExtractor.outputDimensionality, hidden)
        self.featureExtractor2parentC = \
            nn.Linear(featureExtractor.outputDimensionality, hidden)

        self.grammar = grammar
        self.production2index = \
            {p: j+1 for j, (_, _, p) in enumerate(grammar.productions)}
        self.production2index[Index(0)] = 0
        self.index2production = [Index(0)] + [ p for _,_,p in grammar.productions ]

        self.parent = nn.LSTM(hidden, hidden)
        self.sibling = nn.LSTM(hidden, hidden)

        self.encoder = nn.Embedding(len(grammar) + 1, hidden)

        self.siblingPrediction = nn.Linear(hidden, hidden, bias = False)
        self.parentPrediction = nn.Linear(hidden, hidden, bias = False)
        self.prediction = nn.Linear(hidden, len(grammar) + 1, bias = False)

        # todo: do I include the cell state?
        self.defaultSibling = variable(torch.Tensor(hidden).float()).view(1,1,-1)

        self.defaultParent = variable(torch.Tensor(hidden).float()).view(1,1,-1)

    def embedProduction(self,p):
        if p.isIndex: p = Index(0)
        return self.encoder(variable([self.production2index[p]]))

    def indexProduction(self,p):
        if p.isIndex: p = Index(0)
        return variable([self.production2index[p]])

    def predictionFromHidden(self, parent, sibling, alternatives = None):
        """Takes the parent and sibling hidden vectors, optionally with a set of alternatives; returns logits"""
        parent = parent[0] if parent else self.defaultParent
        sibling = sibling[0] if sibling else self.defaultSibling

        predictive = F.tanh(self.parentPrediction(parent) + self.siblingPrediction(sibling))

        r = self.prediction(predictive).view(-1)
        if alternatives is not None:
            haveVariables = any(a.isIndex for a in alternatives)
            mask = variable([ float(int((p in alternatives) or \
                                        (p.isIndex and haveVariables)))
                              for p in self.index2production ]).float().log()
            r += mask
        return F.log_softmax(r)

    def initialParent(self, task):
        features = self.featureExtractor.featuresOfTask(task)
        return (self.featureExtractor2parentH(features).view(1,1,-1),
                self.featureExtractor2parentC(features).view(1,1,-1))

    def updateParent(self, parent, symbol):
        embedding = self.embedProduction(symbol)
        _, h = self.parent(embedding.view(1,1,-1), parent)
        return h

    def updateSibling(self, sibling, symbol):
        embedding = self.embedProduction(symbol)
        _, h = self.sibling(embedding.view(1,1,-1), sibling)
        return h

    def singlePredictionLoss(self, prediction, target):
        return F.nll_loss(prediction.view(1,-1),
                          self.indexProduction(target))

    def programLoss(self, program, task):
        request = task.request
        parent = self.initialParent(task)
        context, root, loss = self._programLoss(request, program,
                                                parent = parent)
        return loss

    def _programLoss(self, request, program, _ = None,
                    parent = None, sibling = None,
                    context = None, environment = []):
        """Returns context, root, loss"""
        if context is None: context = Context.EMPTY

        if request.isArrow():
            assert isinstance(program,Abstraction)
            return self._programLoss(request.arguments[1],
                                    program.body,
                                    context = context,
                                    environment = [request.arguments[0]] + environment,
                                    parent = parent, sibling = sibling)

        f,xs = program.applicationParse()
        candidates = self.grammar.buildCandidates(request, context, environment,
                                                  normalize = False,
                                                  returnProbabilities = False,
                                                  returnTable = True)
        assert f in candidates
        alternatives = candidates.keys()

        _,tp,context = candidates[f]

        argumentTypes = tp.functionArguments()
        assert len(xs) == len(argumentTypes)

        L = self.singlePredictionLoss(self.predictionFromHidden(parent, sibling,
                                                                alternatives = alternatives),
                                      f)

        # Update ancestral rnn, which will be passed to the children
        if xs != []:
            parent = self.updateParent(parent, f)

        sibling = None

        for argumentType, argument in zip(argumentTypes, xs):
            argumentType = argumentType.apply(context)
            context, aroot, aL = self._programLoss(argumentType, argument,
                                                  context = context, environment = environment,
                                                  parent = parent, sibling = sibling)
            L += aL
            sibling = self.updateSibling(sibling, aroot)

        return (context,
                Index(0) if f.isIndex else f,
                L)

    def sample(self, task):
        context, root, p = self._sample(task.request,
                                        parent = self.initialParent(task))
        return p

    def _sample(self, request, _ = None,
               parent = None, sibling = None,
               context = None, environment = []):
        """Returns context , root , expression"""
        if context is None: context = Context.EMPTY

        if request.isArrow():
            context, root, expression = self._sample(request.arguments[1],
                                                    context = context,
                                                    parent = parent, sibling = sibling,
                                                    environment = [request.arguments[0]] + environment)
            return context, root, Abstraction(expression)

        candidates = self.grammar.buildCandidates(request, context, environment,
                                                  normalize = False,
                                                  returnProbabilities = False,
                                                  returnTable = True)

        alternatives = candidates.keys()
        prediction = self.predictionFromHidden(parent, sibling,
                                               alternatives = alternatives).exp()
        f = self.index2production[torch.multinomial(prediction, 1).data[0]]
        root = f
        if f.isIndex:
            # _Sample one of the variables uniformly
            assert f == Index(0)
            f = random.choice([ a for a in alternatives if a.isIndex ])

        _,tp,context = candidates[f]

        argumentTypes = tp.functionArguments()

        # Update ancestral rnn, which will be passed to the children
        if argumentTypes != []:
            parent = self.updateParent(parent, f)

        sibling = None

        for argumentType in argumentTypes:
            argumentType = argumentType.apply(context)
            context, childSymbol, a = self._sample(argumentType,
                                                  context = context, environment = environment,
                                                  parent = parent, sibling = sibling)
            f = Application(f, a)
            sibling = self.updateSibling(sibling, childSymbol)

        return context, root, f

    def enumeration(self, task, interval = 1.):
        request = task.request
        parent = self.initialParent(task)

        lowerBound = 0.
        while True:
            for ll,_,_,e in self._enumeration(request,
                                              lowerBound = lowerBound, upperBound = lowerBound + interval,
                                              parent = parent, sibling = None,
                                              context = Context.EMPTY, environment = []):
                yield ll,e
            lowerBound += interval

    def _enumeration(self, request, _ = None,
                     upperBound = None, lowerBound = None,
                     context = None, environment = None,
                     parent = None, sibling = None):
        """Generates log likelihood, context, root, expression"""
        if upperBound <= 0: return

        if request.isArrow():
            v = request.arguments[0]
            for l, newContext, r, b in self._enumeration(request.arguments[1],
                                                         context = context, environment = [v] + environment,
                                                         upperBound = upperBound,
                                                         lowerBound = lowerBound,
                                                         parent = parent, sibling = sibling):
                yield l, newContext, r, Abstraction(b)
            return

        candidates = self.grammar.buildCandidates(request, context, environment,
                                                  normalize = False,
                                                  returnProbabilities = False,
                                                  returnTable = True)

        alternatives = candidates.keys()
        numberOfVariables = sum(a.isIndex for a in alternatives)
        prediction = self.predictionFromHidden(parent, sibling,
                                               alternatives = alternatives).data
        prediction = dict(zip(self.index2production, prediction))
        # Update the candidates so that they now record what the
        # neural network thinks their likelihood should be
        for a in alternatives:
            if a.isIndex:
                ll = prediction[Index(0)] - math.log(numberOfVariables)
            else:
                ll = prediction[a]
            _,tp,newContext = candidates[a]
            candidates[a] = (ll,tp,newContext)

        for f,(ll,tp,newContext) in candidates.iteritems():
            mdl = -ll
            if not (mdl <= upperBound): continue

            argumentTypes = tp.functionArguments()
            if argumentTypes != []: newParent = self.updateParent(parent, f)
            else: newParent = parent

            root = Index(0) if f.isIndex else f

            for aL, aK, application in \
                self._enumerateApplication(f, argumentTypes,
                                           context = newContext, environment = environment,
                                           upperBound = upperBound + ll,
                                           lowerBound = lowerBound + ll,
                                           parent = newParent, sibling = None):
                yield aL + ll, aK, root, application

    def _enumerateApplication(self, f, xs, _ = None,
                              upperBound = None, lowerBound = None,
                              context = None, environment = None,
                              parent = None, sibling = None):
        if upperBound <= 0: return
        if xs == []:
            if lowerBound < 0. and 0. <= upperBound:
                yield 0., context, f
            return
        request = xs[0].apply(context)
        laterRequests = xs[1:]
        for aL, newContext, argumentRoot, argument in \
            self._enumeration(request,
                              context = context, environment = environment,
                              upperBound = upperBound, lowerBound = 0.,
                              parent = parent, sibling = sibling):
            newFunction = Application(f, argument)
            if laterRequests != []:
                newSibling = self.updateSibling(sibling, argumentRoot)
            else:
                newSibling = None
            for resultL, resultK, result in \
                self._enumerateApplication(newFunction, laterRequests,
                                           context = newContext, environment = environment,
                                           upperBound = upperBound + aL, lowerBound = lowerBound + aL,
                                           parent = parent, sibling = newSibling):
                yield resultL + aL, resultK, result


def _relu(x): return x.clamp(min=0)


class RecognitionModel(nn.Module):
    def __init__(self, featureExtractor, grammar, hidden=[64], activation="relu", cuda=False):
        super(RecognitionModel, self).__init__()
        self.grammar = grammar
        self.use_cuda = cuda
        if cuda: self.cuda()

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
            self.activation = _relu
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
            kl -= math.exp(entry.logPosterior) * g.logLikelihood(frontier.task.request, entry.program)
        return kl
    def HelmholtzKL(self, features, sample, tp):
        variables, productions = self(features)
        g = Grammar(variables, [(productions[k],t,program)
                                for k,(_,t,program) in enumerate(self.grammar.productions) ])
        return - g.logLikelihood(tp, sample)

    def train(self, frontiers, _=None, steps=250, lr=0.001, topK=1, CPUs=1,
              helmholtzRatio=0., helmholtzBatch=5000):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        """
        requests = [ frontier.task.request for frontier in frontiers ]
        frontiers = [ frontier.topK(topK).normalize() for frontier in frontiers if not frontier.empty ]

        # Not sure why this ever happens
        if helmholtzRatio is None:
            helmholtzRatio = 0.

        eprint("Training a recognition model from %d frontiers, %d%% Helmholtz, feature extractor %s."%(
            len(frontiers),
            int(helmholtzRatio*100),
            self.featureExtractor.__class__.__name__))

        # The number of Helmholtz samples that we generate at once
        # Should only affect performance and shouldn't affect anything else
        HELMHOLTZBATCH = helmholtzBatch
        helmholtzSamples = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        with timing("Trained recognition model"):
            for i in range(1,steps + 1):
                losses = []

                if helmholtzRatio < 1.:
                    permutedFrontiers = list(frontiers)
                    random.shuffle(permutedFrontiers)
                else:
                    permutedFrontiers = [None]
                for frontier in permutedFrontiers:
                    # Randomly decide whether to sample from the generative model
                    doingHelmholtz = random.random() < helmholtzRatio
                    if doingHelmholtz:
                        if helmholtzSamples == []:
                            helmholtzSamples = \
                            self.sampleManyHelmholtz(requests,
                                                     HELMHOLTZBATCH,
                                                     1) # TODO THIS IS A HACK
                        attempt = helmholtzSamples.pop()
                        if attempt is not None:
                            program, request, features = attempt
                            self.zero_grad()
                            loss = self.HelmholtzKL(features, program, request)
                        else: doingHelmholtz = False
                    if not doingHelmholtz:
                        if helmholtzRatio < 1.:
                            self.zero_grad()
                            loss = self.frontierKL(frontier)
                        else:
                            # Refuse to train on the frontiers
                            continue

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data[0])
                if i%50 == 0 and losses:
                    eprint("Epoch",i,"Loss",sum(losses)/len(losses))
                    gc.collect()


    def sampleHelmholtz(self, requests, statusUpdate = None):
       request = random.choice(requests)
       program = self.grammar.sample(request, maximumDepth = 6)
       features = self.featureExtractor.featuresOfProgram(program, request)
       if statusUpdate is not None:
           eprint(statusUpdate, end = '')
           flushEverything()
       # Feature extractor failure
       if features is None: return None
       else: return program, request, features

    def sampleManyHelmholtz(self, requests, N, CPUs):
        eprint("Sampling %d programs from the prior on %d CPUs..."%(N,CPUs))
        flushEverything()
        frequency = N/50
        samples = parallelMap(CPUs,
                              lambda n: self.sampleHelmholtz(requests,
                                                             statusUpdate = '.' if n%frequency == 0 else None),
                              range(N))
        eprint()
        flushEverything()
        try:
            self.featureExtractor.finish()
        except AttributeError:
            ()
        eprint()
        flushEverything()
        return samples

    def enumerateFrontiers(self, tasks, likelihoodModel,
                           solver=None,
                           enumerationTimeout=None,
                           CPUs=1, maximumFrontier=None, evaluationTimeout=None):
        with timing("Evaluated recognition model"):
            grammars = {}
            for task in tasks:
                features = self.featureExtractor.featuresOfTask(task)
                variables, productions = self(features)
                grammars[task] = Grammar(variables.data[0],
                                         [ (productions.data[k],t,p)
                                           for k,(_,t,p) in enumerate(self.grammar.productions) ])

        return multicoreEnumeration(grammars, tasks, likelihoodModel,
                                    solver=solver,
                                    enumerationTimeout=enumerationTimeout,
                                    CPUs=CPUs, maximumFrontier=maximumFrontier,
                                    evaluationTimeout=evaluationTimeout)

class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, _=None,
                 tasks=None,
                 cuda=False,
                 # what are the symbols that can occur in the inputs and outputs
                 lexicon=None,
                 # how many hidden units
                 H=32,
                 # Should the recurrent units be bidirectional?
                 bidirectional=False):
        super(RecurrentFeatureExtractor, self).__init__()

        assert tasks is not None, "You must provide a list of all of the tasks, both those that have been hit and those that have not been hit. Input examples are sampled from these tasks."

        # maps from a requesting type to all of the inputs that we ever saw that request
        self.requestToInputs = {tp: [ map(fst, t.examples) for t in tasks if t.request == tp ]
                                for tp in {t.request for t in tasks } }

        assert lexicon
        self.specialSymbols = [
            "STARTING", # start of entire sequence
            "ENDING", # ending of entire sequence
            "STARTOFOUTPUT", # begins the start of the output
            "ENDOFINPUT" # delimits the ending of an input - we might have multiple inputs
        ]
        lexicon += self.specialSymbols
        encoder = nn.Embedding(len(lexicon), H)
        if cuda:
            encoder = encoder.cuda()
        self.encoder = encoder

        self.H = H
        self.bidirectional = bidirectional

        layers = 1

        model = nn.GRU(H, H, layers, bidirectional = bidirectional)
        if cuda:
            model = model.cuda()
        self.model = model

        self.use_cuda = cuda
        self.lexicon = lexicon
        self.symbolToIndex = {symbol: index for index, symbol in enumerate(lexicon) }
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endingIndex = self.symbolToIndex["ENDING"]
        self.startOfOutputIndex = self.symbolToIndex["STARTOFOUTPUT"]
        self.endOfInputIndex = self.symbolToIndex["ENDOFINPUT"]

    @property
    def outputDimensionality(self): return self.H

    # modify examples before forward (to turn them into iterables of lexicon)
    # you should override this if needed
    def tokenize(self,x): return x

    def symbolEmbeddings(self):
        return {s: self.encoder(variable([self.symbolToIndex[s]])).squeeze(0).data.numpy()
                for s in self.lexicon if not (s in self.specialSymbols) }

    def packExamples(self, examples):
        """IMPORTANT! xs must be sorted in decreasing order of size because pytorch is stupid"""
        es = []
        sizes = []
        for xs,y in examples:
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
        for j,e in enumerate(es):
            es[j] += [self.endingIndex]*(m - len(e))

        x = variable(es, cuda=self.use_cuda)
        x = self.encoder(x)
        # x: (batch size, maximum length, E)
        x = x.permute(1,0,2)
        # x: TxBxE
        x = pack_padded_sequence(x, sizes)
        return x, sizes

    def examplesEncoding(self, examples):
        examples = sorted(examples, key = lambda (xs,y): sum(len(z)+1 for z in xs) + len(y), reverse = True)
        x,sizes = self.packExamples(examples)
        outputs, hidden = self.model(x)
        #outputs, sizes = pad_packed_sequence(outputs)
        # I don't know whether to return the final output or the final hidden activations...
        return hidden[0,:,:] + hidden[1,:,:]

    def forward(self, examples):
        tokenized = self.tokenize(examples)
        if not tokenized:
            return None
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

        self.averages, self.deviations = Task.featureMeanAndStandardDeviation(tasks)
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
        self.averages, self.deviations = Task.featureMeanAndStandardDeviation(tasks)
        self.outputDimensionality = len(self.averages)
        self.cuda = cuda
        self.tasks = tasks
    def featuresOfTask(self, t):
        return variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(t.features) ], cuda=self.cuda).float()
    def featuresOfProgram(self, p, t):
        features = self._featuresOfProgram(p,t)
        if features is None: return None
        return variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(features) ], cuda=self.cuda).float()


if __name__ == "__main__":
    from arithmeticPrimitives import *
    g = Grammar.uniform([addition, multiplication, real, k0, k1])

    observations = [ #"(* 0 REAL)",
        "(lambda (* $0 REAL))",
                    "(lambda (+ REAL (* $0 REAL)))"
    ]
    request = arrow(tint,tint)
    tasks = [Task(p,request,[],features = [j])
             for j,p in enumerate(observations) ]
    fe = HandCodedFeatureExtractor(tasks)
    observations = map(Program.parse, observations)

    m = DRNN(g, fe, hidden = 8)
    m.float()


    optimizer = torch.optim.Adam(m.parameters(), lr=0.0001)

    def take(n,g):
        r = []
        for x in g:
            r.append(x)
            if len(r) >= n: break
        return r

    for j in range(100000):
        m.zero_grad()
        l = None
        for t,p in zip(observations,tasks):
            _l = m.programLoss(t, p)
            if l is None: l = _l
            else: l += _l
        if j > 0 and j%150 == 0:
            print l.data[0]/len(observations)
            for t in tasks:
                print t
                print m.sample(t)
                print "enumeration:"
                for ll,e in sorted(take(5,m.enumeration(t)),reverse = True):
                    gt = m.programLoss(e,t).data[0]
                    print ll,gt,"\t",e
                print
        l.backward()
        optimizer.step()


