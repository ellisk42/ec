from enumeration import *
from fragmentGrammar import *
from grammar import *
from heapq import *
from utilities import eprint
# luke
from program import tokeniseProgram, untokeniseProgram, ParseFailure


import time
import gc
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
# luke
import json
import string
import copy


def variable(x, volatile=False, cuda=False):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)


def is_torch_not_a_number(v):
    """checks whether a tortured variable is nan"""
    v = v.data
    if not ((v == v)[0]):
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

class ContextualGrammarNetwork(nn.Module):
    """Like GrammarNetwork but ~contextual~"""
    def __init__(self, inputDimensionality, grammar):
        super(ContextualGrammarNetwork, self).__init__()
        
        # library now just contains a list of indicies which go with each primitive
        self.grammar = grammar
        self.library = {}
        idx = 0
        for prim in grammar.primitives:
            numberOfArguments = len(prim.infer().functionArguments())
            idx_list = list(range(idx, idx+numberOfArguments))
            self.library[prim] = idx_list
            idx += numberOfArguments


        # idx is 1 more than the number of things in library, and we need 2 more than number of things in library
        self.n_grammars = idx+1
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
        

class RecognitionModel(nn.Module):
    def __init__(self,featureExtractor,grammar,hidden=[64],activation="relu",cuda=False,contextual=False):
        super(RecognitionModel, self).__init__()
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
            self.grammarBuilder = ContextualGrammarNetwork(self.outputDimensionality, grammar)
        else:
            self.grammarBuilder = GrammarNetwork(self.outputDimensionality, grammar)
        
        self.grammar = ContextualGrammar.fromGrammar(grammar) if contextual else grammar
        self.generativeModel = grammar

        if cuda: self.cuda()

    def taskEmbeddings(self, tasks):
        return {task: self.featureExtractor.featuresOfTask(task).data.numpy()
                for task in tasks}

    def forward(self, features):
        """returns either a Grammar or a ContextualGrammar
        Takes as input the output of featureExtractor.featuresOfTask"""
        features = self._MLP(features)
        return self.grammarBuilder(features)

    def grammarOfTask(self, task):
        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None
        return self(features)

    def grammarLogProductionsOfTask(self, task):
        """Returns the grammar logits from non-contextual models."""

        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None

        if hasattr(self, 'hiddenLayers'):
            eprint("Found hiddenLayers, extracting hidden layers instead.")
            # Backward compatability with old checkpoints.
            for layer in self.hiddenLayers:
                features = self.activation(layer(features))
            # return features
            return self.noParent[1](features)
        else:
            features = self._MLP(features)

        if self.contextual:
            if hasattr(self.grammarBuilder, 'variableParent'):
                eprint("Found contextual model, extracting logProductions of no-parent model.")
                return self.grammarBuilder.variableParent.logProductions(features)
            else:
                eprint("Found contextual model, extracting logProductions of full network.")
                return self.grammarBuilder.network(features).view(-1)
        else:
            return self.grammarBuilder.logProductions(features)

    def grammarEntropyOfTask(self, task):
        """Returns the entropy of the grammar distribution from non-contextual models for a task."""
        grammarLogProductionsOfTask = self.grammarLogProductionsOfTask(task)

        if grammarLogProductionsOfTask is None: return None

        if hasattr(self, 'entropy'):
            return self.entropy(grammarLogProductionsOfTask)
        else:
            e = Entropy()
            return e(grammarLogProductionsOfTask)

    def taskGrammarLogProductions(self, tasks):
        return {task: self.grammarLogProductionsOfTask(task).data.numpy()
                for task in tasks}

    def taskGrammarEntropies(self, tasks):
        return {task: self.grammarEntropyOfTask(task).data.numpy()
                for task in tasks}

    def frontierKL(self, frontier):
        features = self.featureExtractor.featuresOfTask(frontier.task)
        g = self(features)
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        return - entry.program.logLikelihood(g)

    def frontierBiasOptimal(self, frontier):
        g = self.grammarOfTask(frontier.task)
        if g is None: return None
        l = [entry.program.logLikelihood(g)
             for entry in frontier]
        l = torch.stack(l,1).squeeze(0)
        l = l.max(0)[0]
        l = l.unsqueeze(0)
        return -l

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
              biasOptimal=None,
              defaultRequest=None):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        helmholtzFrontiers: Frontiers from programs enumerated from generative model (optional)
        If helmholtzFrontiers is not provided then we will sample programs during training
        """
        assert (steps is not None) or (timeout is not None), \
            "Cannot train recognition model without either a bound on the number of epochs or bound on the training time"
        if steps is None: steps = 9999999
        if biasOptimal is None: biasOptimal = len(helmholtzFrontiers) > 0
        
        requests = [frontier.task.request for frontier in frontiers]
        if len(requests) == 0 and helmholtzRatio > 0 and len(helmholtzFrontiers) == 0:
            assert defaultRequest is not None, "You are trying to random Helmholtz training, but don't have any frontiers. Therefore we would not know the type of the program to sample. Try specifying defaultRequest=..."
            requests = [defaultRequest]
        frontiers = [frontier.topK(topK).normalize()
                     for frontier in frontiers if not frontier.empty]

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
            if updateCPUs > 1: eprint("Updating Helmholtz tasks with",updateCPUs,"CPUs")
            
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
                
            newTasks = \
             parallelMap(updateCPUs,
                         lambda f: f.calculateTask(),
                         helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + helmholtzBatch],
                         seedRandom=True)
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

        eprint("Training a recognition model from %d frontiers, %d%% Helmholtz, feature extractor %s." % (
            len(frontiers), int(helmholtzRatio * 100), self.featureExtractor.__class__.__name__))
        eprint("Got %d Helmholtz frontiers - random Helmholtz training? : %s"%(
            len(helmholtzFrontiers), len(helmholtzFrontiers) == 0))
        eprint("Contextual?",self.contextual)
        eprint("Bias optimal?",biasOptimal)

        # The number of Helmholtz samples that we generate at once
        # Should only affect performance and shouldn't affect anything else
        helmholtzSamples = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        start = time.time()
        losses, descriptionLengths, realLosses, dreamLosses, realMDL, dreamMDL = [], [], [], [], [], []
        for i in range(1, steps + 1):
            if timeout and time.time() - start > timeout:
                break

            if helmholtzRatio < 1.:
                permutedFrontiers = list(frontiers)
                random.shuffle(permutedFrontiers)
            else:
                permutedFrontiers = [None]
            for frontier in permutedFrontiers:
                # Randomly decide whether to sample from the generative model
                dreaming = random.random() < helmholtzRatio
                if dreaming: frontier = getHelmholtz()
                self.zero_grad()
                loss = self.frontierBiasOptimal(frontier) if biasOptimal else self.frontierKL(frontier)
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
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data.tolist()[0])
                    descriptionLengths.append(min(-e.logPrior for e in frontier))
                    if dreaming:
                        dreamLosses.append(losses[-1])
                        dreamMDL.append(descriptionLengths[-1])
                    else:
                        realLosses.append(losses[-1])
                        realMDL.append(descriptionLengths[-1])
                        
            if (i == 1 or i % 10 == 0) and losses:
                eprint("Epoch", i, "Loss", mean(losses))
                if realLosses and dreamLosses:
                    eprint("\t\t(real loss): ", mean(realLosses), "\t(dream loss):", mean(dreamLosses))
                eprint("\tvs MDL (w/o neural net)", mean(descriptionLengths))
                if realMDL and dreamMDL:
                    eprint("\t\t(real MDL): ", mean(realMDL), "\t(dream MDL):", mean(dreamMDL))
                losses, descriptionLengths, realLosses, dreamLosses, realMDL, dreamMDL = [], [], [], [], [], []
                gc.collect()
        
        eprint("Trained recognition model in",time.time() - start,"seconds")

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
        samples = parallelMap(
            1,
            lambda n: self.sampleHelmholtz(requests,
                                           statusUpdate='.' if n % frequency == 0 else None,
                                           seed=startingSeed + n),
            range(N))
        eprint()
        flushEverything()
        samples = [z for z in samples if z is not None]
        eprint()
        eprint("Got %d/%d valid samples." % (len(samples), N))
        flushEverything()

        return samples

    def enumerateFrontiers(self,
                           tasks,
                           likelihoodModel,
                           solver=None,
                           enumerationTimeout=None,
                           testing=False,
                           CPUs=1,
                           frontierSize=None,
                           maximumFrontier=None,
                           evaluationTimeout=None):
        with timing("Evaluated recognition model"):
            grammars = {task: self.grammarOfTask(task)
                        for task in tasks}
            #untorch seperately to make sure you filter out None grammars
            grammars = {task: grammar.untorch() for task, grammar in grammars.items() if grammar is not None}

        return multicoreEnumeration(grammars, tasks, likelihoodModel,
                                    solver=solver,
                                    testing=testing,
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
            0).data.numpy() for s in self.lexicon if not (s in self.specialSymbols)}

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
                
            
    


class DummyFeatureExtractor(nn.Module):
    def __init__(self, tasks):
        super(DummyFeatureExtractor, self).__init__()
        self.outputDimensionality = 1
        self.recomputeTasks = False
    def featuresOfTask(self, t):
        return variable([0.]).float()
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
        self.outputDimensionality = z_dim*outputImageDimensionality*outputImageDimensionality

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


# TODO
class NewRecognitionModel(nn.Module):
    def __init__(
            self,
            featureExtractor,
            grammar,
            vocabulary=string.printable,
            cuda=False):
        import pinn
        
        super(NewRecognitionModel, self).__init__()
        self.grammar = grammar
        self.use_cuda = cuda
        if cuda:
            self.cuda()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            # Torch sometimes segfaults in multithreaded mode...
            pass
            # torch.set_num_threads(1)

        self.featureExtractor = featureExtractor

        # TODO: modify for regex using pinn
        self.network = pinn.RobustFill(
            input_vocabularies=[vocabulary],
            target_vocabulary=self.getTargetVocabulary(grammar)
        )
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(
                    myParameter is parameter for myParameter in self.parameters())

    # Maybe can kill lambdas completely since they're deterministic
    def getTargetVocabulary(self, grammar):
        return ["(_lambda", ")_lambda", "(", ")"] + \
            ["$" + str(i) for i in range(10)] + \
            [str(p) for p in grammar.primitives]

    def updateGrammar(self, grammar):
        # self.network.set_target_vocabulary(self.getTargetVocabulary(grammar))
        self.network = self.network.with_target_vocabulary(self.getTargetVocabulary(
            grammar))  # Annoying to have to do this, but it's okay - why?

    def train(self, frontiers, _=None, steps=250, lr=0.001, topK=1, CPUs=1,
              helmholtzRatio=0.):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        """
        requests = [frontier.task.request for frontier in frontiers]

        frontiers = [frontier.topK(topK).normalize()
                     for frontier in frontiers if not frontier.empty]

        # Not sure why this ever happens
        if helmholtzRatio is None:
            helmholtzRatio = 0.

        eprint(
            "Training recognition model from %d frontiers, %d%% Helmholtz." %
            (len(frontiers), int(
                helmholtzRatio * 100)))

        #HELMHOLTZBATCH = 250
        HELMHOLTZBATCH = 200
        #recommended total training of 250000

        eprint("trying to cuda, HELMHOLTZBATCH is", HELMHOLTZBATCH)
        self.network.cuda()

        with timing("Trained recognition model"):
            avgLoss = None
            avgPermutedLoss = None

            for i in range(1, steps + 1):
                eprint("step", i, "out of", steps + 1)
                if helmholtzRatio < 1.:
                    permutedFrontiers = list(frontiers)
                    random.shuffle(permutedFrontiers)
                    eprint("not implemented")
                    assert False
                    # eprint("frontiers:")
                    # eprint(frontiers)
                    # eprint("permutedFrontiers:")
                    # eprint(permutedFrontiers)

                else:
                    permutedFrontiers = [None]
                frontier_num = 0
                for frontier in permutedFrontiers:
                    eprint(
                        "frontier num",
                        frontier_num,
                        "out of",
                        len(permutedFrontiers))
                    frontier_num += 1
                    # Randomly decide whether to sample from the generative model
                    # for now, only helmholtz
                    assert helmholtzRatio >= 1
                    doingHelmholtz = random.random() < helmholtzRatio
                    if doingHelmholtz:
                        networkInputs = self.helmholtzNetworkInputs(
                            requests, HELMHOLTZBATCH, CPUs)
                        #eprint("networkInputs[0]:", networkInputs[0])
                        #eprint("networkInputs[1]:", networkInputs[1])
                        loss = self.step(*networkInputs)
                    if not doingHelmholtz:
                        if helmholtzRatio < 1.:
                            # placeholder for now
                            # self.zero_grad()
                            # loss = self.frontierKL(frontier)
                            # fix this later
                            loss = 0
                            eprint(
                                "helmholtz ratio is less than 1. for now only works for ratio = 1")
                            pass
                        else:
                            # Refuse to train on the frontiers
                            pass

                if (i == 1 or i % 5 == 0):
                    # networkInputs = self.helmholtzNetworkInputs(requests, HELMHOLTZBATCH, CPUs)
                    # loss, permutedLoss = self.getCurrentLoss(*networkInputs)
                    avgLoss = (
                        0.9 *
                        avgLoss +
                        0.1 *
                        loss) if avgLoss is not None else loss
                    # avgPermutedLoss = (0.9*avgPermutedLoss +
                    # 0.1*permutedLoss) if avgPermutedLoss is not None else
                    # permutedLoss

                    # inputInformation = avgPermutedLoss - avgLoss
                    eprint("Epoch %3d Loss %2.2f" % (i, avgLoss))
                    gc.collect()

    # def helmholtsNetworkInputs(self, requests, batchSize, CPUs):
    #     helmholtzSamples = self.sampleManyHelmholtz(requests, batchSize, CPUs)
    #     helmholtzSamples = [x for x in helmholtzSamples if x is not None]

    #     inputss = [[_in for (_in, _out) in features] for (program, request, features) in helmholtzSamples]
    #     outputss = [[_out for (_in, _out) in features] for (program, request, features) in helmholtzSamples]
    # targets = [tokeniseProgram(program) for (program, request, features) in
    # helmholtzSamples]

    #     #For now, just concat input + output
    # joinedInputsOutputs = [[inputss[i][j] + outputss[i][j] for j in
    # range(len(inputss[i]))] for i in range(len(inputss))]

    #     #Filter to length <= 30
    #     valid_idxs = [i for i in range(len(targets)) if len(targets[i])<=30 and all(len(example)<=30 for example in joinedInputsOutputs[i])]
    #     batchInputsOutputs = [joinedInputsOutputs[i] for i in valid_idxs]
    #     batchTargets = [targets[i] for i in valid_idxs]

    #     return batchInputsOutputs, batchTargets

    def helmholtzNetworkInputs(self, requests, batchSize, CPUs):
        helmholtzSamples = self.sampleManyHelmholtz(requests, batchSize, CPUs)
        helmholtzSamples = [
            x for x in helmholtzSamples if x is not None]  # good

        # TODO: modify for regexes
        # inputss = [[_in for (_in, _out) in features] for (program, request,
        # features) in helmholtzSamples]

        # TODO: may need to remove the tuple thing - yeah
        outputss = [[_out for _out in features]
                    for (program, request, features) in helmholtzSamples]
        targets = [
            tokeniseProgram(program) for (
                program,
                request,
                features) in helmholtzSamples]
        # For now, just concat input + output
        # joinedInputsOutputs = [[inputss[i][j] + outputss[i][j] for j in
        # range(len(inputss[i]))] for i in range(len(inputss))]

        # Filter to length <= 30
        valid_idxs = [i for i in range(len(targets)) if
                      len(targets[i]) <= 30 and
                      all(len(example[0]) <= 30 for example in outputss[i])]


        # batchInputsOutputs = [joinedInputsOutputs[i] for i in valid_idxs]
        batchOutputs = [outputss[i] for i in valid_idxs]
        batchTargets = [targets[i] for i in valid_idxs]

        return batchOutputs, batchTargets

    # deprecated, does not work
    def shuffledNetworkInputs(self, requests, batchSize, CPUs):
        batchInputs, batchOutputs, batchTargets = self.helmholtzNetworkInputs(
            requests, batchSize, CPUs)
        permutedBatchTargets = batchTargets[:]
        random.shuffle(permutedBatchTargets)
        # why the shuffle for only the targets??
        return batchInputs, batchOutputs, permutedBatchTargets

    def step(self, *networkInputs):
        #eprint("networkInputs:")
        #eprint(*networkInputs)
        #assert False
        score = self.network.optimiser_step(*networkInputs)
        loss = -score
        return loss

    # def getCurrentLoss(self, batchInputsOutputs, batchTargets):
    #     score = self.network.score(batchInputsOutputs, batchTargets)
    #     loss = -score

    #     permutedBatchTargets = batchTargets[:]
    #     random.shuffle(permutedBatchTargets)
    #     permutedScore = self.network.score(batchInputsOutputs, permutedBatchTargets)
    #     permutedLoss = -permutedScore

    #     return loss, permutedLoss

    # TODO: I dont think this is used, it is currently deprecated
    def getCurrentLoss(self, batchInputs, batchOutputs, batchTargets):
        score = self.network.score(batchInputs, batchOutputs, batchTargets)
        loss = -score

        permutedBatchTargets = batchTargets[:]
        random.shuffle(permutedBatchTargets)
        permutedScore = self.network.score(
            batchInputs, batchOutputs, permutedBatchTargets)
        permutedLoss = -permutedScore

        return loss, permutedLoss

    def sampleHelmholtz(self, requests):
        request = random.choice(requests)
        # may want to weigh less likely programs more heavily
        program = self.grammar.sample(request)
        #TODO: use a very simple grammar here, to check that it's working

        #try:
        #program = self.grammar.sample(request, maximumDepth=6, maxAttempts=100)
        #eprint("sampled training program:")
        #eprint(program)
        # >>> Increase maxDepth, might actually make sampling faster
        # >>> Call out to pypy
        features = self.featureExtractor.featuresOfProgram(program, request)
        #eprint("features_outer:")
        #eprint(features)
        # Feature extractor failure
        if features is None:
            return None
        else:
            return program, request, features

    def sampleManyHelmholtz(self, requests, N, CPUs):  # >>> callCompiled
        helmholtzSamples = parallelMap(
            CPUs, lambda _: self.sampleHelmholtz(requests), range(N))
        return helmholtzSamples

    def enumerateFrontiers(self,
                           tasks,
                           likelihoodModel,
                           solver=None,
                           frontierSize=None,
                           enumerationTimeout=None,
                           CPUs=1,
                           maximumFrontier=None,
                           evaluationTimeout=None):
        # need to encorporate likelihood model, solver

        tasks_features = []
        for task in tasks:
            # eprint("Getting proposals for task", task)
            features = self.featureExtractor.featuresOfTask(task)
            # features = [(input, output) for (input, output) in features if len(input[0])<=30 and len(output)<=30]
            # np.random.shuffle(features)

            # had to change the line below for python 3
            # TODO: modify for input output for regexes.

            # TODO: may need to fix this
            features = sorted(features, key=lambda out: len(out[0])**2)
            tasks_features.append((task, features))

            # proposals_scores[task] = []
            # for i in range(1):
            #     inputs = [input[0] for (input, output) in features[:4]]
            #     outputs = [output for (input, output) in features[:4]]
            #     samples, scores = self.network.sampleAndScore([inputs]*500, [outputs]*500, nRepeats=100)
            #     proposals_scores[task].extend(list(set(
            #         (tuple(samples[i]), scores[i]) for i in range(len(samples))
            #     )))

        # network = copy.deepcopy(self.network).cpu() #to send to workers
        network = copy.deepcopy(self.network)
        assert network is not None

        torch.set_default_tensor_type('torch.FloatTensor')
        # print(type(network.input_encoder_init[0].data))
        # self.network.float()
        # print(type(self.network.input_encoder_init[0].data))
        # network = self.network.float()
        # print(type(network.input_encoder_init[0].data))
        # print(self.network.input_encoder_init[0])
        # print(type(self.network.input_encoder_init[0].float()))
        # print(self.network.input_encoder_init[0].float())
        # raise Exception()
        # network = self.network
        assert network is not None
        # TODO
        # Can't callcompiled because program.Primitive doesn't have the right
        # globals
        x = enumerateNetwork(
            network, tasks_features, likelihoodModel, solver=solver,
            frontierSize=frontierSize, enumerationTimeout=enumerationTimeout,
            CPUs=CPUs, maximumFrontier=maximumFrontier,
            evaluationTimeout=evaluationTimeout)

        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        return x

        # return evaluateProposals( #Can't callcompiled because program.Primitive doesn't have the right globals
        #                     proposals_scores, tasks,
        #                     frontierSize = frontierSize, enumerationTimeout=enumerationTimeout,
        #                     CPUs=CPUs, maximumFrontier=maximumFrontier,
        #                     evaluationTimeout=evaluationTimeout)
        # return callCompiled(evaluateProposals,
        #                     proposals_scores, tasks,
        #                     frontierSize = frontierSize, enumerationTimeout=enumerationTimeout,
        #                     CPUs=CPUs, maximumFrontier=maximumFrontier,
        #                     evaluationTimeout=evaluationTimeout)


               
def helmholtzEnumeration(g, request, inputs, timeout, _=None,
                         special=None, evaluationTimeout=None):
    """Returns json (as text)"""
    import json

    message = {"request": request.json(),
               "timeout": timeout,
               "DSL": g.json(),
               "extras": inputs}
    if evaluationTimeout: message["evaluationTimeout"] = evaluationTimeout
    if special: message["special"] = special
    message = json.dumps(message)
    with open('/tmp/hm','w') as handle:
        handle.write(message)
    try:
        process = subprocess.Popen("./helmholtz",
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        response, error = process.communicate(bytes(message, encoding="utf-8"))
    except OSError as exc:
        raise exc
    return response

def backgroundHelmholtzEnumeration(tasks, g, timeout, _=None,
                                   special=None, evaluationTimeout=None):
    requests = list({t.request for t in tasks})
    inputs = {r: list({tuplify(xs)
                       for t in tasks if t.request == r
                       for xs,y in t.examples })
              for r in requests }
    workers = Pool(len(requests))
    promises = [workers.apply_async(helmholtzEnumeration,
                                    args=(g,r,inputs[r],float(timeout)),
                                    kwds={'special': special,
                                          'evaluationTimeout': evaluationTimeout})
                for r in requests ]
    def get():
        results = [p.get() for p in promises]
        frontiers = []
        with timing("(Helmholtz enumeration) Decoded json into frontiers"):
            for request, result in zip(requests, results):
                response = json.loads(result.decode("utf-8"))
                for b, entry in enumerate(response):
                    frontiers.append(Frontier([FrontierEntry(program=Program.parse(p),
                                                             logPrior=entry["ll"],
                                                             logLikelihood=0.)
                                               for p in entry["programs"] ],
                                              task=Task(str(b),
                                                        request,
                                                        [])))
        eprint("Total number of Helmholtz frontiers:",len(frontiers))
        return frontiers
    return get
            
if __name__ == "__main__":
    from arithmeticPrimitives import *
    g = Grammar.uniform([k1,k0,addition,subtraction,multiplication])
    frontiers = helmholtzEnumeration(g,
                         arrow(tint,tint),
                         [[0],[1],[2]],
                         10.)
    eprint("average frontier size",mean(len(f.entries) for f in frontiers ))
    f = DummyFeatureExtractor([])
    r = RecognitionModel(f, g, hidden=[], contextual=True)
    r.trainBiasOptimal(frontiers, frontiers, steps=70)
    g = r.grammarOfTask(frontiers[0].task).untorch()
    frontiers = helmholtzEnumeration(g,
                         arrow(tint,tint),
                         [[0],[1],[2]],
                         10.)
    for f in frontiers:
        eprint(f.summarizeFull())
    eprint("average frontier size",mean(len(f.entries) for f in frontiers ))
    
    
