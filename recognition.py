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

class GrammarNetwork(nn.Module):
    """Neural network that outputs a grammar"""
    def __init__(self, inputDimensionality, grammar):
        super(GrammarNetwork, self).__init__()
        self.logVariable = nn.Linear(inputDimensionality, 1)
        self.logProductions = nn.Linear(inputDimensionality, len(grammar))
        self.grammar = grammar
        
    def forward(self, x):
        """Takes as input inputDimensionality-dimensional vector and returns Grammar
        Tensor-valued probabilities"""
        logVariable = self.logVariable(x)
        logProductions = self.logProductions(x)
        return Grammar(logVariable,
                       [(logProductions[k].view(1), t, program)
                        for k, (_, t, program) in enumerate(self.grammar.productions)],
                       continuationType=self.grammar.continuationType)

class ContextualGrammarNetwork(nn.Module):
    """Like GrammarNetwork but ~contextual~"""
    def __init__(self, inputDimensionality, grammar):
        super(ContextualGrammarNetwork, self).__init__()
        
        self.grammar = grammar
        
        self.variableParent = GrammarNetwork(inputDimensionality, grammar)
        self.noParent = GrammarNetwork(inputDimensionality, grammar)
        self.library = {e: [GrammarNetwork(inputDimensionality, grammar)
                            for n in range(len(e.infer().functionArguments())) ]
                        for ei,e in enumerate(grammar.primitives) }
        # Explicitly register each of the library grammars
        for ei,e in enumerate(grammar.primitives):
            for n in range(len(e.infer().functionArguments())):
                self.add_module("Invention %d, argument %d"%(ei,n),
                                self.library[e][n])

    def forward(self,x):
        return ContextualGrammar(self.noParent(x), self.variableParent(x),
                                 {e: [g(x) for g in gs ]
                                  for e,gs in self.library.items() })

        

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

    def frontierKL(self, frontier):
        features = self.featureExtractor.featuresOfTask(frontier.task)
        g = self(features)
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        return - entry.program.logLikelihood(g)

    def frontierBiasOptimal(self, frontier):
        features = self.featureExtractor.featuresOfTask(frontier.task)
        if features is None: return None
        g = self(features)
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

    def train(self, frontiers, _=None, steps=None, lr=0.0001, topK=5, CPUs=1,
              timeout=None, helmholtzRatio=0., helmholtzBatch=5000):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        """
        assert (steps is not None) or (timeout is not None), \
            "Cannot train recognition model without either a bound on the number of epochs or bound on the training time"
        if steps is None: steps = 9999999
        
        requests = [frontier.task.request for frontier in frontiers]
        frontiers = [frontier.topK(topK).normalize()
                     for frontier in frontiers if not frontier.empty]

        # We replace each program in the frontier with its likelihoodSummary
        # This is because calculating likelihood summaries requires juggling types
        # And type stuff is expensive!
        frontiers = [self.replaceProgramsWithLikelihoodSummaries(f).normalize()
                     for f in frontiers]

        # Not sure why this ever happens
        if helmholtzRatio is None:
            helmholtzRatio = 0.

        eprint("Training a recognition model from %d frontiers, %d%% Helmholtz, feature extractor %s." % (
            len(frontiers), int(helmholtzRatio * 100), self.featureExtractor.__class__.__name__))

        # The number of Helmholtz samples that we generate at once
        # Should only affect performance and shouldn't affect anything else
        HELMHOLTZBATCH = helmholtzBatch
        helmholtzSamples = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        if timeout:
            start = time.time()

        with timing("Trained recognition model"):
            for i in range(1, steps + 1):
                if timeout and time.time() - start > timeout:
                    break
                losses = []

                if helmholtzRatio < 1.:
                    permutedFrontiers = list(frontiers)
                    random.shuffle(permutedFrontiers)
                else:
                    permutedFrontiers = [None]
                for frontier in permutedFrontiers:
                    # Randomly decide whether to sample from the generative
                    # model
                    doingHelmholtz = random.random() < helmholtzRatio
                    if doingHelmholtz:
                        if helmholtzSamples == []:
                            helmholtzSamples = \
                            list(self.sampleManyHelmholtz(requests,
                                                          HELMHOLTZBATCH,
                                                          CPUs))
                        if len(helmholtzSamples) == 0:
                            eprint(
                                "WARNING: Could not generate any Helmholtz samples. Disabling Helmholtz.")
                            helmholtzRatio = 0.
                            doingHelmholtz = False
                        else:
                            attempt = helmholtzSamples.pop()
                            if attempt is not None:
                                self.zero_grad()
                                loss = self.frontierKL(attempt)
                            else:
                                doingHelmholtz = False
                    if not doingHelmholtz:
                        if helmholtzRatio < 1.:
                            self.zero_grad()
                            loss = self.frontierKL(frontier)
                        else:
                            # Refuse to train on the frontiers
                            continue

                    if is_torch_invalid(loss):
                        if doingHelmholtz:
                            eprint("Invalid real-data loss!")
                        else:
                            eprint("Invalid Helmholtz loss!")
                    else:
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.data.tolist()[0])
                        if False:
                            if doingHelmholtz:
                                eprint(
                                    "\tHelmholtz data point loss:",
                                    loss.data.tolist()[0])
                            else:
                                eprint(
                                    "\tReal data point loss:",
                                    loss.data.tolist()[0])
                if (i == 1 or i % 10 == 0) and losses:
                    eprint("Epoch", i, "Loss", sum(losses) / len(losses))
                    gc.collect()

    def trainBiasOptimal(self, frontiers, helmholtzFrontiers, _=None,
                         steps=None, lr=0.0001, timeout=None, CPUs=None,
                         evaluationTimeout=0.001,
                         helmholtzRatio=0.5):
        assert (steps is not None) or (timeout is not None), \
            "Cannot train recognition model without either a bound on the number of epochs or bound on the training time"
        if steps is None: steps = 9999999
        # We replace each program in the frontier with its likelihoodSummary
        # This is because calculating likelihood summaries requires juggling types
        # And type stuff is expensive!
        frontiers = [self.replaceProgramsWithLikelihoodSummaries(f)
                     for f in frontiers]
        class HelmholtzEntry():
            def __init__(self, frontier, owner):
                self.request = frontier.task.request
                self.task = None
                self.programs = [e.program for e in frontier]
                self.frontier = owner.replaceProgramsWithLikelihoodSummaries(frontier)
                self.owner = owner

            def clear(self): self.task = None

            def calculateTask(self):
                assert self.task is None
                p = random.choice(self.programs)
                return self.owner.featureExtractor.taskOfProgram(p, self.request)

            def makeFrontier(self):
                assert self.task is not None
                f = Frontier(self.frontier.entries,
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
            HELMHOLTZBATCHSIZE = 500
            newTasks = \
             parallelMap(1,
                         lambda f: f.calculateTask(),
                        helmholtzFrontiers[helmholtzIndex[0]:helmholtzIndex[0] + HELMHOLTZBATCHSIZE])
            badIndices = []
            for i in range(helmholtzIndex[0], min(helmholtzIndex[0] + HELMHOLTZBATCHSIZE,
                                                  len(helmholtzFrontiers))):
                helmholtzFrontiers[i].task = newTasks[i - helmholtzIndex[0]]
                if helmholtzFrontiers[i].task is None: badIndices.append(i)
            # Permanently kill anything which failed to give a task
            for i in reversed(badIndices):
                assert helmholtzFrontiers[i].task is None
                del helmholtzFrontiers[i]
            
                
        eprint("Training bias optimal w/ %d Helmholtz frontiers"%len(helmholtzFrontiers))
        with timing("Precomputed a Helmholtz batch"): updateHelmholtzTasks()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        if timeout:
            start = time.time()

        frontiers = [f for f in frontiers if not f.empty ]

        with timing("Trained recognition model"):
            for i in range(1, steps + 1):
                if timeout and time.time() - start > timeout:
                    break
                losses = []
                descriptionLengths = []

                permutedFrontiers = list(frontiers)
                random.shuffle(permutedFrontiers)
                for frontier in permutedFrontiers:
                    dreaming = random.random() < helmholtzRatio
                    if dreaming:
                        frontier = getHelmholtz()
                    self.zero_grad()
                    loss = self.frontierBiasOptimal(frontier)
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
                if (i == 1 or i % 10 == 0) and losses:
                    eprint("Epoch", i, "Loss", sum(losses) / len(losses),
                           "\n\tvs MDL (w/o neural net)", sum(descriptionLengths) / len(descriptionLengths))
                    gc.collect()
                    


    def sampleHelmholtz(self, requests, statusUpdate=None, seed=None):
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        program = self.generativeModel.sample(request, maximumDepth=6, maxAttempts=100)
        if program is None:
            return None
        task = self.featureExtractor.taskOfProgram(program, request)
        #eprint("extracted features")

        if statusUpdate is not None:
            # eprint(statusUpdate, end='')
            flushEverything()
        if task is None:
            return None

        if hasattr(self.featureExtractor, 'lexicon'):
            if self.featureExtractor.tokenize(task.examples) is None:
                return None

        frontier = Frontier([FrontierEntry(program=program,
                                           logLikelihood=0., logPrior=0.)],
                            task=task)
        #eprint("replacing with likelihood summary")
        frontier = self.replaceProgramsWithLikelihoodSummaries(frontier)
        #eprint("successfully got a sample")
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

    def enumerateHelmholtz(self, requests, timeout, CPUs=1):
        if len(requests) > 1:
            return [f
                    for fs in \
                    parallelMap(CPUs, lambda request: (request, self.enumerateHelmholtz(request, timeout)),
                                requests)
                    for f in fs] 
        request = requests[0]
        frontier = {} # maps from (task key) > (task, program, ll)
        startTime = time.time()
        for lb in range(0,99):
            if time.time() - startTime > timeout: break
            for ll,_,e in self.generativeModel.enumeration(Context.EMPTY, [], request,
                                                           lowerBound=float(lb), upperBound=float(lb+1)):
                if time.time() - startTime > timeout: break

                task = self.featureExtractor.taskOfProgram(e, request)
                if task is None: continue

                k = self.featureExtractor.hashOfTask(task)
                if k not in frontier or frontier[k][-1] < ll:
                    frontier[k] = (task, e, ll)

        return [ self.replaceProgramsWithLikelihoodSummaries(Frontier([FrontierEntry(program,
                                                                                     logLikelihood=0.,
                                                                                     logPrior=0.)],
                                                                      task=task))
                 for t, program, _ in frontier.values() ]
                    
                    

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
            grammars = {task: self.grammarOfTask(task).untorch()
                        for task in tasks}

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
                 bidirectional=False):
        super(RecurrentFeatureExtractor, self).__init__()

        assert tasks is not None, "You must provide a list of all of the tasks, both those that have been hit and those that have not been hit. Input examples are sampled from these tasks."

        # maps from a requesting type to all of the inputs that we ever saw
        # that request
        self.requestToInputs = {
            tp: [
                list(
                    map(
                        fst,
                        t.examples)) for t in tasks if t.request == tp] for tp in {
                t.request for t in tasks}}

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
        candidateInputs = list(self.requestToInputs[tp])
        # Loop over the inputs in a random order and pick the first one that
        # doesn't generate an exception
        random.shuffle(candidateInputs)
        for xss in candidateInputs:
            ys = []

            for xs in xss:
                try:
                    y = runWithTimeout(lambda: p.runWithArguments(xs),0.01)
                except:
                    break

                ys.append(y)
            if len(ys) == len(xss):
                return Task("Helmholtz", tp, list(zip(xss,ys)))

        return None

    


class DummyFeatureExtractor(nn.Module):
    def __init__(self, tasks):
        super(DummyFeatureExtractor, self).__init__()
        self.outputDimensionality = 1
    def featuresOfTask(self, t):
        return variable([0.]).float()
    def taskOfProgram(self, p, t):
        return None

class ImageFeatureExtractor(nn.Module):
    def __init__(self, tasks):
        super(ImageFeatureExtractor, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5), stride=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8, 16, kernel_size=(5, 5)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # nn.Conv2d(16, 8, kernel_size=(3, 3), stride=2),
            # nn.Tanh(),
            #            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.outputDimensionality = 16

    def forward(self, v):
        w = int(len(v)**0.5)
        variabled = variable(v).float().view((w, w))
        variabled = torch.unsqueeze(variabled, 0)
        variabled = torch.unsqueeze(variabled, 0)
        y = self.l1(variabled)
        y = y.view((y.shape[0], -1))
        output = y  # self.fc(y)
        return output.view(-1)


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
    
    
