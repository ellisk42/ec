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


class RecognitionModel(nn.Module):
    def __init__(
            self,
            featureExtractor,
            grammar,
            hidden=[64],
            activation="relu",
            cuda=False):
        super(RecognitionModel, self).__init__()
        self.grammar = grammar
        self.use_cuda = cuda
        if cuda:
            self.cuda()

        self.featureExtractor = featureExtractor
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(
                    myParameter is parameter for myParameter in self.parameters())

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
            raise Exception('Unknown activation function ' + str(activation))

        self.logVariable = nn.Linear(inputDimensionality, 1)
        self.logProductions = nn.Linear(inputDimensionality, len(self.grammar))
        if cuda:
            self.logVariable = self.logVariable.cuda()
            self.logProductions = self.logProductions.cuda()

    def productionEmbedding(self):
        # Calculates the embedding 's of the primitives based on the last
        # weight layer
        w = self.logProductions._parameters['weight']
        # w: len(self.grammar) x E
        if self.use_cuda:
            w = w.cpu()
        e = dict({p: w[j, :].data.numpy()
                  for j, (t, l, p) in enumerate(self.grammar.productions)})
        e[Index(0)] = w[0, :].data.numpy()
        return e

    def taskEmbeddings(self, tasks):
        return {task: self.featureExtractor.featuresOfTask(task).data.numpy()
                for task in tasks}

    def forward(self, features):
        for layer in self.hiddenLayers:
            features = self.activation(layer(features))
        h = features
        # added the squeeze
        return self.logVariable(h), self.logProductions(h)

    def frontierKL(self, frontier):
        features = self.featureExtractor.featuresOfTask(frontier.task)
        variables, productions = self(features)
        g = Grammar(
            variables, [
                (productions[k].view(1), t, program) for k, (_, t, program) in enumerate(
                    self.grammar.productions)])
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        return - entry.program.logLikelihood(g)

    def replaceProgramsWithLikelihoodSummaries(self, frontier):
        return Frontier(
            [FrontierEntry(
                program=self.grammar.closedLikelihoodSummary(
                    frontier.task.request,
                    e.program),
                logLikelihood=e.logLikelihood,
                logPrior=e.logPrior) for e in frontier],
            task=frontier.task)

    def train(self, frontiers, _=None, steps=250, lr=0.0001, topK=1, CPUs=1,
              timeout=None, helmholtzRatio=0., helmholtzBatch=5000):
        """
        helmholtzRatio: What fraction of the training data should be forward samples from the generative model?
        """
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

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
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

    def sampleHelmholtz(self, requests, statusUpdate=None, seed=None):
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        #eprint("About to draw a sample")
        program = self.grammar.sample(request, maximumDepth=6, maxAttempts=100)
        #eprint("sample", program)
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
            #eprint("tokenizing...")
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
            grammars = {}
            for task in tasks:
                features = self.featureExtractor.featuresOfTask(task)
                variables, productions = self(features)
                # eprint("variable")
                # eprint(variables.data[0])
                # for k in range(len(self.grammar.productions)):
                #     eprint("production",productions.data[k])
                grammars[task] = Grammar(
                    variables.data.tolist()[0], [
                        (productions.data.tolist()[k], t, p) for k, (_, t, p) in enumerate(
                            self.grammar.productions)])

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
        if cuda:
            encoder = encoder.cuda()
        self.encoder = encoder

        self.H = H
        self.bidirectional = bidirectional

        layers = 1

        model = nn.GRU(H, H, layers, bidirectional=bidirectional)
        if cuda:
            model = model.cuda()
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
        if f is None:
            eprint(t)
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


class MLPFeatureExtractor(nn.Module):
    def __init__(self, tasks, cuda=False, H=16):
        super(MLPFeatureExtractor, self).__init__()

        self.averages, self.deviations = Task.featureMeanAndStandardDeviation(
            tasks)
        self.tasks = tasks
        self.use_cuda = cuda

        self.outputDimensionality = H
        hidden = nn.Linear(len(self.averages), H)
        if cuda:
            hidden = hidden.cuda()
        self.hidden = hidden

    def featuresOfTask(self, t):
        f = variable([(f - self.averages[j]) / self.deviations[j]
                      for j, f in enumerate(t.features)], cuda=self.use_cuda).float()
        return self.hidden(f).clamp(min=0)

    def featuresOfProgram(self, p, t):
        features = self._featuresOfProgram(p, t)
        if features is None:
            return None
        f = variable([(f - self.averages[j]) / self.deviations[j]
                      for j, f in enumerate(features)], cuda=self.use_cuda).float()
        return self.hidden(f).clamp(min=0)


class HandCodedFeatureExtractor(object):
    def __init__(self, tasks, cuda=False):
        self.averages, self.deviations = Task.featureMeanAndStandardDeviation(
            tasks)
        self.outputDimensionality = len(self.averages)
        self.cuda = cuda
        self.tasks = tasks

    def featuresOfTask(self, t):
        return variable([(f - self.averages[j]) / self.deviations[j]
                         for j, f in enumerate(t.features)], cuda=self.cuda).float()

    def featuresOfProgram(self, p, t):
        features = self._featuresOfProgram(p, t)
        if features is None:
            return None
        return variable([(f - self.averages[j]) / self.deviations[j]
                         for j, f in enumerate(features)], cuda=self.cuda).float()


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
    def __init__(self, tasks, cuda=False):
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

    def featuresOfProgram(self, p, t):
        features = self._featuresOfProgram(p, t)
        if features is None:
            return None
        # return [(self.stringify(x), self.stringify(y)) for (x,y) in features]
        return [(list(feature),) for feature in features]
        # TODO: should possibly match


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

    """def enumerateFrontiers(self, tasks,
                           frontierSize=None, enumerationTimeout=None,
                           CPUs=1, maximumFrontier=None, evaluationTimeout=None):
                           """

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

import threading

def handleRecognitionServer(message, qi, qo):
    message = pickle.loads(message)
    qi.put(message)
    return pickle.dumps(qo.get())

def launchRecognitionServer():
    from multiprocessing import Process
    Process(target=_launchRecognitionServer,args=()).start()
def _launchRecognitionServer():
    import queue
    
    qi = queue.Queue()
    qo = queue.Queue()

    trainer = threading.Thread(target=training_thread, args=(qi,qo))
    trainer.start()
    runPythonServer(4599, handleRecognitionServer, threads=1, args=(qi,qo))
    
def training_thread(qi,qo):
    rm = None
    helmholtzRatio = 0.
    helmholtzBatch = 500
    featureExtractor = None
    optimizer = None
    DSL = None
    cuda = False
    lr=0.0001
    helmholtzSamples = []

    while True:
        try:
            message = qi.get(block=(rm is None))
            if message == "get":
                qo.put(rm)
                continue
            if "helmholtzRatio" in message:
                helmholtzRatio = message['helmholtzRatio']
            if "cuda" in message:
                cuda = message["cuda"]
            if "lr" in message:
                lr = message["lr"]
            if "featureExtractor" in message:
                featureExtractor = message['featureExtractor']
            if "DSL" in message:
                DSL = message["DSL"]
                rm = RecognitionModel(featureExtractor,
                                      DSL,
                                      hidden=[],
                                      cuda=cuda)
                optimizer = torch.optim.Adam(rm.parameters(), lr=lr, eps=1e-3, amsgrad=True)
                helmholtzSamples = []
            if "frontiers" in message:
                frontiers = [frontier.normalize()
                             for frontier in message["frontier"] if not frontier.empty ]
                frontiers = [rm.replaceProgramsWithLikelihoodSummaries(f).normalize()
                             for f in frontiers]
                requests = [frontier.task.request for frontier in frontiers]
                permutedFrontiers = []
            if "evaluate" in message:
                assert rm is not None
                tasks = message["evaluate"]
                with timing("Evaluated recognition model"):
                    grammars = {}
                    for task in tasks:
                        features = rm.featureExtractor.featuresOfTask(task)
                        variables, productions = rm(features)
                        grammars[task] = Grammar(
                            variables.data.tolist()[0], [
                                (productions.data.tolist()[k], t, p) for k, (_, t, p) in enumerate(
                                    rm.grammar.productions)])
                qo.put(grammars)
            else:
                qo.put("okay")
        except Empty:
            # Another training step
            if len(permutedFrontiers) == 0:
                permutedFrontiers = list(frontiers)
                random.shuffle(permutedFrontiers)
            frontier = permutedFrontiers.pop()
            
            # Randomly decide whether to sample from the generative
            # model
            doingHelmholtz = random.random() < helmholtzRatio
            if doingHelmholtz:
                if helmholtzSamples == []:
                    helmholtzSamples = \
                        list(rm.sampleManyHelmholtz(requests,
                                                    helmholtzBatch,
                                                    1))
                if len(helmholtzSamples) == 0:
                    eprint("WARNING: Could not generate any Helmholtz samples. Disabling Helmholtz.")
                    helmholtzRatio = 0.
                    doingHelmholtz = False
                else:
                    attempt = helmholtzSamples.pop()
                    if attempt is not None:
                        rm.zero_grad()
                        loss = rm.frontierKL(attempt)
                    else:
                        doingHelmholtz = False
            if not doingHelmholtz:
                if helmholtzRatio < 1.:
                    rm.zero_grad()
                    loss = rm.frontierKL(frontier)
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
