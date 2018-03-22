from utilities import eprint, exp, log, timing, valid
from task import Task
import random
import gc


class AllOrNothingLikelihoodModel:
    def __init__(self, timeout=None):
        self.timeout = timeout

    def score(self, program, task):
        logLikelihood = task.logLikelihood(program, self.timeout)
        return valid(logLikelihood), logLikelihood


class EuclideanLikelihoodModel:
    """Likelihood is based on Euclidean distance between features"""
    def __init__(self, featureExtractor, successCutoff=0.9):
        self.extract = featureExtractor
        self.successCutoff = successCutoff

    def score(self, program, task):
        taskFeat = self.extract.featuresOfTask(task)
        progFeat = self.extract.featuresOfProgram(program, task.request)
        assert len(taskFeat) == len(progFeat)
        distance = sum((x1-x2)**2 for x1, x2 in zip(taskFeat, progFeat))
        logLikelihood = float(-distance)  # FIXME: this is really naive
        return exp(logLikelihood) > self.successCutoff, logLikelihood


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.autograd import Variable

    class FeatureDiscriminatorLikelihoodModel(nn.Module):
        def __init__(self, tasks, featureExtractor,
                     successCutoff=0.6, H=8, trainingSuccessRatio=0.5):
            super(FeatureDiscriminatorLikelihoodModel, self).__init__()
            self.extract = featureExtractor
            self.successCutoff = successCutoff
            self.trainingSuccessRatio = trainingSuccessRatio

            self.W = nn.Linear(featureExtractor.outputDimensionality, H)
            self.output = nn.Linear(H, 1)

            # training on initialization
            self.train(tasks)

        def forward(self, examples):
            """
            Examples is a list of feature sets corresponding to a particular example.
            Output in [0,1] whether all examples correspond to the same program
            """
            assert all(len(x) == self.extract.outputDimensionality for x in examples)
            examples = [F.tanh(self.W(ex)) for ex in examples]
            maxed, _ = torch.max(torch.stack(examples), dim=0)
            return F.sigmoid(self.output(maxed))

        def train(self, tasks, steps=400):
            # list of list of features for each example in each task
            optimizer = torch.optim.Adam(self.parameters())
            with timing("Trained discriminator"):
                losses = []
                for i in xrange(steps):
                    self.zero_grad()
                    if random.random() <= self.trainingSuccessRatio:
                        # success
                        t = random.choice(tasks)
                        features = [self.extract.featuresOfTask(
                                        Task(t.name, t.request, [ex], t.features))
                                    for ex in t.examples]
                        loss = (self(features) - 1.0)**2
                    else:
                        # fail
                        t1, t2 = random.sample(tasks, 2)
                        features1 = [self.extract.featuresOfTask(
                                         Task(t1.name, t1.request, [ex], t1.features))
                                     for ex in t1.examples[:len(t1.examples)/2]]
                        features2 = [self.extract.featuresOfTask(
                                         Task(t2.name, t2.request, [ex], t2.features))
                                     for ex in t2.examples[len(t2.examples)/2:]]
                        features = features1 + features2
                        loss = self(features)**2

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data[0])
                    if not i % 50:
                        eprint("Discriminator Epoch", i, "Loss", sum(losses)/len(losses))
                        gc.collect()

        def score(self, program, task):
            taskFeatures = self.extract.featuresOfTask(task)
            progFeatures = self.extract.featuresOfProgram(program, task.request)
            likelihood = self([taskFeatures] + [progFeatures])
            likelihood = float(likelihood)
            return likelihood > self.successCutoff, log(likelihood)
except ImportError:
    pass
