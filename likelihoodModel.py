from utilities import eprint, exp, log, timing, valid
from task import Task, EvaluationTimeout
import random
import gc
from pregex import pregex
import signal
from program import *
from utilities import *
from collections import Counter


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
        distance = sum((x1 - x2)**2 for x1, x2 in zip(taskFeat, progFeat))
        logLikelihood = float(-distance)  # FIXME: this is really naive
        return exp(logLikelihood) > self.successCutoff, logLikelihood


def unigram_regex_bound(X):
    c = Counter(X)
    regexes = [
        pregex.create(".+"),
        pregex.create("\d+"),
        pregex.create("\w+"),
        pregex.create("\s+"),
        pregex.create("\\u+"),
        pregex.create("\l+")]
    regex_scores = []
    for r in regexes:
        regex_scores.append(sum(c[x] * r.match(x) for x in c)/float(sum([len(x) for x in X])) )
    return max(regex_scores)

class ProbabilisticLikelihoodModel:

    def __init__(self, timeout):
        self.timeout = timeout
        # i need timeout

    def score(self, program, task, testing=False):
        # need a try, catch here for problems, and for timeouts
        # can copy task.py for the timeout structure
        try:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, self.timeout)
            try:
                string_pregex = program.evaluate([])
                # if 'left_paren' in program.show(False):
                #eprint("string_pregex:", string_pregex)
                #eprint("string_pregex:", string_pregex)
                preg = string_pregex  # pregex.create(string_pregex)
            except IndexError:
                # free variable
                return False, NEGATIVEINFINITY
            except Exception as e:
                eprint("Exception during evaluation:", e)
                if "Attempt to evaluate fragment variable" in e:
                    eprint("program (bc fragment error)", program)
                return False, NEGATIVEINFINITY

        #tries and catches

        # include prior somehow
        # right now, just summing up log likelihoods. IDK if this is correct.
        # also not using prior at all.
            cum_ll = 0

            example_list = [example[1] for example in task.examples]
            c_example_list = Counter(example_list)

            for c_example in c_example_list:
                #might want a try, except around the following line:

                try:
                    #eprint("about to match", program)
                    #print("preg:", preg)
                    ll = preg.match(c_example)
                    #eprint("completed match", ll, program)
                except ValueError as e:
                    eprint("ValueError:", e)
                    ll = float('-inf')
                
                #eprint("pregex:", string_pregex)
                #eprint("example[1]", example[1])

                if ll == float('-inf'):
                    return False, NEGATIVEINFINITY
                else:
                    #ll_per_char = ll/float(len(example[1]))
                    #cum_ll_per_char += ll_per_char

                    cum_ll += c_example_list[c_example] * ll
            
            #normalized_cum_ll_per_char = cum_ll_per_char/float(len(task.examples))
            #avg_char_num = sum([len(example[1]) for example in task.examples])/float(len(task.examples))
            
            cutoff_ll = unigram_regex_bound(example_list)   

            normalized_cum_ll = cum_ll/ float(sum([len(example) for example in example_list]))

            #testing = True 
            if testing:
                success = normalized_cum_ll > cutoff_ll
            else:
                success = normalized_cum_ll > float('-inf')

            #eprint("cutoff_ll:", cutoff_ll, ", norm_cum_ll:", normalized_cum_ll)	

            return success, normalized_cum_ll

        except EvaluationTimeout:
            eprint("Timed out while evaluating", program)
            return False, NEGATIVEINFINITY
        finally:
            signal.signal(signal.SIGVTALRM, lambda *_: None)
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)


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
            assert all(
                len(x) == self.extract.outputDimensionality for x in examples)
            examples = [F.tanh(self.W(ex)) for ex in examples]
            maxed, _ = torch.max(torch.stack(examples), dim=0)
            return F.sigmoid(self.output(maxed))

        def train(self, tasks, steps=400):
            # list of list of features for each example in each task
            optimizer = torch.optim.Adam(self.parameters())
            with timing("Trained discriminator"):
                losses = []
                for i in range(steps):
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
                            for ex in t1.examples[:len(t1.examples) / 2]]
                        features2 = [self.extract.featuresOfTask(
                            Task(t2.name, t2.request, [ex], t2.features))
                            for ex in t2.examples[len(t2.examples) / 2:]]
                        features = features1 + features2
                        loss = self(features)**2

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data[0])
                    if not i % 50:
                        eprint(
                            "Discriminator Epoch",
                            i,
                            "Loss",
                            sum(losses) /
                            len(losses))
                        gc.collect()

        def score(self, program, task):
            taskFeatures = self.extract.featuresOfTask(task)
            progFeatures = self.extract.featuresOfProgram(
                program, task.request)
            likelihood = self([taskFeatures] + [progFeatures])
            likelihood = float(likelihood)
            return likelihood > self.successCutoff, log(likelihood)
except ImportError:
    pass
