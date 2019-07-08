from dreamcoder.task import Task, EvaluationTimeout
import gc
from dreamcoder.utilities import *
from collections import Counter
import math

from dreamcoder.domains.regex.groundtruthRegexes import gt_dict

gt_dict = {"Data column no. "+str(num): r_str for num, r_str in gt_dict.items()}

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

def longest_common_substr(arr):
    #array of examples 

# Python 3 program to find the stem
# of given list of words
# function to find the stem (longest
# common substring) from the string array
    # Determine size of the array
    n = len(arr)

    # Take first word from array
    # as reference
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l) :
        for j in range( i + 1, l + 1) :
            # generating all possible substrings
            # of our reference string arr[0] i.e s
            stem = s[i:j]
            k = 1
            for k in range(1, n):

                # Check if the generated stem is
                # common to to all words
                if stem not in arr[k]:
                    break

                # If current substring is present in
                # all strings and its length is greater
                # than current result
            if (k + 1 == n and len(res) < len(stem)): res = stem
    return res 

def add_string_constants(tasks):
    for task in tasks:
        task.str_const = longest_common_substr([example[1] for example in task.examples])
    return tasks

def get_gt_ll(name, examples):
    #gets groundtruth from dict
    import pregex as pre
    r_str = gt_dict[name]
    preg = pre.create(r_str)

    if type(examples[0]) == list:
        examples = [ "".join(example) for example in examples]

    s = sum( preg.match(example) for example in examples)
    if s == float("-inf"):
        print("bad for ", name)
        print('preg:', preg)
        print('preg sample:', [preg.sample() for i in range(3)])
        print("exs", examples)
        #assert False 
    return s


def add_cutoff_values(tasks, ll_cutoff):
    from dreamcoder.domains.regex.makeRegexTasks import makeNewTasks
    if ll_cutoff is None or ll_cutoff == "None":
        for task in tasks:
            task.ll_cutoff = None
        return tasks
    if ll_cutoff == "gt":
        from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
        for task in tasks:
            task.ll_cutoff = None
            task.gt = get_gt_ll(task.name, [example[1] for example in task.examples])
            task.gt_test = get_gt_ll(task.name,
                                     [example[1] for example in regexHeldOutExamples(task) ])
        return tasks
    elif ll_cutoff == "plus":
        for task in tasks:
            task.ll_cutoff = regex_plus_bound([example[1] for example in task.examples])
        return tasks
    elif ll_cutoff == "bigram":
        eprint("WARNING: using entire corpus to make bigram model")
        #this means i do it twice, which is eh whatever
        model = make_corpus_bigram(show_tasks(makeNewTasks()))
        for task in tasks:
            task.ll_cutoff = bigram_corpus_score([example[1] for example in task.examples], model)
        return tasks
    elif ll_cutoff =="unigram":
        eprint("WARNING: using entire corpus to make unigram model")
        #this means i do it twice, which is eh whatever
        model = make_corpus_unigram(show_tasks(makeNewTasks()))
        for task in tasks:
            task.ll_cutoff = unigram_corpus_score([example[1] for example in task.examples], model)
        return tasks
    elif ll_cutoff =="mix":
        eprint("WARNING: using entire corpus to make bigram model")
        eprint("WARNING: using entire corpus to make unigram model")
        #this means i do it twice, which is eh whatever
        unigram = make_corpus_unigram(show_tasks(makeNewTasks()))
        bigram = make_corpus_bigram(show_tasks(makeNewTasks()))
        for task in tasks:
            uniscore = unigram_corpus_score([example[1] for example in task.examples], unigram)
            biscore = bigram_corpus_score([example[1] for example in task.examples], bigram)
            task.ll_cutoff = math.log(0.75*math.exp(biscore) + 0.25*math.exp(uniscore))
        return tasks
    else:
        eprint("not implemented")
        eprint("cutoff val:")
        eprint(ll_cutoff)
        assert False

def show_tasks(dataset):
    task_list = []
    for task in dataset:
        task_list.append([example[1] for example in task.examples])
    return task_list

def regex_plus_bound(X):
    from pregex import pregex
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


def make_corpus_unigram(C):
    str_list = [example + '\n' for task in C for example in task]
    c = Counter(char for example in str_list for char in example )
    n = sum(c.values())

    logp = {x:math.log(c[x]/n) for x in c}
    return logp

def unigram_corpus_score(X, logp):
    task_ll = 0
    for x in X:
        x = x + '\n'
        task_ll += sum( logp.get(c, float('-inf')) for c in x)/len(x)

    ll = task_ll/len(X)
    return ll

def unigram_task_score(X):
    """
    Given a list of strings, X, calculate the maximum log-likelihood per character for a unigram model over characters (including STOP symbol)
    """
    c = Counter(x for s in X for x in s)
    c.update("end" for s in X)
    n = sum(c.values())
    logp = {x:math.log(c[x]/n) for x in c}
    return sum(c[x]*logp[x] for x in c)/n

def make_corpus_bigram(C):
    #using newline as "end"
    #C is a list of tasks

    #make one big list of strings
    str_list = [example + '\n' for task in C for example in task]

    #make list of 
    head_count = Counter(element[0] for element in str_list)
    head_n = sum(head_count.values())
    head_logp = {x:math.log(head_count[x]/head_n) for x in head_count}

    body_count = Counter(element[i:i+2] for element in str_list for i in range(len(element)-1))
    body_bigram_n = sum(body_count.values())
    #body_count/body_bigram_n gives the joint of a bigram
    body_character_n = Counter(char for element in str_list for char in element)
    body_unigram_n = sum(body_character_n.values())

    body_logp = {x:math.log(body_count[x] / body_bigram_n / body_character_n[x[0]] * body_unigram_n) for x in body_count}

    return {**head_logp, **body_logp}

def bigram_corpus_score(X, logp):
    #assume you have a logp dict
    task_ll = 0
    for x in X:
        bigram_list = [x[0]] + [x[i:i+2] for i in range(len(x)-1)] + [x[-1] + '\n']
        bigram_list = [ ''.join(b) if isinstance(b,list) else b
                        for b in bigram_list ]

        string_ll = sum(logp.get(bigram, float('-inf')) for bigram in bigram_list) #/(len(x) + 1)

        task_ll += string_ll

    ll = task_ll #/len(X)
    return ll


class ProbabilisticLikelihoodModel:

    def __init__(self, timeout):
        self.timeout = timeout
        # i need timeout

    def score(self, program, task):
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
            
            #cutoff_ll = regex_plus_bound(example_list)   

            normalized_cum_ll = cum_ll/ float(sum([len(example) for example in example_list]))



            #TODO: change the way normalized_cum_ll is calculated 
            #TODO: refactor to pass in bigram_model, and others
            #TODO: refactor to do 95% certainty thing josh wants
            success = normalized_cum_ll > task.ll_cutoff



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


if __name__=="__main__":

    arr = ['MAM.OSBS.2014.06', 'MAM.OSBS.2013.07', 'MAM.OSBS.2013.09', 'MAM.OSBS.2014.05', 'MAM.OSBS.2014.11']
    stems = longest_common_substr(arr)
    print(stems)



