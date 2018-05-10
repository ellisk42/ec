from ec import explorationCompression, commandlineArguments, Task, ecIterator
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs, median, standardDeviation, mean
from makeTextTasks import makeTasks, delimiters, loadPBETasks
from textPrimitives import primitives
from listPrimitives import bootstrapTarget
from program import *
from recognition import *
from enumeration import *

import random

class ConstantInstantiateVisitor(object):
    def __init__(self, words):
        self.words = words
    def primitive(self,e):
        if e.name == "STRING": return Primitive("STRING", e.tp, random.choice(self.words))
        return e
    def invented(self,e): return e.body.visit(self)
    def index(self,e): return e
    def application(self,e):
        return Application(e.f.visit(self), e.x.visit(self))
    def abstraction(self,e):
        return Abstraction(e.body.visit(self))



class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        return examples

    def __init__(self, tasks):
        lexicon = {c
                   for t in tasks
                   for xs,y in self.tokenize(t.examples)
                   for c in reduce(lambda u,v: u+v, list(xs) + [y]) }
                
        super(LearnedFeatureExtractor, self).__init__(lexicon = list(lexicon),
                                                      H = 64,
                                                      tasks = tasks,
                                                      bidirectional = True)
    def featuresOfProgram(self, p, tp):
        # Instantiate STRING w/ random words
        p = p.visit(ConstantInstantiateVisitor.SINGLE)
        return super(LearnedFeatureExtractor, self).featuresOfProgram(p,tp)


if __name__ == "__main__":
    doChallenge = True
    
    tasks = makeTasks()
    eprint("Generated",len(tasks),"tasks")

    for t in tasks: t.mustTrain = False
    
    test, train = testTrainSplit(tasks, 0.75)
    eprint("Split tasks into %d/%d test/train"%(len(test),len(train)))
    
    challenge, challengeCheating = loadPBETasks()
    eprint("Got %d challenge PBE tasks"%len(challenge))

    ConstantInstantiateVisitor.SINGLE = \
     ConstantInstantiateVisitor(map(list,list({ tuple([c for c in s])
                                                for t in test + train + challenge
                                                for s in t.stringConstants })))
    
    baseGrammar = Grammar.uniform(primitives + bootstrapTarget())
    
    evaluationTimeout = 0.0005
    # We will spend 30 minutes on each challenge problem
    challengeTimeout = 30 * 60
    
    generator = ecIterator(baseGrammar, train,
                           testingTasks = test,
                           outputPrefix = "experimentOutputs/text",
                           evaluationTimeout = evaluationTimeout,
                           compressor="pypy", # 
                           **commandlineArguments(
                               steps = 500,
                               iterations = 10,
                               helmholtzRatio = 0.5,
                               topK = 2,
                               maximumFrontier = 2,
                               structurePenalty = 10.,
                               a = 3,
                               activation = "tanh",
                               CPUs = numberOfCPUs(),
                               featureExtractor = LearnedFeatureExtractor,
                               pseudoCounts = 10.0))
    if doChallenge:
        eprint("challenge problems before learning...")
        challengeFrontiers, times = multicoreEnumeration(baseGrammar, challenge, "all-or-nothing",
                                         CPUs=numberOfCPUs(),
                                         solver="ocaml",
                                         maximumFrontier=1,
                                         enumerationTimeout=challengeTimeout,
                                         evaluationTimeout=evaluationTimeout)
        eprint(Frontier.describe(challengeFrontiers))
        summaryStatistics("Challenge problem search time",times)
        eprint("done evaluating challenge problems before learning")

    for result in generator:
        if not doChallenge: continue
        
        eprint("Evaluating on challenge problems...")
        if result.recognitionModel is not None:
            recognizer = result.recognitionModel
            challengeFrontiers, times = \
                                        recognizer.enumerateFrontiers(challenge, "all-or-nothing",
                                          CPUs=numberOfCPUs(),
                                          solver="ocaml",
                                          maximumFrontier=1,
                                          enumerationTimeout=challengeTimeout,
                                          evaluationTimeout=evaluationTimeout)
        else:
            challengeFrontiers, times = \
                multicoreEnumeration(result.grammars[-1], challenge, "all-or-nothing",
                                     CPUs=numberOfCPUs(),
                                     solver="ocaml",
                                     maximumFrontier=1,
                                     enumerationTimeout=challengeTimeout,
                                     evaluationTimeout=evaluationTimeout)
        eprint("Challenge problem enumeration results:")
        eprint(Frontier.describe(challengeFrontiers))
        summaryStatistics("Challenge problem search time",times)
