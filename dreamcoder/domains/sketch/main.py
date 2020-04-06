from dreamcoder.dreamcoder import *
from dreamcoder.domains.sketch.sketchPrimitives import *
from dreamcoder.domains.sketch.makeSketchTasks import *
from dreamcoder.utilities import *
from dreamcoder.grammar import Grammar

import os
import datetime

def SketchCNN():
    pass

g0 = Grammar.uniform(primitives, continuationType=tsketch)

def dreamOfSketches(grammar=g0, N=50, make_montage=True):
    request = arrow(tsketch, tsketch)
    programs = [p for _ in range(N) for p in [grammar.sample(request, maximumDepth=15)] if p is not None]

    # randomTowers = [tuple(centerTower(t))
    #                 for _ in range(N)
    #                 for program in [grammar.sample(request,
    #                                                maximumDepth=12,
    #                                                maxAttempts=100)]
    #                 if program is not None
    #                 for t in [executeTower(program, timeout=0.5) or []]
    #                 if len(t) >= 1 and len(t) < 100 and towerLength(t) <= 360.]
    # matrix = [renderPlan(p,Lego=True,pretty=True)
    #           for p in randomTowers]

    # # Only visualize if it has something to visualize.
    # if len(matrix) > 0:
    #     import scipy.misc
    #     if make_montage:
    #         matrix = montage(matrix)
    #         scipy.misc.imsave('%s.png'%prefix, matrix)
    #     else:
    #         for n,i in enumerate(matrix):
    #             scipy.misc.imsave(f'{prefix}/{n}.png', i)
    # else:
    #     eprint("Tried to visualize dreams, but none to visualize.")
    return programs


def main(arguments):
        g0 = Grammar.uniform(primitives)

        # TasksTrain = makeSupervisedTasks(trainset=arguments["trainset"])[:2]
        TasksTrain = makeSupervisedTasks(trainset=["practice_shaping", "practice"], Nset=[20])

        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "experimentOutputs/sketch/%s"%timestamp
        evaluationTimeout = 0.001 # seconds, how long allowed

        os.system(f"mkdir -p {outputDirectory}")

        if False:
            arguments["featureExtractor"] = SketchCNN

        if arguments["skiptesting"]==False and len(test)>0:
                generator = ecIterator(g0, TasksTrain, testingTasks=test,
                        outputPrefix="%s/sketch"%outputDirectory,
                        evaluationTimeout=evaluationTimeout,
                        **arguments) # 
        else:
                print("NO TESTING TASKS INCLUDED")
                generator = ecIterator(g0, TasksTrain,
                        outputPrefix="%s/sketch"%outputDirectory,
                        evaluationTimeout=evaluationTimeout,
                        **arguments) # 

        for result in generator:
                continue
