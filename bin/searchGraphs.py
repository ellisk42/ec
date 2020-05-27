try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *

import dill
import numpy as np

exclude_lst = [
    "arch leg 1",
    "arch leg 2",
    "arch leg 6",
    "arch leg 8",
    "bridge (2) of arch 2",
    "bridge (2) of arch 4",
    "bridge (2) of arch 5",
    "bridge (3) of arch 3",
    "bridge (3) of arch 5",
    "bridge (4) of arch 1",
    "bridge (4) of arch 4",
    "bridge (5) of arch 3",
    "bridge (5) of arch 4",
    "bridge (5) of arch 5",
    "bridge (6) of arch 4",
    "bridge (6) of arch 5",
    "bridge (7) of arch 1",
    "bridge (7) of arch 4",
    "arch 1/2 pyramid 2",
    "brickwall, 3x1",
    "brickwall, 3x3",
    "brickwall, 3x4",
    "brickwall, 4x2",
    "brickwall, 4x4",
    "brickwall, 5x2",
    "brickwall, 6x1",
    "brickwall, 6x4",]


def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None, mode='fractionHit'):
    import matplotlib.pyplot as plot

    def averageLoss(n, predicate):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        losses = [ min([defaultLoss] + [1 - math.exp(r.loss) for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    def fractionHit(n, predicate):
        """simply plots fraction of tasks hit at all"""
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        hits = [ rs != [] for rs in results ]
        return sum(hits)/float(len(hits))*100

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('percent correct')
    if mode =='fractionHit': plot.ylim(bottom=0., top=100.)
    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        if mode =='fractionHit':
            plot.plot(xs, [fractionHit(n,lambda r: r.time < x) for x in xs],
                  label=names[n], linewidth=4)
            #plot.xscale('log')
        else:
            plot.plot(xs, [averageLoss(n,lambda r: r.time < x) for x in xs],
                  label=names[n], linewidth=4)
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('percent correct')
    if mode =='fractionHit': plot.ylim(bottom=0., top=100.)
    for n in range(len(testResults)):
        #xs = list(range(max([0]+[r.evaluations for tr in testResults[n] for r in tr] ) + 1))
        xs = list(range(18000))
        if mode =='fractionHit':
            plot.plot(xs, [fractionHit(n,lambda r: r.evaluations <= x) for x in xs],
                  label=names[n], linewidth=4)
            #plot.xscale('log')
        else:
            plot.plot(xs, [averageLoss(n,lambda r: r.evaluations <= x) for x in xs],
                  label=names[n], linewidth=4)
    plot.legend()
    if export:
        plot.savefig(f"{export}_evaluations.eps")
    else:
        plot.show()
        

if __name__ == '__main__':

    n = 3
    ID = 'towers' + str(n)

    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}RNN_SRE=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}REPL_SRE=True.pickle', 'REPL modular value')]

    # nameSalt = "towersLong"
    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}LongREPL_SRE=True_graph=True.pickle', 'REPL modular value (longer)')]

    # nameSalt = "towersSamplePolicy"
    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}SamplePolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}SamplePolicyREPL_SRE=True_graph=True.pickle', 'REPL modular value')]

    # ****
    nameSalt = "towersREPLPolicy"
    paths = [(f'experimentOutputs/{ID}SamplePolicySample_SRE=True_graph=True.pickle', 'Sample (no value)'),
        (f'experimentOutputs/{ID}REPLPolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value'),
        (f'experimentOutputs/{ID}REPLPolicySymbolic_SRE=True_graph=True.pickle', 'Symbolic value')]

    nameSalt = "towersLongSamplePolicy"
    paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample (no value)'),
        (f'experimentOutputs/{ID}SamplePolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}LongSamplePolicyREPL_SRE=True_graph=True.pickle', 'REPL modular value (longer)'),
        (f'experimentOutputs/{ID}Symbolic_SRE=True_graph=True.pickle', 'Symbolic value')]


    nameSalt = "towersSamplePolicyHashing"
    ID = 'towers' + str(n) + 'SamplePolicyHashing'
    paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
        (f'experimentOutputs/{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value'),
        (f'experimentOutputs/{ID}Symbolic_SRE=True_graph=True.pickle', 'Symbolic value'),
        ]
    #print("WARNING: using the REPLPolicyHashing runs")

    graph="_graph=True"
    #mode="Prior"
    nameSalt = "SMCOracle" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    ID = 'towers' + str(n)
    runType ="SMC" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    paths = [
        (f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle', 'Policy only (no value)'),
        (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'REPL modular value'),
        (f'experimentOutputs/towers20SMCSemiOracle_SRE=True_graph=True.pickle', 'Oracle Value fun'),
        (f'experimentOutputs/{ID}{runType}Symbolic_SRE=True{graph}.pickle', 'Symbolic value')
        ]

  
    graph=""
    nameSalt = "AstarPseudoResult" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    ID = 'towers' + str(n)
    runType ="PolicyOnlyPseudoResult" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    paths = [
        (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'Abstract REPL policy (ours)'),
        (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN Policy'),
        (f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle', 'Bigram Policy'),
        ]



    with open('biasedtasks.p', 'rb') as h: biasedtasks = dill.load(h)
    timeout=30
    outputDirectory = 'plots'
    paths, names = zip(*paths)

    for mode in ['test']: #['test', 'train']:

        testResults = []
        for path in paths:
            with open(path, 'rb') as h:
                r = dill.load(h)

            #optimize for speed
            for task, results in r.testingSearchStats[-1].items():
                if r.testingSearchStats[-1][task]:
                    r.testingSearchStats[-1][task] = r.testingSearchStats[-1][task][:1]

            delTasks = []
            seenBridges = False
            for task, results in r.testingSearchStats[-1].items():
                if "from bridges" in task.name: 
                    if seenBridges: delTasks.append(task)
                    seenBridges = True
                if r.testingSearchStats[-1][task] and r.testingSearchStats[-1][task][0].evaluations < 100:
                    print(task.name)
                if "pyramid on top" in task.name: delTasks.append(task)
                if task.name in exclude_lst: delTasks.append(task)
            for task in delTasks: del r.testingSearchStats[-1][task]

            # print(path)
            # for task, results in r.testingSearchStats[-1].items():
            #     if "Max twoArches" in task.name and r.testingSearchStats[-1][task]:
            #         print(task.name, r.testingSearchStats[-1][task][0].evaluations)
            #         print(r.testingSearchStats[-1][task][0].program)



            if hasattr(r, 'testingNumOfProg'):
                minN = float('inf')
                for task, results in r.testingSearchStats[-1].items():
                    if not results:
                        minN = min(minN, r.testingNumOfProg[-1][task])

                print("min of max N prog searched is", minN  )

            #import pdb; pdb.set_trace()
            res = r.searchStats[-1] if mode=='train' else r.testingSearchStats[-1]
            testResults.append( list(res.values()) )

        plotTestResults(testResults, timeout,
                        defaultLoss=1.,
                        names=names,
                        export=f"{outputDirectory}/{nameSalt}{ID}{mode}_curve.eps",
                        mode='fractionHit')
