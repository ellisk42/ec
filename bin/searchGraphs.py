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
        return sum(hits)/float(len(hits))

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')
    if mode =='fractionHit': plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        if mode =='fractionHit':
            plot.plot(xs, [fractionHit(n,lambda r: r.time < x) for x in xs],
                  label=names[n])
        else:
            plot.plot(xs, [averageLoss(n,lambda r: r.time < x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('percent correct')
    if mode =='fractionHit': plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        #xs = list(range(max([0]+[r.evaluations for tr in testResults[n] for r in tr] ) + 1))
        xs = list(range(200))
        if mode =='fractionHit':
            plot.plot(xs, [fractionHit(n,lambda r: r.evaluations <= x) for x in xs],
                  label=names[n])
        else:
            plot.plot(xs, [averageLoss(n,lambda r: r.evaluations <= x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(f"{export}_evaluations.png")
    else:
        plot.show()
        

if __name__ == '__main__':

    n = 20
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
    mode="Prior"
    nameSalt = "towersPrior"
    ID = 'towers' + str(n)
    runType = "Prior"
    paths = [(f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle', 'Sample from prior only (no value)'),
        (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'REPL modular value'),
        (f'experimentOutputs/{ID}{runType}Symbolic_SRE=True{graph}.pickle', 'Symbolic value')
        ]


    timeout=1200
    outputDirectory = 'plots'
    paths, names = zip(*paths)

    for mode in ['test']: #['test', 'train']:

        testResults = []
        for path in paths:
            with open(path, 'rb') as h:
                r = dill.load(h)
                
            from dreamcoder.showTowerTasks import showTowersAndSolutions, computeValue
            #showTowersAndSolutions(r)
            #computeValue(r)
            #assert 0
            #import pdb; pdb.set_trace()
            res = r.searchStats[-1] if mode=='train' else r.testingSearchStats[-1]
            testResults.append( list(res.values()) )

        plotTestResults(testResults, timeout,
                        defaultLoss=1.,
                        names=names,
                        export=f"{outputDirectory}/{nameSalt}{ID}{mode}_curve.png",
                        mode='fractionHit')
