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
    plot.ylabel('Average Loss')
    if mode =='fractionHit': plot.ylim(bottom=0.)
    for n in range(len(testResults)):
        xs = list(range(max(r.evaluations for tr in testResults[n] for r in tr ) + 1))
        #xs = list(range(4000))
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

    paths = [('experimentOutputs/listCathyTestGraph_SRE=True_graph=True.pickle', 'mock' )]
    paths = [('experimentOutputs/listCathyTestEnum_SRE=True_graph=True.pickle', 'mock' )]
    paths = [('experimentOutputs/listCathyTestIT=1.pickle', 'mock')]

    # paths = [('experimentOutputs/experimentOutputs/listCathyTestEnum.pickle', 'Enum')
    #           ('experimentOutputs/listCathyTestRNN.pickle', 'RNN')
    #           ('experimentOutputs/listCathyTestREPL.pickle', 'Abstract REPL') ]

    timeout=300
    outputDirectory = 'plots'
    paths, names = zip(*paths)

    for mode in ['test', 'train']:

        testResults = []
        for path in paths:
            with open(path, 'rb') as h:
                r = dill.load(h)
            import pdb; pdb.set_trace()
            res = r.searchStats[-1] if mode=='train' else r.testingSearchStats[-1]
            testResults.append( list(res.values()) )

        plotTestResults(testResults, timeout,
                        defaultLoss=1.,
                        names=names,
                        export=f"{outputDirectory}/{mode}_curve.png",
                        mode='fractionHit')
