from ec import *
import dill
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

import matplotlib
#from test_unpickle import loadfun
def loadfun(x):
    with open(x, 'rb') as handle:
        result = dill.load(handle)
    return result

TITLEFONTSIZE = 14
TICKFONTSIZE = 12
LABELFONTSIZE = 14

matplotlib.rc('xtick', labelsize=TICKFONTSIZE)
matplotlib.rc('ytick', labelsize=TICKFONTSIZE)


class Bunch(object):
    def __init__(self, d):
        self.__dict__.update(d)

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]


relu = 'relu'
tanh = 'tanh'
sigmoid = 'sigmoid'
DeepFeatureExtractor = 'DeepFeatureExtractor'
LearnedFeatureExtractor = 'LearnedFeatureExtractor'
TowerFeatureExtractor = 'TowerFeatureExtractor'


def parseResultsPath(p):
    def maybe_eval(s):
        try:
            return eval(s)
        except BaseException:
            return s

    p = p[:p.rfind('.')]
    domain = p[p.rindex('/') + 1: p.index('_')]
    rest = p.split('_')[1:]
    if rest[-1] == "baselines":
        rest.pop()
    parameters = {ECResult.parameterOfAbbreviation(k): maybe_eval(v)
                  for binding in rest if '=' in binding
                  for [k, v] in [binding.split('=')]}
    parameters['domain'] = domain
    return Bunch(parameters)


def plotECResult(
        resultPaths,
        interval=False,
        timePercentile=False,
        colors='rgbycm',
        labels=None,
        failAsTimeout=False,
        title=None,
        testingTimeout=None,
        export=None,
        showSolveTime=True,
        showTraining=False,
        iterations=None,
        maxP=110,
        showEpochs=False):
    results = []
    parameters = []
    for j, path in enumerate(resultPaths):
        result = loadfun(path)
        print("loaded path:", path)

        if hasattr(result, "baselines") and result.baselines:
            for name, res in result.baselines.items():
                results.append(res)
                p = parseResultsPath(path)
                p["baseline"] = name.replace("_", " ")
                parameters.append(p)
        else:
            results.append(result)
            parameters.append(parseResultsPath(path))

    if testingTimeout is not None:
        for r in results:
            r.testingSearchTime = [ [t for t in ts if t <= testingTimeout ]
                                    for ts in result.testingSearchTime ]
    # Collect together the timeouts, which determine the style of the line
    # drawn
    timeouts = sorted(set(r.enumerationTimeout for r in parameters),
                      reverse=2)
    
    f, a1 = plot.subplots(figsize=(4, 3))
    xlabel = 'Epoch' if showEpochs else 'Iteration'
    a1.set_xlabel('Epoch', fontsize=LABELFONTSIZE)
    a1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if showSolveTime:
        a1.set_ylabel('%  Solved (solid)', fontsize=LABELFONTSIZE)
        a2 = a1.twinx()
        a2.set_ylabel('Solve time (dashed)', fontsize=LABELFONTSIZE)
    else:
        if not showTraining:
            a1.set_ylabel('% Testing Tasks Solved', fontsize=LABELFONTSIZE)
        else:
            a1.set_ylabel('% Tasks Solved', fontsize=LABELFONTSIZE)
            

    n_iters = max(len(result.learningCurve) for result in results)
    if iterations and n_iters > iterations:
        n_iters = iterations

    plot.xticks(range(0, n_iters), fontsize=TICKFONTSIZE)

    colors = ["#D95F02", "#1B9E77"] + ["#000000"]*100
    
    for result, p, color in zip(results, parameters, colors):
        if hasattr(p, "baseline") and p.baseline:
            ys = [100. * result.learningCurve[-1] /
                  len(result.taskSolutions)] * n_iters
        elif showTraining:
            ys = [100.*t/float(len(result.taskSolutions))
                  for t in result.learningCurve[:iterations]]
        else:
            ys = [100. * len(t) / result.numTestingTasks
                  for t in result.testingSearchTime[:iterations]]

        xs = list(range(0, len(ys)))
        if showEpochs:
            if 'taskBatchSize' not in p.__dict__: p.__dict__['taskBatchSize'] = len(result.taskSolutions)
            xs = [ (p.taskBatchSize / (float(len(result.taskSolutions)))) * i for i in xs]
        if labels:
            usedLabels.append((labels[0], color))
            labels = labels[1:]
            
        l, = a1.plot(xs, ys, color=color, ls='-')
        
        if showSolveTime:
            if failAsTimeout:
                assert testingTimeout is not None
                result.testingSearchTime = [ ts + [testingTimeout]*(result.numTestingTasks - len(ts))
                                             for ts in result.testingSearchTime ]
                result.searchTimes = [ ts + [p.enumerationTimeout]*(len(result.taskSolutions) - len(ts))
                                       for ts in result.searchTimes ]

            if not showTraining: times = result.testingSearchTime[:iterations]
            else: times = result.searchTimes[:iterations]
            a2.plot(xs,
                    [mean(ts) if not timePercentile else median(ts)
                         for ts in times],
                    color=color, ls='--')
            if interval:
                a2.fill_between(range(len(times)),
                                [percentile(ts, 0.75) if timePercentile else mean(ts) + standardDeviation(ts)
                                 for ts in times],
                                [percentile(ts, 0.25) if timePercentile else mean(ts) - standardDeviation(ts)
                                 for ts in times],
                                facecolor=color, alpha=0.2)

    a1.set_ylim(ymin=0, ymax=maxP)
    a1.yaxis.grid()
    a1.set_yticks(range(0, maxP, 20))
    plot.yticks(range(0, maxP, 20), fontsize=TICKFONTSIZE)

    if showSolveTime:
        a2.set_ylim(ymin=0)
        starting, ending = a2.get_ylim()
        ending10 = 10*int(ending/10 + 1)
        a2.yaxis.set_ticks([ int(ending10/6)*j
                             for j in range(0, 6)]) 
        # if False:
        #     # [int(zz) for zz in np.arange(starting, ending, (ending - starting)/5.)])
        #     a2.yaxis.set_ticks([20 * j for j in range(6)])
        # else:
        #     a2.yaxis.set_ticks([50 * j for j in range(6)])
        for tick in a2.yaxis.get_ticklabels():
            tick.set_fontsize(TICKFONTSIZE)

    if title is not None:
        plot.title(title, fontsize=TITLEFONTSIZE)

    if labels is not None:
        legends.append(a1.legend(loc='lower right', fontsize=9,
                                 #bbox_to_anchor=(1, 0.5),
                                 handles=[mlines.Line2D([], [], color=color, ls='-',
                                                        label=label)
                                          for label, color in usedLabels]))
    f.tight_layout()
    if export:
        plot.savefig(export)
        eprint("Exported figure ",export)
        if export.endswith('.png'):
            os.system('convert -trim %s %s' % (export, export))
        os.system('feh %s' % export)
    else:
        f.show()
        

def tryIntegerParse(s):
    try:
        return int(s)
    except BaseException:
        return None


if __name__ == "__main__":
    import sys

    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--checkpoints",nargs='+')
    parser.add_argument("--title","-t",type=str,
                        default="")
    parser.add_argument("--iterations","-i",
                        type=int, default=None,
                        help="number of iterations of EC to show")
    parser.add_argument("--names","-n",
                        type=str, default="",
                        help="comma-separated list of names to put on the plot for each checkpoint")
    parser.add_argument("--export","-e",
                        type=str, default=None)
    parser.add_argument("--percentile","-p",
                        default=False, action="store_true",
                        help="When displaying error bars for synthesis times, this option will cause it to show 25%/75% interval. By default it instead shows +/-1 stddev")
    parser.add_argument("--interval",
                        default=False, action="store_true",
                        help="Should we show error bars for synthesis times?")
    parser.add_argument("--testingTimeout",
                        default=None, type=float,
                        help="Retroactively pretend that the testing timeout was something else. WARNING: This will only give valid results if you are retroactively pretending that the testing timeout was smaller than it actually was!!!")
    parser.add_argument("--failAsTimeout",
                        default=False, action="store_true",
                        help="When calculating average solve time, should you count missed tasks as timeout OR should you just ignore them? Default: ignore them.")
    parser.add_argument("--showTraining",
                        default=False, action="store_true",
                        help="Graph results for training tasks. By default only shows results for testing tasks.")
    parser.add_argument("--maxPercent","-m",
                        type=int, default=110,
                        help="Maximum percent for the percent hits graph")
    parser.add_argument("--showEpochs",
                        default=False, action="store_true",
                        help='X-axis is real-valued percentage of training tasks seen, instead of iterations.')
    
    arguments = parser.parse_args()
    
    plotECResult(arguments.checkpoints,
                 testingTimeout=arguments.testingTimeout,
                 timePercentile=arguments.percentile,
                 export=arguments.export,
                 title=arguments.title,
                 failAsTimeout=arguments.failAsTimeout,
                 labels=arguments.names.split(","),
                 interval=arguments.interval,
                 iterations=arguments.iterations,
                 showTraining=arguments.showTraining,
                 maxP=arguments.maxPercent,
                 showEpochs=arguments.showEpochs)
