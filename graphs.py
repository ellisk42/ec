from ec import *
from regexes import *
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
                  for binding in rest
                  for [k, v] in [binding.split('=')]}
    parameters['domain'] = domain
    return Bunch(parameters)


def plotECResult(
        resultPaths,
        colors='rgbycm',
        label=None,
        title=None,
        export=None,
        showSolveTime=False,
        iterations=None):
    results = []
    parameters = []
    for j, path in enumerate(resultPaths):
#        with open(path, 'rb') as handle:
        print("path:", path)
        result = loadfun(path)

        if hasattr(result, "baselines") and result.baselines:
            for name, res in result.baselines.items():
                results.append(res)
                p = parseResultsPath(path)
                p["baseline"] = name.replace("_", " ")
                parameters.append(p)
        else:
            results.append(result)
            p = parseResultsPath(path)
            parameters.append(p)

    # Collect together the timeouts, which determine the style of the line
    # drawn
    timeouts = sorted(set(r.enumerationTimeout for r in parameters),
                      reverse=2)
    timeoutToStyle = {
        size: style for size, style in zip(
            timeouts, [
                "-", "--", "-."])}

    f, a1 = plot.subplots(figsize=(4, 3))
    a1.set_xlabel('Iteration', fontsize=LABELFONTSIZE)
    a1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if showSolveTime:
        a1.set_ylabel('%  Solved (solid)', fontsize=LABELFONTSIZE)
    else:
        a1.set_ylabel('% Testing Tasks Solved', fontsize=LABELFONTSIZE)
        

    if showSolveTime:
        a2 = a1.twinx()
        a2.set_ylabel('Solve time (dashed)', fontsize=LABELFONTSIZE)

    n_iters = max(len(result.learningCurve) for result in results)
    if iterations and n_iters > iterations:
        n_iters = iterations

    plot.xticks(range(0, n_iters), fontsize=TICKFONTSIZE)

    recognitionToColor = {False: "teal", True: "orange"}

    for result, p in zip(results, parameters):
        if hasattr(p, "baseline") and p.baseline:
            ys = [100. * result.learningCurve[-1] /
                  len(result.taskSolutions)] * n_iters
        else:
            ys = [100. * len(t) / result.numTestingTasks
                  for t in result.testingSearchTime[:iterations]]
        color = recognitionToColor[p.useRecognitionModel]
        l, = a1.plot(list(range(0, len(ys))), ys, color=color, ls='-')

        if showSolveTime:
            a2.plot(range(len(result.testingSearchTime[:iterations])),
                    [sum(ts) / float(len(ts)) for ts in result.testingSearchTime[:iterations]],
                    color=color, ls='--')

    a1.set_ylim(ymin=0, ymax=110)
    a1.yaxis.grid()
    a1.set_yticks(range(0, 110, 20))
    plot.yticks(range(0, 110, 20), fontsize=TICKFONTSIZE)

    if showSolveTime:
        a2.set_ylim(ymin=0)
        starting, ending = a2.get_ylim()
        if True:
            # [int(zz) for zz in np.arange(starting, ending, (ending - starting)/5.)])
            a2.yaxis.set_ticks([20 * j for j in range(5)])
        else:
            a2.yaxis.set_ticks([50 * j for j in range(6)])
        for tick in a2.yaxis.get_ticklabels():
            print(tick)
            tick.set_fontsize(TICKFONTSIZE)

    if title is not None:
        plot.title(title, fontsize=TITLEFONTSIZE)

    # if label is not None:
    legends = []
    if len(timeouts) > 1:
        legends.append(a1.legend(loc='lower right', fontsize=14,
                                 #bbox_to_anchor=(1, 0.5),
                                 handles=[mlines.Line2D([], [], color='black', ls=timeoutToStyle[timeout],
                                                        label="timeout %ss" % timeout)
                                          for timeout in timeouts]))
    if False:
        # FIXME: figure out how to have two separate legends
        plot.gca().add_artist(
            plot.legend(
                loc='lower left',
                fontsize=20,
                handles=[
                    mlines.Line2D(
                        [],
                        [],
                        color=recognitionToColor[True],
                        ls='-',
                        label="DreamCoder"),
                    mlines.Line2D(
                        [],
                        [],
                        color=recognitionToColor[False],
                        ls='-',
                        label="No NN")]))

    f.tight_layout()
    if export:
        plot.savefig(export)  # , additional_artists=legends)
        if export.endswith('.png'):
            os.system('convert -trim %s %s' % (export, export))
        os.system('feh %s' % export)
    else:
        plot.show()
        

def tryIntegerParse(s):
    try:
        return int(s)
    except BaseException:
        return None


if __name__ == "__main__":
    import sys

    def label(p):
        #l = p.domain
        l = ""
        if hasattr(p, 'baseline') and p.baseline:
            l += "baseline %s" % p.baseline
            return l
        if p.useRecognitionModel:
            if hasattr(p, 'helmholtzRatio') and p.helmholtzRatio > 0:
                l += "DreamCoder"
            else:
                l += "AE"
        else:
            l += "no NN"
        if hasattr(p, "frontierSize"):
            l += " (frontier size %s)" % p.frontierSize
        else:
            l += " (timeout %ss)" % p.enumerationTimeout
        return l
    arguments = sys.argv[1:]
    export = [a for a in arguments if a.endswith('.png') or a.endswith('.eps')]
    export = export[0] if export else None
    title = [
        a for a in arguments if not any(
            a.endswith(s) for s in {
                '.eps',
                '.png',
                '.pickle'})]

    # pass in an integer on the command line to  number of plotted iterations
    iterations = [tryIntegerParse(a) for a in arguments if tryIntegerParse(a)]
    iterations = None if iterations == [] else iterations[0]

    plotECResult([a for a in arguments if a.endswith('.pickle')],
                 export=export,
                 title=title[0] if title else "DSL learning curves",
                 label=label,
                 showSolveTime=True,
                 iterations=iterations)
