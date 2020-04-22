try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.dreamcoder import *
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

import matplotlib
#from test_unpickle import loadfun

# Required import for unpickling older checkpoints:
from dreamcoder.domains.tower.main import TowerCNN


def loadfun(x):
    with open(x, 'rb') as handle:
        result = dill.load(handle)
    return result

TITLEFONTSIZE = 14
TICKFONTSIZE = 12
LABELFONTSIZE = 14

matplotlib.rc('xtick', labelsize=TICKFONTSIZE)
matplotlib.rc('ytick', labelsize=TICKFONTSIZE)

def shuffled(g):
    import random
    g = list(g)
    random.shuffle(g)
    return g

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

def padSearchTimes(result, testingTimeout, enumerationTimeout):
    result.testingSearchTime = [ ts + [testingTimeout]*(result.numTestingTasks - len(ts))
                                     for ts in result.testingSearchTime ]
    result.searchTimes = [ ts + [enumerationTimeout]*(len(result.taskSolutions) - len(ts))
                               for ts in result.searchTimes ]

def updatePriors(result, path):
    jobs = set()
    numberOfChecks = 0
    numberOfPastChecks = 0
    maximumChecks = 10
    for frontierList in result.frontiersOverTime.values():
        for t,f in enumerate(frontierList):
            g = result.grammars[t]
            for e in f:
                jobs.add((e.program,f.task.request,g))
            if numberOfChecks < maximumChecks and t > 0:
                numberOfChecks += 1
                if abs(g.logLikelihood(f.task.request, e.program) - e.logPrior) < 0.001:
                    numberOfPastChecks += 1
    if numberOfPastChecks == numberOfChecks:
        print(f"Looks like {path} has already had its prior probabilities updated! Or does not use the neural network.")
        if False: # debugging
            for frontierList in result.frontiersOverTime.values():
                for t,f in enumerate(frontierList):
                    g = result.grammars[t]
                    for e in f:
                        print(t)
                        assert g.logLikelihood(f.task.request, e.program) == e.logPrior        
        return 
    print(f"About to update prior probabilities for {len(jobs)} program/grammar pairs")
    #print(f"WARNING: Will overwrite {path} with new prior probabilities once we are done")
    with timing("updated prior probabilities"):
        job2likelihood = batchLikelihood(jobs)
        for frontierList in result.frontiersOverTime.values():
            for t,f in enumerate(frontierList):
                g = result.grammars[t]
                for e in f:
                    e.logPrior = job2likelihood[(e.program, f.task.request, g)]
    if False:
        temporary = makeTemporaryFile()
        with open(temporary, 'wb') as handle:
            dill.dump(result, handle)
        os.system(f"mv {temporary} {path}")
        os.system(f"rm {temporary}")
                
def getCutOffHits(result, cutOff):
    """Return a list of hit percentages; currently only testing tasks supported"""
    from dreamcoder.likelihoodModel import add_cutoff_values
    from bin.examineFrontier import testingRegexLikelihood
    from dreamcoder.domains.regex.groundtruthRegexes import badRegexTasks
    
    tasks = [t for t in result.getTestingTasks()
             if t.name not in badRegexTasks]
    
    add_cutoff_values(tasks, cutOff)
    learningCurve = []
    while True:
        iteration = len(learningCurve)
        print(f"Calculating hit tasks for iteration {iteration}. We do this once per iteration and once per checkpoint so will take a while :(")
        hs = 0
        for ti,t in enumerate(tasks):
            if iteration >= len(result.frontiersOverTime[t]):
                assert ti == 0
                return learningCurve
            frontier = result.frontiersOverTime[t][iteration]
            if len(frontier) == 0: continue
            frontier = frontier.normalize()
            bestLikelihood = frontier.bestPosterior
            if cutOff == "gt":
                if arguments.testingLikelihood:
                    p = bestLikelihood.program
                    hs += int(testingRegexLikelihood(t, p) >= t.gt_test - 0.001)
                else:
                    hs += int(bestLikelihood.logLikelihood >= t.gt - 0.001)
            elif cutOff == "unigram" or cutOff == "bigram":
                assert False, "why are you not using a ground truth cut off"
                if bestLikelihood >= t.ll_cutoff: hs += 1
            else: assert False
        learningCurve.append(100.*hs/len(tasks))
            
stupidProgram = None
stupidRegex = None
def addStupidRegex(frontier, g):
    global stupidProgram
    global stupidRegex
    import pregex as pre
    
    if stupidProgram is None:
        from dreamcoder.domains.regex.regexPrimitives import reducedConcatPrimitives
        reducedConcatPrimitives()
        stupidProgram = Program.parse("(lambda (r_kleene (lambda (r_dot $0)) $0))")
        stupidRegex = stupidProgram.evaluate([])(pre.String(""))
        
    if any( e.program == stupidProgram for e in frontier ): return frontier.normalize()
    lp = g.logLikelihood(frontier.task.request, stupidProgram)    
    ll = sum(stupidRegex.match("".join(example)) for _,example in frontier.task.examples)
    fe = FrontierEntry(logPrior=lp, logLikelihood=ll, program=stupidProgram)
    return Frontier(frontier.entries + [fe],
                    task=frontier.task).normalize()
    
def getLikelihood(likelihood, result, task, iteration):
    from bin.examineFrontier import testingRegexLikelihood

    frontier = result.frontiersOverTime[task][iteration]
    frontier = addStupidRegex(frontier, result.grammars[iteration])

    if likelihood == "marginal":
        if arguments.testingLikelihood:
            return lse([ e.logPosterior + testingRegexLikelihood(task, e.program)
                         for e in frontier ])
        else:
            return lse([ e.logPosterior + e.logLikelihood
                         for e in frontier ])
        
    
    if likelihood == "maximum":
        assert not arguments.testingLikelihood
            
        return max(e.logLikelihood for e in frontier) if len(frontier) > 0 else 0.
    if likelihood == "MAP":
        if not arguments.testingLikelihood:
            return frontier.bestPosterior.logLikelihood if len(frontier) > 0 else 0.
        else:
            return testingRegexLikelihood(task, frontier.bestPosterior.program)
    if likelihood == "task":
        if arguments.testingLikelihood:
            return lse([testingRegexLikelihood(task, e.program) + e.logPrior
                        for e in frontier])
        else:
            return lse([e.logLikelihood + e.logPrior
                            for e in frontier])
    assert False
    
def getTestingLikelihood(likelihood, result, iteration):
    from dreamcoder.domains.regex.groundtruthRegexes import badRegexTasks
    testingTasks = [t for t in result.getTestingTasks()
                    if t.name not in badRegexTasks]

    print("Getting testing likelihoods; we have to do this once per checkpoint and once per iteration so hang on to your seat!")
    from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
    totalCharacters = sum( len(s)
        for t in testingTasks
        for _,s in regexHeldOutExamples(t))
    print("Total number of characters in testing tasks is",totalCharacters)    
    return sum(getLikelihood(likelihood, result, task, iteration)
               for task in testingTasks )/totalCharacters
def getTrainingLikelihood(likelihood, result, iteration):
    totalCharacters = sum(len(s)
                          for task in result.taskSolutions.keys()
                          for _,s in task.examples)
    return sum(getLikelihood(likelihood, result, task, iteration)
               for task in result.taskSolutions.keys() )/totalCharacters
    
def averageCurves(curves):
    xs = {x
          for xs,_ in curves
          for x in xs }
    xs = list(sorted(list(xs)))
    curves = [{x:y for x,y in zip(xs,ys) }
              for xs,ys in curves ]
    ys = []
    e = []
    for x in xs:
        y_ = []
        for curve in curves:
            if x in curve: y_.append(curve[x])
        mean = sum(y_)/len(y_)
        variance = sum((y - mean)**2 for y in y_ )/len(y_)
        sem = variance**0.5
        e.append(sem)
        ys.append(mean)
        
    return xs,ys,e

def medianCurves(curves):
    xs = {x
          for xs,_ in curves
          for x in xs }
    xs = list(sorted(list(xs)))
    curves = [{x:y for x,y in zip(xs,ys) }
              for xs,ys in curves ]
    ys = []
    tops = []
    bottoms = []
    for x in xs:
        y_ = []
        for curve in curves:
            if x in curve: y_.append(curve[x])
        ys.append(percentile(y_,0.5))
        tops.append(percentile(y_,0.75))
        bottoms.append(percentile(y_,0.25))
        
    return xs,ys,tops,bottoms

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

def showSynergyMatrix(results):
    # For each result, compile the total set of tasks that are ever solved by that run
    everSolved = []
    for r in results:
        everSolved.append({ t.name for t,f in r.allFrontiers.items() if not f.empty })
        N = len(r.allFrontiers)

    print("Of the",len(results),"checkpoints that you gave me, here is a matrix showing the overlap between the tasks solved:")

    for y in range(len(results)):
        if y == 0: print("".join( f"\tck{i + 1}" for i in range(len(results)) ))
        for x in range(len(results)):
            if x == 0: print("ck%d"%(y+1),
                             end="\t")
            intersection = len(everSolved[x]&everSolved[y])
            improvementOverBaseline = intersection/N
            print(int(improvementOverBaseline*100 + 0.5),
                  end="%\t")
        print()

    if len(results) == 3:
        print("Here's the percentage of tasks that are uniquely solved by the first checkpoint:")
        print(int(len(everSolved[0] - everSolved[1] - everSolved[2])/len(everSolved[0])*100 + 0.5),
              end="%")
        print()
    
def matplotlib_colors():
    from matplotlib import colors as mcolors
    return list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

def plotECResult(
        resultPaths,
        cutoff=None,
        likelihood=None,
        alpha=1.,
        onlyTime=False,
        xLabel=None,
        interval=False,
        timePercentile=False,
        labels=None,
        failAsTimeout=False,
        title=None,
        testingTimeout=None,
        export=None,
        showSolveTime=True,
        showTraining=False,
        iterations=None,
        maxP=110,
        showEpochs=False,
        colors=None,
        epochFrequency=1,
        averageColors=False,
        numTasks=False):
    assert not (onlyTime and not showSolveTime)
    if onlyTime: assert testingTimeout

    # we create a special key for caching
    # we concatenate this with the checkpoint name
    # and that gives us a magic filename where we will get the data
    cachingKey = (cutoff, likelihood, iterations, showTraining)


    # make a parallel list of colors
    colorNames = matplotlib_colors()
    currentColor = None
    results = []
    parameters = []
    for path in resultPaths:
        if path in colorNames or path[0] == '#':
            currentColor = path
            if colors is None:
                colors = []
            continue
        
        result = path
        print("will be loading path:", path)

        results.append(result)
        parameters.append(parseResultsPath(path))
        if currentColor is not None:
            colors.append(currentColor)

    
    f, a1 = plot.subplots(figsize=(arguments.width, arguments.height))
    if xLabel != "":
        a1.set_xlabel(xLabel or "Wake/Sleep Cycles", fontsize=LABELFONTSIZE)
    a1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if onlyTime:
        a1.set_ylabel('Search Time',
                      fontsize=LABELFONTSIZE)
        timeAxis = a1
        solveAxis = None
    else:
        if arguments.ylabel is not None:
            ylabel = arguments.ylabel
        elif likelihood is None:
            if numTasks:
                ylabel = 'Num: %s Solved%s'%("Training" if showTraining else "Test",
                                           " (solid)" if showSolveTime else "")
            else: 
                ylabel = '%% %s Solved%s'%("Training" if showTraining else "Test",
                                           " (solid)" if showSolveTime else "")
        elif likelihood == "maximum":
            ylabel = "log P(t|p^*)"
        elif likelihood == "marginal":
            if arguments.testingLikelihood:
                ylabel = "$\\leq\\log $P$($test$|$train$)$"
            else:
                ylabel = "log \\sum_p P(train|p)P(p|train,Lib)"
        elif likelihood == "task":
            ylabel = "$\\leq\\log $P$($tasks$|$Lib$)$"
        elif likelihood == "MAP":
            if arguments.testingLikelihood:
                ylabel = "log P(train|p*)"
            else:
                ylabel = "log P(test|p*)"            
        else:
            assert False
            
        a1.set_ylabel(ylabel,
                      fontsize=LABELFONTSIZE)
        solveAxis = a1
        if showSolveTime:
            a2 = a1.twinx()
            a2.set_ylabel('Solve time (dashed)', fontsize=LABELFONTSIZE)
            timeAxis = a2
        else:
            timeAxis = None

    assert iterations is not None, "you must specify number of iterations on commandline. ` -i <some number>`"
    n_iters = iterations

    plot.xticks(range(0, n_iters), fontsize=TICKFONTSIZE)

    if colors is None:
        assert not averageColors, "If you are averaging the results from checkpoints with the same color, then you need to tell me what colors the checkpoints should be. Try passing --colors ... or specifying the colors alongside --checkpoints ..."
        colors = ["#D95F02", "#1B9E77", "#662077", "#FF0000"] + ["#000000"]*100
    usedLabels = []

    #showSynergyMatrix(results)

    cyclesPerEpic = None
    plotCommands_solve = {} # Map from (color,line style) to (xs,ys) for a1
    plotCommands_time = {} # Map from (color,line style) to (xs,ys) for a2
    for result, p, color in zip(results, parameters, colors):
        path = result
        print("loading path:",result)
        cachingFileKey = "data/." + "gc_" + str(computeMD5hash(str((path,cachingKey))))
        if arguments.caching and os.path.exists(cachingFileKey):
            print("just kidding I found a cache file:")
            print(cachingFileKey)
            ys = loadPickle(cachingFileKey)
        else:
            result = loadfun(result)
            if likelihood is not None or cutoff is not None:
                if arguments.goodPrior:
                    print("WARNING: Skipping prior update - you better already have updated the priors!")
                else:
                    updatePriors(result, path)
            if testingTimeout is not None:
                result.testingSearchTime = [ [t for t in ts if t <= testingTimeout ]
                                             for ts in result.testingSearchTime ]

            if likelihood is None:
                if showTraining:
                    if numTasks:
                        ys = [t for t in result.learningCurve[:iterations]]
                    else:
                        ys = [100.*t/float(len(result.taskSolutions)) for t in result.learningCurve[:iterations]]
                else:
                    if cutoff is None:
                        if numTasks:
                            ys = [len(t) for t in result.testingSearchTime[:iterations]]
                        else:
                            ys = [100. * len(t) / result.numTestingTasks for t in result.testingSearchTime[:iterations]]
                    else:
                        ys = getCutOffHits(result, cutoff)[:iterations]
            else:
                ys = [(getTrainingLikelihood if showTraining else getTestingLikelihood)(likelihood, result, iteration)
                      for iteration in range(iterations) ]

        if arguments.caching and not os.path.exists(cachingFileKey):
            print("Updating cache")
            dumpPickle(ys,cachingFileKey)
            
        xs = list(range(0, len(ys)))
        if showEpochs:
            assert False, "showEpochs deprecated"
        if labels:
            if len(usedLabels) == 0 or usedLabels[-1][1] != color:
                usedLabels.append((labels[0], color))
                labels = labels[1:]

        plotCommands_solve[(color,'-')] = plotCommands_solve.get((color,'-'),[]) + [(xs,ys)]
        
        if showSolveTime:
            if onlyTime:
                for style in [':','-']:
                    if style == '-':
                        padSearchTimes(result, testingTimeout, p.enumerationTimeout)                        
                    if not showTraining: times = result.testingSearchTime[:iterations]
                    else: times = result.searchTimes[:iterations]
                    ys = [mean(ts) if not timePercentile else median(ts)
                          for ts in times]
                    plotCommands_time[(color,style)] = plotCommands_time.get((color,style),[]) + [(xs,ys)]
                    padSearchTimes(result, testingTimeout, p.enumerationTimeout)                    
            else:
                if failAsTimeout:
                    assert testingTimeout is not None
                    padSearchTimes(result, testingTimeout, p.enumerationTimeout)
                if not showTraining: times = result.testingSearchTime[:iterations]
                else: times = result.searchTimes[:iterations]

                ys = [mean(ts) if not timePercentile else median(ts)
                      for ts in times]
                plotCommands_time[(color,'--')] = plotCommands_time.get((color,'--'),[]) + [(xs,ys)]
            if interval and result is results[0]:
                assert not averageColors, "FIXME"
                a2.fill_between(xs,
                                [percentile(ts, 0.75) if timePercentile else mean(ts) + standardDeviation(ts)
                                 for ts in times],
                                [percentile(ts, 0.25) if timePercentile else mean(ts) - standardDeviation(ts)
                                 for ts in times],
                                facecolor=color, alpha=0.2)

    if averageColors:
        plotCommands_solve = {kl: averageCurves(curves)
                         for kl, curves in plotCommands_solve.items() }
        plotCommands_time = {kl: averageCurves(curves)
                         for kl, curves in plotCommands_time.items() }
        if solveAxis:
            for (color,ls),(xs,ys,es) in plotCommands_solve.items():
                solveAxis.fill_between(xs,
                                       [y - e for y,e in zip(ys,es)],
                                       [y + e for y,e in zip(ys,es)],
                                       facecolor=color,alpha=0.4)
            for (color,ls),(xs,ys,es) in plotCommands_solve.items():
                solveAxis.plot(xs,ys,color=color,ls=ls)
        if timeAxis:
            for (color,ls),(xs,ys,es) in plotCommands_time.items():
                timeAxis.errorbar(xs,ys,yerr=es,color=color,ls=ls)
    elif arguments.median:
        assert not timeAxis, "time not currently supported with median"
        assert solveAxis, "median only applies for solved tasks, not timing. use --percentile"
        plotCommands_solve = {kl: medianCurves(curves)
                              for kl, curves in plotCommands_solve.items() }
        
        for (color,ls),(xs,ys,ts,bs) in plotCommands_solve.items():
            solveAxis.fill_between(xs,bs,ts,facecolor=color,alpha=0.4)
        for (color,ls),(xs,ys,ts,bs) in plotCommands_solve.items():
            solveAxis.plot(xs,ys,color=color,ls=ls)
            
    else:
        if solveAxis:
            for (color,ls,xs,ys) in shuffled([ (color,ls,xs,ys)
                                            for (color,ls),cs in plotCommands_solve.items()
                                            for xs,ys in cs]):
                solveAxis.plot(xs,ys,color=color,ls=ls,alpha=alpha)
                # solveAxis.plot([0,iterations - 1],[float(bl)]*2,
                #                color='k')
        if timeAxis:
            for (color,ls,xs,ys) in shuffled([ (color,ls,xs,ys)
                                               for (color,ls),cs in plotCommands_time.items()
                                               for xs,ys in cs]):
                timeAxis.plot(xs,ys,color=color,ls=ls,alpha=alpha)

    if arguments.baselines:
        for n in range(len(arguments.baselines)//2):
            name = arguments.baselines[2*n]
            bl = arguments.baselines[2*n + 1]
            print("baseline",name,bl)
            name = name.replace(',','\n')
            bl = float(bl)
            if name.startswith('style'):
                name = name[len('style'):]
                name = name.split('!')
                color = name[0]
                if len(name) > 1:
                    style = name[1]
                    assert len(name) == 2
                else:
                    style = '-'
                plot.axhline(bl,-0.5,iterations,
                             color=color,lw=3,
                             linestyle=style,
                             zorder=-1.)
            else:
                plot.axhline(bl,-0.5,iterations,
                             color='k',lw=3,
                             zorder=-1.)
                solveAxis.text(iterations, bl, name, ha='left', va='center', fontweight='bold',
                               fontsize=TICKFONTSIZE)
        plot.subplots_adjust(right=0.9)

    if solveAxis and likelihood is None:
        a1.set_ylim(ymin=0, ymax=maxP)
        a1.yaxis.grid()
        a1.set_yticks(range(0, maxP, 20))
        plot.yticks(range(0, maxP, 20), fontsize=TICKFONTSIZE)

    cycle_label_frequency = 1
    if n_iters >= 10: cycle_label_frequency = 2
    if n_iters >= 20: cycle_label_frequency = 5
    for n, label in enumerate(a1.xaxis.get_ticklabels()):
        if n%cycle_label_frequency != 0:
            label.set_visible(False)

    if showEpochs: assert False

    if showSolveTime:
        timeAxis.set_ylim(ymin=0)
        starting, ending = timeAxis.get_ylim()
        ending10 = 10*int(ending/10 + 1)
        timeAxis.yaxis.set_ticks([ int(ending10/6)*j
                                   for j in range(0, 6)]) 
        for tick in timeAxis.yaxis.get_ticklabels():
            tick.set_fontsize(TICKFONTSIZE)

    if title is not None:
        plot.title(title, fontsize=TITLEFONTSIZE)

    if labels is not None:
        a1.legend(loc='lower right', fontsize=9,
                  fancybox=True, shadow=True,
                  handles=[mlines.Line2D([], [], color=color, ls='-',
                                         label=label)
                           for label, color in usedLabels])
    f.tight_layout()
    if export:
        plot.savefig(export)
        eprint("Exported figure ",export)
        if export.endswith('.png'):
            os.system('convert -trim %s %s' % (export, export))
        os.system('feh %s' % export)
    else:
        f.show()
        

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--checkpoints",nargs='+')
    parser.add_argument("--colors",nargs='+')
    parser.add_argument("--title","-t",type=str,
                        default="")
    parser.add_argument("--iterations","-i",
                        type=int, default=None,
                        help="number of iterations/epochs of EC to show. If combined with --showEpochs this will bound the number of epochs.")
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
    parser.add_argument("--x-label", dest="xLabel", default=None)
    parser.add_argument("--y-label", dest="ylabel", default=None)
    parser.add_argument("--showEpochs",
                        default=False, action="store_true",
                        help='X-axis is real-valued percentage of training tasks seen, instead of iterations.')
    parser.add_argument("--noTime",
                        default=False, action="store_true",
                        help='Do not show solve time.')
    parser.add_argument("--onlyTime",
                        default=False, action="store_true",
                        help='Only shows solve time and show both failAsTimeout time and actual time')
    parser.add_argument("--epochFrequency",
                        default=1, type=int,
                        help="Frequency with which to show epoch markers.")
    parser.add_argument("--averageColors",
                        default=False, action="store_true",
                        help="If multiple curves are assigned the same color, then we will average them")
    parser.add_argument("--alpha",
                        default=1., type=float,
                        help="Transparency of plotted lines")
    parser.add_argument("--likelihood",
                        type=str, choices=["maximum", "task",
                                           "marginal", "MAP"],
                        default=None)
    parser.add_argument("--cutoff",
                        type=str, choices=["bigram","unigram","gt"],
                        default=None)
    parser.add_argument("--testingLikelihood",
                        default=False, action='store_true')
    parser.add_argument("--goodPrior",
                        default=False, action='store_true',
                        help="Do not update priors when doing cutoffs and likelihoods")
    parser.add_argument("--baselines",
                        default=[],nargs='+')
    parser.add_argument("--palette","-c",
                        default=False, action='store_true')
    parser.add_argument("--median",
                        default=False, action='store_true',
                        help="Plots the median and 25/75 percentile of hits over time")
    parser.add_argument("--width",'-w',default=5,type=float)
    parser.add_argument("--height",'-H',default=2.5,type=float)
    parser.add_argument("--caching",
                        default=False, action='store_true',
                        help="try to cashe the calculation of the curves on disk")
    parser.add_argument("--numTasks",
                        default=False, action='store_true',
                        help="only show the number of tasks instead of percentile")



    arguments = parser.parse_args()

    if arguments.palette:
        # taken from https://learnui.design/tools/data-color-picker.html
        # we need to make sure that purple and teal are mapped onto colors that contrasts nicely
        # because we use these color schemes in other plots
        mapping = dict(zip(["teal","orange","purple","cyan",
                            "c1","c2","c3","c4","c5","c6"],
                           ["#003f5c","#ffa600","#ef5675","#7a5195",
                            "#003f5c","#444e86","#955196","#dd5182","#ff6e54","#ffa600"]))
        arguments.checkpoints = [mapping.get(ck,ck)
                                 for ck in arguments.checkpoints ]

    if arguments.likelihood: arguments.noTime = True

    assert not (arguments.median and arguments.averageColors), "cannot average colors and also take the median of the colors"
    
    plotECResult(arguments.checkpoints,
                 likelihood=arguments.likelihood,
                 cutoff=arguments.cutoff,
                 onlyTime=arguments.onlyTime,
                 xLabel=arguments.xLabel,
                 testingTimeout=arguments.testingTimeout,
                 timePercentile=arguments.percentile,
                 export=arguments.export,
                 title=arguments.title,
                 failAsTimeout=arguments.failAsTimeout,
                 labels=arguments.names.split(",") if arguments.names != "" else None,
                 interval=arguments.interval,
                 iterations=arguments.iterations,
                 showTraining=arguments.showTraining,
                 maxP=arguments.maxPercent,
                 showSolveTime=not arguments.noTime,
                 showEpochs=arguments.showEpochs,
                 epochFrequency=arguments.epochFrequency,
                 colors=arguments.colors,
                 alpha=arguments.alpha,
                 averageColors=arguments.averageColors,
                 numTasks=arguments.numTasks)
