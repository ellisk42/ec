from ec import *

import dill
import numpy as np

import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines


class Bunch(object):
    def __init__(self,d):
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
    p = p[:p.rfind('.')]
    domain = p[p.rindex('/')+1 : p.index('_')]
    rest = p.split('_')[1:]
    if rest[-1] == "baselines":
        rest.pop()
    parameters = { ECResult.parameterOfAbbreviation(k): eval(v)
                   for binding in rest
                   for [k,v] in [binding.split('=')] }
    parameters['domain'] = domain
    return Bunch(parameters)

def taskColor(task):
    n = task.name

    numberOfZeros = sum(c == "0" for c in n )
    if numberOfZeros == 0: return "r"
    if numberOfZeros == 1: return "y"
    if numberOfZeros == 2: return "y"
    if numberOfZeros == 3: return "y"
    if numberOfZeros == 4: return "y"
    assert False
    
    if "0x^4" not in n: return "r"
    if "0x^3" not in n: return "r"
    if "0x^2" not in n: return "g"
    return "g"

def PCAembedding(e, label = lambda l: l, color = lambda ll: 'b'):
    """e: a map from object to vector
    label: a function from object to how it should be labeled
    """
    primitives = e.keys()
    matrix = np.array([ e[p] for p in primitives ])
    N,D = matrix.shape

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale

    matrix = scale(matrix)
    solver = PCA(n_components = 2)
    matrix = solver.fit_transform(matrix)

    e = dict({p: matrix[j,:]
              for j,p in enumerate(primitives) })
    primitiveVectors = list(e.iteritems())
    
    plot.scatter([ v[0] for _,v in primitiveVectors ],
                 [ v[1] for _,v in primitiveVectors ],
                 c = [ color(p) for p,_ in primitiveVectors ])
    for p,v in primitiveVectors:
        l = label(p)
        if not isinstance(l,(str,unicode)): l = str(l)
        plot.annotate(l,
                      (v[0] + random.random(),
                       v[1] + random.random()))

def plotECResult(resultPaths, colors='rgbycm', label=None, title=None, export=None,
                 showSolveTime = False,
                 iterations = None):
    results = []
    parameters = []
    for j,path in enumerate(resultPaths):
        with open(path,'rb') as handle:
            result = dill.load(handle)
            if hasattr(result, "baselines") and result.baselines:
                for name, res in result.baselines.iteritems():
                    results.append(res)
                    p = parseResultsPath(path)
                    p["baseline"] = name.replace("_", " ")
                    parameters.append(p)
            else:
                results.append(result)
                p = parseResultsPath(path)
                parameters.append(p)

    # Collect together the timeouts, which determine the style of the line drawn
    timeouts = sorted(set( r.enumerationTimeout for r in parameters ),
                           reverse = 2)
    timeoutToStyle = {size: style for size, style in zip(timeouts,["-","--","-."]) }

    f,a1 = plot.subplots(figsize = (5,5))
    a1.set_xlabel('Iteration', fontsize = 22)
    a1.xaxis.set_major_locator(MaxNLocator(integer = True))

    if showSolveTime:
        a1.set_ylabel('% Tasks Solved (solid)', fontsize = 22)
    else:
        a1.set_ylabel('% Tasks Solved', fontsize = 22)

    if showSolveTime:
        a2 = a1.twinx()
        a2.set_ylabel('Avg solve time (dashed)', fontsize = 22)

    n_iters = max(len(result.learningCurve) for result in results)
    if iterations and n_iters > iterations: n_iters = iterations

    plot.xticks(range(0, n_iters), fontsize = 20)

    recognitionToColor = {False: "r", True: "b"}

    for result, p in zip(results, parameters):
        if hasattr(p, "baseline") and p.baseline:
            ys = [ 100. * result.learningCurve[-1] / len(result.taskSolutions) ]*n_iters
        else:
            ys = [ 100. * x / len(result.taskSolutions) for x in result.learningCurve[:iterations]]
        color = recognitionToColor[p.useRecognitionModel]
        l, = a1.plot(range(0, len(ys)), ys, color + timeoutToStyle[p.enumerationTimeout])
        # if label is not None:
        #     l.set_label(label(p))
        
        if showSolveTime:
            a2.plot(range(len(result.searchTimes[:iterations])),
                    [ sum(ts)/float(len(ts)) for ts in result.searchTimes[:iterations]],
                    color + '--')
            
    a1.set_ylim(ymin = 0, ymax = 110)
    a1.yaxis.grid()
    a1.set_yticks(range(0,110,20))
    plot.yticks(range(0,110,20),fontsize = 20)

    if showSolveTime:
        a2.set_ylim(ymin = 0)
        starting, ending = a2.get_ylim()
        a2.yaxis.set_ticks(np.arange(starting, ending, (ending - starting)/5.))

    if title is not None:
        plot.title(title, fontsize = 26)

    #if label is not None:
    legends = []
    legends.append(a1.legend(loc = 'lower right', fontsize = 14,
              #bbox_to_anchor=(1, 0.5),
              handles = [mlines.Line2D([],[],color = 'black',ls = timeoutToStyle[timeout],
                                       label = str(timeout))
                         for timeout in timeouts ]))
    if False:
        # FIXME: figure out how to have two separate legends
        plot.gca().add_artist(plot.legend(loc = 'lower left', fontsize = 20,
                                  handles = [mlines.Line2D([],[],color = recognitionToColor[True],ls = '-',
                                                           label = "DreamCoder"),
                                             mlines.Line2D([],[],color = recognitionToColor[False],ls = '-',
                                                           label = "No NN")]))

    f.tight_layout()
    if export:
        plot.savefig(export)#, additional_artists=legends)
        if export.endswith('.png'):
            os.system('convert -trim %s %s'%(export, export))
        os.system('feh %s'%export)
    else: plot.show()
    assert False

    for result in results:
        if hasattr(result, 'recognitionModel') and result.recognitionModel is not None:
            plot.figure()
            PCAembedding(result.recognitionModel.productionEmbedding(), label = prettyProgram)
            if export:
                export = export[:-4] + "_DSLembedding" + export[-4:]
                plot.savefig(export)
                os.system("feh %s"%(export))
            else: plot.show()
            plot.figure()
            tasks = result.taskSolutions.keys()
            PCAembedding(result.recognitionModel.taskEmbeddings(tasks),
                         label = lambda thing: thing,
                         color = taskColor) 
            if export:
                export = export[:-4] + "_task_embedding" + export[-4:]
                plot.savefig(export)
                os.system("feh %s"%(export))
            else: plot.show()

            if isinstance(result.recognitionModel.featureExtractor, RecurrentFeatureExtractor):
                plot.figure()
                colormap = {}
                for j in range(26): colormap[chr(ord('a') + j)] = 'b'
                for j in range(26): colormap[chr(ord('g') + j)] = 'g'
                for j in [" ",",",">","<"]: colormap[j] = 'r'
                
                PCAembedding(result.recognitionModel.featureExtractor.symbolEmbeddings(),
                             label = lambda thing: thing,
                             color = lambda thing: colormap.get(thing,'k'))
                plot.show()

def tryIntegerParse(s):
    try: return int(s)
    except: return None
    
if __name__ == "__main__":
    import sys
    def label(p):
        #l = p.domain
        l = ""
        if hasattr(p, 'baseline') and p.baseline:
            l += "baseline %s"%p.baseline
            return l
        if p.useRecognitionModel:
            if hasattr(p,'helmholtzRatio') and p.helmholtzRatio > 0:
                l += "DreamCoder"
            else:
                l += "AE"
        else: l += "no NN"
        if hasattr(p,"frontierSize"):
            l += " (frontier size %s)"%p.frontierSize
        else:
            l += " (timeout %ss)"%p.enumerationTimeout
        return l
    arguments = sys.argv[1:]
    export = [ a for a in arguments if a.endswith('.png') or a.endswith('.eps') ]
    export = export[0] if export else None
    title = [ a for a in arguments if not any(a.endswith(s) for s in {'.eps', '.png', '.pickle'})  ]

    # pass in an integer on the command line to  number of plotted iterations
    iterations = [ tryIntegerParse(a) for a in arguments if tryIntegerParse(a) ]
    iterations = None if iterations == [] else iterations[0]
    
    plotECResult([ a for a in arguments if a.endswith('.pickle') ],
                 export = export,
                 title = title[0] if title else "DSL learning curves",
                 label = label,
                 showSolveTime = True,
                 iterations = iterations)
