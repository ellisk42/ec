from ec import *

import dill
import numpy as np

import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator

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

def plotECResult(resultPaths, colors='rgbycm', label=None, title=None, export=None, showLogLikelihood = False):
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

    f,a1 = plot.subplots(figsize = (5,2.5))
    a1.set_xlabel('Iteration')
    a1.xaxis.set_major_locator(MaxNLocator(integer = True))
    a1.set_ylabel('% Tasks Solved (solid)', fontsize = 11)

    if showLogLikelihood:
        a2 = a1.twinx()
        a2.set_ylabel('Avg log likelihood (dashed)', fontsize = 11)

    n_iters = max(len(result.learningCurve) for result in results)

    for color, result, p in zip(colors, results, parameters):
        if hasattr(p, "baseline") and p.baseline:
            ys = [ 100. * result.learningCurve[-1] / len(result.taskSolutions) ]*n_iters
        else:
            ys = [ 100. * x / len(result.taskSolutions) for x in result.learningCurve]
        l, = a1.plot(range(1, len(ys) + 1), ys, color + '-')
        if label is not None:
            l.set_label(label(p))
        if showLogLikelihood:
            a2.plot(range(1,len(result.averageDescriptionLength) + 1),
                    [ -l for l in result.averageDescriptionLength],
                    color + '--')
            
    a1.set_ylim(ymin = 0, ymax = 110)
    a1.yaxis.grid()
    a1.set_yticks(range(0,110,20))

    if showLogLikelihood:
        starting, ending = a2.get_ylim()#a2.set_ylim(ymax = 0)
        a2.yaxis.set_ticks(np.arange(starting, ending, (ending - starting)/5.))

    if title is not None:
        plot.title(title)

    if label is not None:
        a1.legend(loc = 'lower right', fontsize = 9)

    f.tight_layout()
    if export:
        plot.savefig(export)
        if export.endswith('.png'):
            os.system('convert -trim %s %s'%(export, export))
        os.system('feh %s'%export)
    else: plot.show()

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
                         label = lambda _: "",
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
                             label = lambda _: "",
                             color = lambda thing: colormap.get(thing,'k'))
                plot.show()


if __name__ == "__main__":
    import sys
    def label(p):
        #l = p.domain
        l = ""
        if hasattr(p, 'baseline') and p.baseline:
            l += " (baseline %s)"%p.baseline
            return l
        l += "frontier size %s"%p.frontierSize
        if p.useRecognitionModel:
            if hasattr(p,'helmholtzRatio') and p.helmholtzRatio > 0:
                l += " (DreamCoder)"
            else:
                l += " (AE)"
        else: l += " (no NN)"
        return l
    arguments = sys.argv[1:]
    export = [ a for a in arguments if a.endswith('.png') or a.endswith('.eps') ]
    export = export[0] if export else None
    title = [ a for a in arguments if not any(a.endswith(s) for s in {'.eps', '.png', '.pickle'})  ]
    plotECResult([ a for a in arguments if a.endswith('.pickle') ],
                 export = export,
                 title = title[0] if title else "DSL learning curves",
                 label = label)
