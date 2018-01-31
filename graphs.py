from ec import *

import numpy as np

import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator

class Bunch(object):
    def __init__(self,d):
        self.__dict__.update(d)

relu = 'relu'
tanh = 'tanh'
sigmoid = 'sigmoid'

def parseResultsPath(p):
    p = p[:p.rfind('.')]
    domain = p[p.rindex('/')+1 : p.index('_')]
    parameters = { k: eval(v)
                   for binding in p.split('_')[1:]
                   for [k,v] in [binding.split('=')] }
    parameters['domain'] = domain
    return Bunch(parameters)

def PCAembedding(e):
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
    vectors = e.values()
    plot.scatter([ v[0] for v in vectors ],
                 [ v[1] for v in vectors ])
    for p,v in e.iteritems():
        plot.annotate(prettyProgram(p), v)
    
def plotECResult(results, colors = 'rgbky', label = None, title = None, export = None):
    parameters = []
    for j,result in enumerate(results):
        parameters.append(parseResultsPath(result))
        with open(result,'rb') as handle: results[j] = pickle.load(handle)

    f,a1 = plot.subplots(figsize = (5,4))
    a1.set_xlabel('Iteration')
    a1.xaxis.set_major_locator(MaxNLocator(integer = True))
    a1.set_ylabel('% Hit Tasks (solid)')
    a2 = a1.twinx()
    a2.set_ylabel('Avg log likelihood (dashed)')


    for j, (color, result) in enumerate(zip(colors, results)):
        l, = a1.plot(range(1,len(result.learningCurve) + 1),
                    [ 100. * x / len(result.taskSolutions)
                      for x in result.learningCurve],
                    color + '-')
        if label is not None:
            l.set_label(label(parameters[j]))

        a2.plot(range(1,len(result.averageDescriptionLength) + 1),
                [ -l for l in result.averageDescriptionLength],
                color + '--')

    a1.set_ylim(ymin = 0, ymax = 110)
    a1.yaxis.grid()
    a1.set_yticks(range(0,110,10))
    #a2.set_ylim(ymax = 0)
    
    if title is not None:
        plot.title(title)

    if label is not None:
        a1.legend(loc = 'lower right', fontsize = 9)
        
    f.tight_layout()
    if export:
        plot.savefig(export)
        os.system('convert -trim %s %s'%(export, export))
        os.system('feh %s'%export)
    else: plot.show()

    for result in results:
        if result.embedding is not None:
            plot.figure()
            PCAembedding(result.embedding)
            if export:
                export = export[:-len(".png")] + "_embedding.png"
                plot.savefig(export)
            else: plot.show()


if __name__ == "__main__":
    import sys
    def label(p):
        l = "%s, frontier size %s"%(p.domain, p.frontierSize)
        if p.useRecognitionModel:
            if hasattr(p,'helmholtzRatio') and p.helmholtzRatio > 0:
                l += " (neural Helmholtz)"
            else:
                l += " (neural)"
        return l
    arguments = sys.argv[1:]
    export = [ a for a in arguments if a.endswith('.png') ]
    export = export[0] if export else None
    title = [ a for a in arguments if not any(a.endswith(s) for s in ['.png','.pickle'])  ]
    plotECResult([ a for a in arguments if a.endswith('.pickle') ],
                 export = export,
                 title = title[0] if title else "DSL learning curves",
                 label = label)
