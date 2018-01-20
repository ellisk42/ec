from ec import *

import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator

class Bunch(object):
    def __init__(self,d):
        self.__dict__.update(d)

def parseResultsPath(p):
    p = p[:p.rfind('.')]
    domain = p[p.index('/')+1 : p.index('_')]
    parameters = { k: eval(v)
                   for binding in p.split('_')[1:]
                   for [k,v] in [binding.split('=')] }
    parameters['domain'] = domain
    return Bunch(parameters)

def plotECResult(results, colors = 'rgbky', label = None, title = None):
    parameters = []
    for j,result in enumerate(results):
        parameters.append(parseResultsPath(result))
        with open(result,'rb') as handle: results[j] = pickle.load(handle)

    f,a1 = plot.subplots()
    a1.set_xlabel('Iteration')
    a1.xaxis.set_major_locator(MaxNLocator(integer = True))
    a1.set_ylabel('% Hit Tasks (solid)')
    a2 = a1.twinx()
    a2.set_ylabel('Avg description length in nats (dashed)')


    for j, (color, result) in enumerate(zip(colors, results)):
        l, = a1.plot(range(1,len(result.learningCurve) + 1),
                    [ 100. * x / len(result.taskSolutions)
                      for x in result.learningCurve],
                    color + '-')
        if label is not None:
            l.set_label(label(parameters[j]))

        a2.plot(range(1,len(result.averageDescriptionLength) + 1),
                result.averageDescriptionLength,
                color + '--')

    a1.set_ylim(ymin = 0, ymax = 110)
    a1.set_yticks(range(0,110,10))
    a2.set_ylim(ymin = 0)
    
    if title is not None:
        plot.title(title)

    if label is not None:
        a1.legend(loc = 'best')
        
    f.tight_layout()
    plot.show()


if __name__ == "__main__":
    import sys
    plotECResult(sys.argv[1:],
                 title = "DSL learning curves",
                 label = lambda p: "%s, frontier size %s%s"%(p.domain, p.frontierSize,
                                              " (neural)" if p.useRecognitionModel else ""))
