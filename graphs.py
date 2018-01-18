from ec import *

import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator

def plotECResult(results, colors = 'rgbky'):
    for j,result in enumerate(results):
        with open(result,'rb') as handle: results[j] = pickle.load(handle)

    f,a1 = plot.subplots()
    a2 = a1.twinx()
    for color, result in zip(colors, results):
        a1.plot(range(1,len(result.learningCurve) + 1),
                [ 100. * x / len(result.taskSolutions)
                  for x in result.learningCurve],
                color + '-')
        a1.set_xlabel('Iteration')
        a1.xaxis.set_major_locator(MaxNLocator(integer = True))

        a1.set_ylabel('% Hit Tasks')

        a2.plot(range(1,len(result.averageDescriptionLength) + 1),
                result.averageDescriptionLength,
                color + '--')

        a2.set_ylabel('Average description length (nats)')



    f.tight_layout()
    plot.show()


if __name__ == "__main__":
    import sys
    plotECResult(sys.argv[1:])
