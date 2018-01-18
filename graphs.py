from ec import *

import matplotlib.pyplot as plot

def plotECResult(result, hitColor = 'r', descriptionLengthColor = 'b'):
    with open(result,'rb') as handle: result = pickle.load(handle)

    f,a1 = plot.subplots()
    a1.plot(range(1,len(result.learningCurve) + 1),
            result.learningCurve,
            hitColor + '-')
    a1.set_xlabel('Iteration')

    a1.set_ylabel('# Hit Tasks', color = hitColor)
    a1.tick_params('y',colors = hitColor)

    a2 = a1.twinx()
    a2.plot(range(1,len(result.averageDescriptionLength) + 1),
            result.averageDescriptionLength,
            descriptionLengthColor + '-')

    a2.set_ylabel('Average description length', color = descriptionLengthColor)
    a2.tick_params('y',colors = descriptionLengthColor)


    f.tight_layout()
    plot.show()


if __name__ == "__main__":
    plotECResult("experimentOutputs/polynomial_aic=1.0_arity=0_frontierSize=500_iterations=5_maximumFrontier=None_pseudoCounts=10.0_structurePenalty=0.001_topK=1_useRecognitionModel=False.pickle")
