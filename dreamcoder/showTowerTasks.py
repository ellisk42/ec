
import scipy.misc
from dreamcoder.domains.tower.towerPrimitives import ttower, arrow
from dreamcoder.utilities import timing, mean
import time
import math
from matplotlib import pyplot

basePath = "towerTasks/"

def showTowersAndSolutions(r, basePath ):

    tasks = list(r.taskSolutions.keys())

    testingTasks = r.getTestingTasks()


    for i, t in enumerate(tasks):

        hit = bool(r.searchStats[0][t])
        print(f"""
TASK ID: {i}
    task name: {t.name}
    hit with my testing situation? {hit}
    first task solution (from kevin): {r.taskSolutions[t].entries}
""")

        hitstr = "KevinHIT" if r.taskSolutions[t].entries else "KevinMISS"
        scipy.misc.imsave(basePath+'train'+str(i)+hitstr+'.png', t.getImage())



    for i, t in enumerate(testingTasks):

        hit = bool(r.testingSearchStats[0][t])
        print(f"""
TEST TASK ID: {i}
    task name: {t.name}
    Max number of prog checked: {r.testingNumOfProg[-1][t]}
    hit with my testing situation? {hit}
""")

        if hit:
            solution = r.testingSearchStats[0][t][0].program
            print("found solution:", solution)
            print("number of prog enum before hit", r.testingSearchStats[0][t][0].evaluations)
        #print(f"task solutions (from kevin): {r.recognitionTaskMetrics[t].get('frontier', 'not solved by kevin')}")
        hitstr = "HIT" if hit else "MISS"
        scipy.misc.imsave(basePath+'test'+str(i)+hitstr+'.png', t.getImage())


def computeValue(r):
    recModel = r.recognitionModel
    task = list(r.taskSolutions.keys())[38]
    print(task.name)
    g = recModel.grammarOfTask(task)
    prog = r.taskSolutions[task].bestPosterior.program
    print(prog)
    tp = arrow(ttower, ttower)
    ll = g.logLikelihood(tp, prog)
    print(ll)
    from dreamcoder.zipper import sampleSingleStep,baseHoleOfType,findHoles
    h = baseHoleOfType(tp)
    zippers = findHoles(h, tp)
    print(h)
    for i in range(10):
        newOb = h
        newZippers = zippers
        for i in range(4):
            newOb, newZippers = sampleSingleStep(g, newOb, tp, holeZippers=newZippers, maximumDepth=8)
        print(newOb)
        #print(newZippers)
    from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
    lm = AllOrNothingLikelihoodModel()
    print(lm.score(newOb, task))
    print(prog == newOb)
    print(r.taskSolutions[task].bestPosterior)
    logLikelihood = task.logLikelihood(prog, None)
    print(logLikelihood)
    #import pdb; pdb.set_trace()


def testTask(rS, rR, rRNN, i, verbose=True, nSamples=1):
    
    from dreamcoder.symbolicAbstractTowers import SymbolicAbstractTowers
    concreteHead = SymbolicAbstractTowers()

    tasks = rS.getTestingTasks()
    task = tasks[i]



    if verbose: print(i)
    if verbose: print(task.name)
    gS = rS.recognitionModel.grammarOfTask(task)
    gR = rR.recognitionModel.grammarOfTask(task)
    gRNN = rRNN.recognitionModel.grammarOfTask(task)
    
    SampleStats = rS.testingSearchStats[-1]
    REPLStats = rR.testingSearchStats[-1]
    RNNStats = rRNN.testingSearchStats[-1]

    if verbose: print(len(SampleStats[task]))
    # if len(SampleStats[task]) == 0: 
    #     return 0, 0

    tp = arrow(ttower, ttower)    

    if SampleStats[task]: #if hit task before:
        for j in range(1):
            prog = SampleStats[task][j].program
            if verbose: print('\t', prog)
            llS = gS.logLikelihood(tp, prog)
            if verbose: print(llS)
            llR = gR.logLikelihood(tp, prog)
            if verbose: print(llR)
            if verbose: print()

    #return llS, llR

    from dreamcoder.zipper import sampleSingleStep,baseHoleOfType,findHoles

    h = baseHoleOfType(tp)
    zippers = findHoles(h, tp)
    if verbose: print(h)
    runs = []
    valueTimes = []
    concreteTimes = []
    rnnTimes = []
    for _ in range(nSamples):
        newOb = h
        newZippers = zippers
        concreteValues=[]
        neuralValues=[]
        rnnValues=[]
        while newZippers:
            newOb, newZippers = sampleSingleStep(gS, newOb, tp, holeZippers=newZippers, maximumDepth=8)
            
            t = time.time()
            value = rR.recognitionModel.valueHead.computeValue(newOb, task)
            dt = time.time() - t
            valueTimes.append( dt )
            if newZippers: neuralValues.append(value)

            t = time.time()
            RNNValue = rRNN.recognitionModel.valueHead.computeValue(newOb, task)
            dt = time.time() - t
            rnnTimes.append(dt)
            if newZippers: rnnValues.append(RNNValue )
        
            t = time.time()
            concreteValue = concreteHead.computeValue(newOb, task)
            dt = time.time() - t
            concreteTimes.append(dt)

            if newZippers: concreteValues.append(concreteValue)
            if verbose: print('\t',newOb)
            if verbose: print("value", value)
            if verbose: print("concrete value", concreteValue)
            if verbose: print("rnn value", RNNValue)
        #print(newOb)
        if verbose: print()
        logLikelihood = task.logLikelihood(newOb, None)
        if verbose: print("task likelihood", logLikelihood)
        if verbose: print()
        if verbose: print()
        if logLikelihood == 0.0: assert all([v == 0.0 for v in concreteValues])
        for j in range(1, len(concreteValues)):
            assert not concreteValues[j-1] > concreteValues[j], str(i)

        runs.append( (logLikelihood == 0.0, neuralValues, concreteValues, rnnValues) )
    
    valueTime = mean( valueTimes )
    concreteTime = mean( concreteTimes )
    rnnTime = mean(rnnTimes)

    print(f"average value times: {valueTime}")
    print(f"average symbolic times: {concreteTime}")
    print(f"average rnn times: {rnnTime}")
    return runs



def computeConfusionMatrixFromScores(dataLst, cutoff, normalize=False):
    #dataLst is a list of tuples (score, hitBool)
    #cutoff should be a negative log likelihood, because that's what scores are
    tp, tn, fp, fn = 0,0,0,0

    for score, hit in dataLst:
        if hit:
            if score <= cutoff: #todo eq?
                tp += 1
            else: fn += 1
        else:
            if score <= cutoff:
                fp += 1
            else: tn += 1

    if normalize:
        tpr = tp/ (tp + fn)
        tnr = tn/ (tn + fp)
        fpr = fp/ (fp + tn)
        fnr = fn/ (fn + tp)
        return tpr, tnr, fpr, fnr

    return tp, tn, fp, fn 


def precisionAndRecall(dataLst, cutoff):

    tp, tn, fp, fn = computeConfusionMatrixFromScores(dataLst, -math.log(cutoff))
    precision = tp/ (tp + fp)
    recall = tp/ (tp + fn )

    return precision, recall

def graphPrecisionRecall(symbolicDataLst, neuralDataLst, rnnDataLst, path, nSamp=500):
    
    symb_precision, symb_recall = precisionAndRecall(symbolicDataLst, 0.5)

    cutoffs = [x/float(nSamp) for x in range(1, nSamp)]
    repl_precision, repl_recall = zip(*  [precisionAndRecall(neuralDataLst, cutoff) for cutoff in cutoffs]) 
    rnn_precision, rnn_recall = zip(*  [precisionAndRecall(rnnDataLst, cutoff) for cutoff in cutoffs])
    
    pyplot.plot(rnn_recall, rnn_precision, marker='.', label='RNN value')
    pyplot.plot(repl_recall, repl_precision, marker='.', label='neural modular REPL value')
    pyplot.plot([symb_recall], [symb_precision], linestyle='--', marker='o', label='Symbolic abstraction')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.savefig (path)
    #pyplot.show()



