try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *
from dreamcoder.domains.tower.towerPrimitives import ttower, tint, saveTowerImage, animateTower
import scipy.misc

import dill
import numpy as np
import sys

from matplotlib import pyplot





if __name__ == '__main__':
    from dreamcoder.domains.tower.makeTowerTasks import makeNewMaxTasks, makeBiasedTasks
    import scipy.misc
    ts = makeBiasedTasks()
    path = 'biasedTowerTasks/'
    #os.system(f"mkdir {path}")
    os.system(f"rm -f {path}*")
    for i,t in enumerate(ts):
        scipy.misc.imsave(path+str(i)+'.png', t.getImage()) 
    assert 0


    sys.setrecursionlimit(8000)
    n = 20
    ID = 'towers' + str(n)

    nameSalt = "towers"

    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value')]


    paths = [(f'experimentOutputs/{ID}Sample_SRE=True.pickle', 'Sample'),
        (f'experimentOutputs/{ID}RNN_SRE=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}REPL_SRE=True.pickle', 'REPL modular value')]

    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True.pickle', 'Sample'),
    #     (f'experimentOutputs/towers{n}SamplePolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/towers{n}LongSamplePolicyREPL_SRE=True_graph=True.pickle', 'REPL modular value')]
    # print("WARNING: using the long samplePolicy Repl run and rnn")

    # ID = 'towers' + str(n) + 'REPLPolicyHashing'
    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    #     (f'experimentOutputs/towers{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/towers{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value')]
    # print("WARNING: using the REPLPolicyHashing runs")


    # graph="_graph=True"
    # #mode="Prior"
    # nameSalt = "towersAstarCountNodes" #"BigramSamplePolicy" #
    # ID = 'towers' + str(n)
    # runType = "AstarCountNodes" #"BigramSamplePolicy" #
    # paths = [
    #     (f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle', 'Sample from prior only (no value)'),
    #     #(f'experimentOutputs/{ID}{runType}Symbolic_SRE=True{graph}.pickle', 'Symbolic value'),
    #     (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'REPL modular value'),
        
    #     ]

    # print("n is:")
    # print(n)


    paths, names = zip(*paths)

    with open(paths[0], 'rb') as h:
        rS = dill.load(h)

    with open(paths[1], 'rb') as h:
        rRNN = dill.load(h)

    with open(paths[2], 'rb') as h:
        rR = dill.load(h)
        
    from dreamcoder.showTowerTasks import showTowersAndSolutions, computeValue, testTask
    from dreamcoder.showTowerTasks import computeConfusionMatrixFromScores, graphPrecisionRecall
    # REPL3Max = [226, 536, 231, 444, 329, 246, 260, 163, 273, 180, 136, 259,
    #     168, 264, 170, 314, 257, 190, 364, 318, 278, 47, 281, 281, 80,
    #     108, 370, 208, 74, 135, 123, 244, 384, 120, 236, 409, 75, 139,
    #     486, 287, 296, 188, 231, 314, 196, 253, 245, 269, 379, 133, 213,
    #     270, 225, 151] 
    testingTasks = rS.getTestingTasks()
    assert testingTasks == rR.getTestingTasks()





    # path = 'grammarImages/'
    # g=rS.grammars[-1]
    # for i, (ll, tp, p) in enumerate(g.productions):
    #     if not p.isInvented: continue
    #     args = tp.functionArguments()

    #     d = {
    #         tint: '4',
    #         ttower: '$0',
    #         arrow(ttower, ttower): '(lambda (1x3 $0 ))',
    #         arrow(arrow(tint, ttower), ttower): '(lambda (lambda (1x3 $0 )))'
    #         }

    #     argStrs = [ d[t] for t in args]
    #     expr = Program.parse(f"( lambda ({p} {' '.join(argStrs)} ))")

    #     #saveTowerImage(path+str(i)+'_arg=4', expr)
    #     animateTower(path+str(i)+'_arg=4', expr)
    #     #scipy.misc.imsave(path+str(i)+'.png', executreTower(expr))


    assert 0


    
    SampleStats = rS.testingSearchStats[-1]
    REPLStats = rR.testingSearchStats[-1]
    RNNStats = rRNN.testingSearchStats[-1]

    SHitsRMisses = [t for t, lst in SampleStats.items() if ( lst != [] and REPLStats[t]==[] ) ]
    Smisses = [t for t, lst in SampleStats.items() if  lst == []]
    SHits = [t for t, lst in SampleStats.items() if ( lst != [] ) ]
    #for i in SHits: if 'Max' in i.name: print(i)
    RHits = [t for t, lst in REPLStats.items() if ( lst != [] ) ]
    # for t in SHits:
    #         print(t, SampleStats[t][0].evaluations, rR.testingNumOfProg[-1][t])
    # print()
    # count = 0
    # for t in RHits:
    #     print(t, REPLStats[t][0].evaluations)
    #     if rS.testingNumOfProg[-1][t] < REPLStats[t][0].evaluations:
    #         if SampleStats[t] == []:
    #             print("not reached by symbolic",rS.testingNumOfProg[-1][t] )
    #             count += 1

    # print(count)
    # #assert 0

    from dreamcoder.domains.tower.makeTowerTasks import makeMaxTasks

    maxTasks = makeMaxTasks()

    # for t in SHits:                                                                                                                                    
    #     if 'Max' in t.name:
    #         print(t.name)
    #         print(maxTasks.index(t))                                                                                                                                     
    #         print(SampleStats[t][0].program)   
    #         print(SampleStats[t][0].evaluations)                                                                                                                 
    #         if REPLStats[t] != []:
    #             print("\trepl solve in:", REPLStats[t][0].evaluations)
    #         else: print("\trepl not solve")
    #         print()
    # assert 0

    from dreamcoder.Astar import Astar
    from likelihoodModel import AllOrNothingLikelihoodModel


    rR.recognitionModel.solver = Astar(rR.recognitionModel)
    rS.recognitionModel.solver = Astar(rS.recognitionModel)


    ID = 'towers' + str(3)
    likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
    for i in range(4,5):
        tasks = [maxTasks[i]]
        task = tasks[0]
        print(task.name)
        g = rS.recognitionModel.grammarOfTask(task).untorch()
        g = rS.grammars[-1]
        ret = rR.recognitionModel.solver.infer(g, tasks, likelihoodModel, 
                                    timeout=1000000,
                                    elapsedTime=0,
                                    evaluationTimeout=0.01,
                                    maximumFrontiers={tasks[0]: 2},
                                    CPUs=1,
                                    )
        print("===================finished====================")
        #print(ret)
        # ret2 = rS.recognitionModel.solver.infer(g, tasks, likelihoodModel, 
        #                             timeout=1000000,
        #                             elapsedTime=0,
        #                             evaluationTimeout=0.01,
        #                             maximumFrontiers={tasks[0]: 2},
        #                             CPUs=1,
        #                             )



        print("task", i)
        print("repl hit?", list(ret[3].values())[0] != [] )
        #print("sample hit?", list(ret2[3].values())[0] != [] )
        print("num for repl", ret[2])
        #print("num for sample", ret2[2])
        print()

    assert 0

    # for i, t in enumerate(testingTasks):
    #     if t in Smisses: print(i)
    # assert 0
    # print("sample hit, repl miss:")
    # for i, t in enumerate(testingTasks):
    #     hitSample = bool(rS.testingSearchStats[0][t])
    #     hitREPL = bool(rR.testingSearchStats[0][t])
    #     if hitSample:
    #         sampleMin = rS.testingSearchStats[0][t][0].evaluations
    #         if sampleMin < rR.testingNumOfProg[-1][t] and not hitREPL:
    #             print(i)
    #             print("sampleMin:", sampleMin)
    #             print("max from repl", rR.testingNumOfProg[-1][t])

    # print("repl hit, sample miss:")
    # for i, t in enumerate(testingTasks):
    #     hitSample = bool(rS.testingSearchStats[0][t])
    #     hitREPL = bool(rR.testingSearchStats[0][t])

    #     if hitREPL:
    #         if not hitSample:
    #             print(i)
    #             print("not hit at all by sample")

    #         else:
    #             sampleMin = rS.testingSearchStats[0][t][0].evaluations
    #             if sampleMin >rR.testingNumOfProg[-1][t]:
    #                 print(i)
    #                 print("sampleMin", sampleMin)
    #                 print("max from repl", rR.testingNumOfProg[-1][t])

    # sHits = 0
    # replHits = 0
    # for i, t in enumerate(testingTasks):
    #     hitSample = bool(rS.testingSearchStats[0][t])
    #     hitREPL = bool(rR.testingSearchStats[0][t])
    #     if hitSample: sampleMin = rS.testingSearchStats[0][t][0].evaluations

    #     if hitSample and sampleMin <= rR.testingNumOfProg[-1][t]:
    #         sHits+=1
    #     if hitREPL: replHits += 1

    #     hitRNN = bool(rRNN.testingSearchStats[0][t])

    #     if hitREPL and not hitRNN and rR.testingSearchStats[0][t][0].evaluations > rRNN.testingNumOfProg[-1][t]:
    #         print("this task didn't have enough RNN evals", i)
    #         print()
    # print()
    # print("repl", replHits)
    # print("sample", sHits)


    # reductions = []
    # for i, t in enumerate(testingTasks):
    #     hitSample = bool(rS.testingSearchStats[0][t])
    #     hitREPL = bool(rR.testingSearchStats[0][t])

    #     if hitSample: sampleMin = rS.testingSearchStats[0][t][0].evaluations
    #     if hitREPL: REPLMin = rR.testingSearchStats[0][t][0].evaluations

    #     if hitSample and hitREPL:
    #         print("reduction factor:", sampleMin/REPLMin)
    #         print("num samples required", sampleMin)            
    #         reductions.append( (sampleMin, sampleMin/REPLMin))


    # reductions = sorted(reductions, key=lambda x: x[0])

    # print(reductions)
    # samples, reductionRate = zip(*reductions)


    # pyplot.plot(samples, reductionRate, marker='o', label='Sample/REPL')
    # # axis labels
    # pyplot.xlabel('num samples required')
    # pyplot.ylabel('reduction rate')
    # # show the plot
    # pyplot.savefig ('plots/reductionRateit20.png')


        # if hitSample:
        #     if rR.testingNumOfProg[-1][t] <= rS.testingSearchStats[0][t][0].evaluations:# \
        #         #and rS.testingSearchStats[0][t][0].evaluations <= 200:
        #         print(i)
        #         print("num enum:", rS.testingSearchStats[0][t][0].evaluations)
        #         print("max from sample", rR.testingNumOfProg[-1][t])

    # assert 0
    # showTowersAndSolutions(rR, "towersTasksREPL/")
    # assert 0
    # basePath = 'towersNames/'

    # for t in SHitsRMisses:
    #     print( testingTasks.index(t) )
    #     print(t)
    #     minEvals = min(res.evaluations for res in SampleStats[t] )
    #     print("minEvals:", minEvals)
    #     scipy.misc.imsave(basePath+'test'+t.name+'.png', t.getImage())

    #     print()

    RNNindices =  [20, 23, 25, 26, 27, 28, 29, 31, 50]
    RNNindices =  [20, 23, 25, 27, 29, 50]

    REPLindices = [19, 22, 23, 25, 26, 27, 28, 29, 31, 44, 50, 51]
    REPLindices = [19, 23, 25, 27, 29, 44, 50, 51] 
    

    p = 0
    n = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    tpV, fpV, fnV, tnV = [], [], [], []

    rS.recognitionModel.to_cpu()
    rR.recognitionModel.to_cpu()
    rRNN.recognitionModel.to_cpu()

    symbolicDataLst = []
    neuralDataLst = []
    rnnDataLst = []
    for i in range(len(testingTasks)):
        print(i)
        runs = testTask(rS, rR, rRNN, i, verbose=True, nSamples=10, usePrior=False)
        for run in runs:
            hit, nvs, cvs, rnnvs = run
            for n, c, rnn in zip(nvs, cvs, rnnvs):
                symbolicDataLst.append( (c, hit) )
                neuralDataLst.append( (n, hit) ) 
                rnnDataLst.append( (rnn, hit) ) 

    tp, tn, fp, fn = computeConfusionMatrixFromScores(symbolicDataLst, 0.5, normalize=False)
    print("symbolic confusion matrix:")
    print(f" true pos: {tp}   false pos: {fp}")
    print(f"false neg: {fn}    true neg: {tn}")

    print(f"recall: {tp/ (tp + fn )}")
    print(f"precision: {tp/ (tp + fp)}")

    for cutoff in [x/10. for x in range(1, 10) ]:
        tp, tn, fp, fn = computeConfusionMatrixFromScores(neuralDataLst, -math.log(cutoff), normalize=False)
        print()
        print(f"neural, with cutoff of {cutoff}")
        print(f"\trecall: {tp/ (tp + fn )}")
        print(f"\tprecision: {tp/ (tp + fp)}")


    path = 'plots/precisionRecallit3samp10postLoop.png'
    graphPrecisionRecall(symbolicDataLst, neuralDataLst, rnnDataLst, path, nSamp=500)
