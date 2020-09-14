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
from dreamcoder.domains.rb.rbPrimitives import robustFillPrimitives

robustFillPrimitives()

import dill
import numpy as np

exclude_lst = [
    "arch leg 1",
    "arch leg 2",
    "arch leg 6",
    "arch leg 8",
    "bridge (2) of arch 2",
    "bridge (2) of arch 4",
    "bridge (2) of arch 5",
    "bridge (3) of arch 3",
    "bridge (3) of arch 5",
    "bridge (4) of arch 1",
    "bridge (4) of arch 4",
    "bridge (5) of arch 3",
    "bridge (5) of arch 4",
    "bridge (5) of arch 5",
    "bridge (6) of arch 4",
    "bridge (6) of arch 5",
    "bridge (7) of arch 1",
    "bridge (7) of arch 4",
    "arch 1/2 pyramid 2",
    "brickwall, 3x1",
    "brickwall, 3x3",
    "brickwall, 3x4",
    "brickwall, 4x2",
    "brickwall, 4x4",
    "brickwall, 5x2",
    "brickwall, 6x1",
    "brickwall, 6x4",]

FILTER_OUT = [
        (('Karrie.Covelli.882.+129', 'Bogle.Jani', 'H.+7.Cortes.+169', 'Ducati125.of.588.843'), ('Covelli', 'Jani', '+7', 'of') ),

        ( ('UC)+176)Jeanice)+174', 'Mariel)Carlene)Ducati100)Jeff', 'R)+144', 'Partida)FreeHafer)+130)D'), ('+176', 'Carlene', '+144', 'FreeHafer') ),

        ( ('+155-Covelli-Constable-405', 'Kotas-028', 'Launa-Hornak', '504-Jeanice-K'), ('Covelli', '028', 'Hornak', 'Jeanice') ),

        ( ('365 Aylward', 'Celsa Latimore', '438 20 MA FreeHafer', '6 Kimberley 095'), ('3A', 'CL', '42MF', '6K0') ),

        ( ('Honda550 +180 Q', '46 439', '751 Drexel L J', 'Dr UC K Rowden'), ('H+Q', '44', '7DLJ', 'DUKR') ),

        ( ('+155 174 Dr Haven', '888 Penn 50 UC', '43 390 Phillip', '21 B'), ('+1DH', '8P5U', '43P', '2B') ),

        ( ('856 +138 424 Montiel', 'Trinidad 311 33', 'California 86', 'O Jeanice'), ('(+138)', '(311)', '(86)', '(Jeanice)') ),

        #( ('Lain-Edison.C-Temple', '-Spell.Rowden Arbor', '9-Ducati125.976.Alida', '-Haven.80'), ('Lain-(Edison).C-Temple', '-(Spell).Rowden Arbor', '9-(Ducati125).976.Alida', '-(Haven).80') ),

        ( ('Madelaine +189Andria', 'Hornak 575 MA JacquilineAndria', '+68 +161 Heintz York', '+13 20 +7'), ('Madelaine +189Andria', 'Hornak 575 MA JacquilineAndria', '+68 +161 Heintz YorkAndria', '+13 20 +7Andria') ),

        ( ('Marcus +108 Ramthun Rudolf', 'Hopkins 701 F', '+163 +129997', 'Quashie Miah'), ('Marcus +108 Ramthun Rudolf997', 'Hopkins 701 F997', '+163 +129997', 'Quashie Miah997') ),

        ( ('520 T769', 'Ducati125 A Eccleston +198769', '+169 +163 +129 46', 'Andria +140 Spell'), ('520 T769', 'Ducati125 A Eccleston +198769', '+169 +163 +129 46769', 'Andria +140 Spell769') ),

        ( ('Ducati250 HoustonScalia', 'Ramthun Beata Chism FreeHaferScalia', 'UIUC 526', 'Angeles T N'), ('Ducati250 HoustonScalia', 'Ramthun Beata Chism FreeHaferScalia', 'UIUC 526Scalia', 'Angeles T NScalia') ),

        ( ('Bogle Miah Honda250 Trinidad', 'Ghoston Bobo Scalia Chism', 'Annalisa Latimore ChismRamthun', '107 CollegeRamthun'), ('Bogle Miah Honda250 TrinidadRamthun', 'Ghoston Bobo Scalia ChismRamthun', 'Annalisa Latimore ChismRamthun', '107 CollegeRamthun') ),

        ( ('+194 517 Bobo568', '+23 10 IL 844', '+47 P568', 'MD Hopkins 394'), ('+194 517 Bobo568', '+23 10 IL 844568', '+47 P568', 'MD Hopkins 394568') ),

        ( ('158 Quashie Hage', '647 Seamons 40 Teddy', 'Ferrari250 +58 AndrewColumbia', 'Cambridge MD 875 Ducati125'), ('158 Quashie HageColumbia', '647 Seamons 40 TeddyColumbia', 'Ferrari250 +58 AndrewColumbia', 'Cambridge MD 875 Ducati125Columbia') ),
    ]



def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None, mode='fractionHit', maxLens=[],maxEvals=10000000):
    import matplotlib.pyplot as plot

    def averageLoss(n, predicate):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        losses = [ min([defaultLoss] + [1 - math.exp(r.loss) for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    def fractionHit(n, predicate):
        """simply plots fraction of tasks hit at all"""
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if predicate(r)]
                    for rs in results ]
        hits = [ rs != [] for rs in results ]
        return sum(hits)/float(len(hits))*100

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('percent correct')
    if mode =='fractionHit': plot.ylim(bottom=0., top=100.)
    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        if mode =='fractionHit':
            plot.plot(xs, [fractionHit(n,lambda r: r.time < x) for x in xs],
                  label=names[n], linewidth=4)
            #plot.xscale('log')
        else:
            plot.plot(xs, [averageLoss(n,lambda r: r.time < x) for x in xs],
                  label=names[n], linewidth=4)
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('percent correct')
    if mode =='fractionHit': plot.ylim(bottom=0., top=100.)
    for n in range(len(testResults)):
        #xs = list(range(max([0]+[r.evaluations for tr in testResults[n] for r in tr] ) + 1))
        xs = list(range(min(maxLens[n], maxEvals)))
        if mode =='fractionHit':
            plot.plot(xs, [fractionHit(n,lambda r: r.evaluations <= x) for x in xs],
                  label=names[n], linewidth=4)
            #plot.xscale('log')
        else:
            plot.plot(xs, [averageLoss(n,lambda r: r.evaluations <= x) for x in xs],
                  label=names[n], linewidth=4)
    plot.legend()
    if export:
        plot.savefig(f"{export}_evaluations.eps")
    else:
        plot.show()
        

if __name__ == '__main__':

    n = 3
    ID = 'towers' + str(n)

    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}RNN_SRE=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}REPL_SRE=True.pickle', 'REPL modular value')]

    # nameSalt = "towersLong"
    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}LongREPL_SRE=True_graph=True.pickle', 'REPL modular value (longer)')]

    # nameSalt = "towersSamplePolicy"
    # paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    #     (f'experimentOutputs/{ID}SamplePolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
    #     (f'experimentOutputs/{ID}SamplePolicyREPL_SRE=True_graph=True.pickle', 'REPL modular value')]

    # ****
    nameSalt = "towersREPLPolicy"
    paths = [(f'experimentOutputs/{ID}SamplePolicySample_SRE=True_graph=True.pickle', 'Sample (no value)'),
        (f'experimentOutputs/{ID}REPLPolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value'),
        (f'experimentOutputs/{ID}REPLPolicySymbolic_SRE=True_graph=True.pickle', 'Symbolic value')]

    nameSalt = "towersLongSamplePolicy"
    paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample (no value)'),
        (f'experimentOutputs/{ID}SamplePolicyRNN_SRE=True_graph=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}LongSamplePolicyREPL_SRE=True_graph=True.pickle', 'REPL modular value (longer)'),
        (f'experimentOutputs/{ID}Symbolic_SRE=True_graph=True.pickle', 'Symbolic value')]


    nameSalt = "towersSamplePolicyHashing"
    ID = 'towers' + str(n) + 'SamplePolicyHashing'
    paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
        (f'experimentOutputs/{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value'),
        (f'experimentOutputs/{ID}Symbolic_SRE=True_graph=True.pickle', 'Symbolic value'),
        ]
    #print("WARNING: using the REPLPolicyHashing runs")

    graph="_graph=True"
    #mode="Prior"
    nameSalt = "SMCOracle" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    ID = 'towers' + str(n)
    runType ="SMC" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    paths = [
        (f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle', 'Policy only (no value)'),
        (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN value'),
        (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'REPL modular value'),
        (f'experimentOutputs/towers20SMCSemiOracle_SRE=True_graph=True.pickle', 'Oracle Value fun'),
        (f'experimentOutputs/{ID}{runType}Symbolic_SRE=True{graph}.pickle', 'Symbolic value')
        ]

  
    graph=""
    nameSalt = "AstarPseudoResult" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    ID = 'towers' + str(n)
    runType ="PolicyOnlyPseudoResult" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    paths = [
        (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'Abstract REPL policy (ours)'),
        (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN Policy'),
        (f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle', 'Bigram Policy'),
        ]


    graph="_graph=True"
    nameSalt = "Astar" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    ID = 'rb'
    runType ="PolicyOnly" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    paths = [
        (f'experimentOutputs/{ID}{runType}REPL_SRE=True{graph}.pickle', 'Abstract REPL policy (ours)'),
        (f'experimentOutputs/{ID}{runType}RNN_SRE=True{graph}.pickle', 'RNN Policy'),
        (f'experimentOutputs/{ID}{runType}Bigram_SRE=True{graph}.pickle', 'Bigram Policy'),
        ]

    graph="_graph=True"
    nameSalt = "SMCpseudoWithSynthBaselines" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    ID = 'rb'
    runType ="PolicyOnly" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    maxEvals = 5000
    useMaxLens = True
    paths = [
        (f'experimentOutputs/rbPolicyOnlyPseudoResultSMCREPL_SRE=True.pickle', 'Abstract REPL policy (ours)'),
        (f'experimentOutputs/rbPolicyOnlyPseudoResultSMCRNN_SRE=True.pickle', 'RNN Policy'),
        #(f'experimentOutputs/rbPolicyOnlyPseudoResultSMCBigram_SRE=True.pickle', 'Bigram Policy'),
        (f'experimentOutputs/rbPolicyOnlyPseudoResultSMCREPLNoConcrete_SRE=True.pickle', 'modular without concrete semantics'),
        (f'robustfill_baseline_results.p', 'RobustFill')
        ]


    # graph=""
    # nameSalt = "AstarPseudoResultBoth" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    # ID = 'towers' + str(n)
    # runType ="PseudoResult" #"Helmholtz" #"BigramAstarCountNodes" #"BigramSamplePolicy" #
    # maxEvals = 20000
    # useMaxLens = True
    # paths = [
    #     (f"experimentOutputs/towers3PolicyOnlyPseudoResultRNNRLValue=Falsecontrastive=Falseseperate=True_SRE=True.pickleDebugnoConcrete", "modular, no repl"),
    #     (f'experimentOutputs/{ID}PolicyOnly{runType}REPL_SRE=True{graph}.pickle', 'Abstract REPL policy only (weights not shared w value)'),
    #     #(f'experimentOutputs/{ID}{runType}REPLRLValue=True_SRE=True{graph}.pickleDebug', 'Abstract REPL policy + Value'),
    #     #(f'experimentOutputs/{ID}{runType}REPLRLValue=False_SRE=True{graph}.pickleDebug', 'Abstract REPL policy only (weights shared w value)'),
    #     #(f'experimentOutputs/{ID}{runType}RNNRLValue=True_SRE=True{graph}.pickleDebug', 'RNN Policy + Value'),
    #     #(f'experimentOutputs/{ID}{runType}RNNRLValue=False_SRE=True{graph}.pickleDebug', 'RNN Policy only (weights shared w value)'),
    #     (f'experimentOutputs/{ID}PolicyOnly{runType}RNN_SRE=True{graph}.pickle', 'RNN Policy only (weights not shared w value)'),
    #     #(f'experimentOutputs/{ID}PolicyOnly{runType}REPLRLValue=Falsecontrastive=True_SRE=True{graph}.pickleDebug', 'Abstract REPL policy + Value (contrastive value training)'),
    #     #(f'experimentOutputs/{ID}PolicyOnly{runType}RNNRLValue=Falsecontrastive=True_SRE=True{graph}.pickleDebug', 'RNN Policy + Value (contrastive value training)'),
    #     #(f"experimentOutputs/towers3PolicyOnlyPseudoResultREPLRLValue=Truecontrastive=Falseseperate=True_SRE=True.pickleDebug", "Abstract REPL policy + RL value (seperate weights)"),
    #    # (f"experimentOutputs/towers3PolicyOnlyPseudoResultRNNRLValue=Truecontrastive=Falseseperate=True_SRE=True.pickleDebug", "RNN policy + RL value (seperate weights)"),
    #    (f"experimentOutputs/towers3PolicyOnlyPseudoResultREPLRLValue=Falsecontrastive=Falseseperate=True_SRE=True.pickleDebugreplLongTest", "Abstract REPL (ours, long run)"),
    #     ]





    with open('biasedtasks.p', 'rb') as h: biasedtasks = dill.load(h)
    timeout=120
    outputDirectory = 'plots'
    paths, names = zip(*paths)

    for mode in ['test']: #['test', 'train']:

        testResults = []
        maxLens = []
        for path in paths:
            with open(path, 'rb') as h:
                r = dill.load(h)

            #optimize for speed

            for task, results in r.testingSearchStats[-1].items():
                if r.testingSearchStats[-1][task]:
                    r.testingSearchStats[-1][task] = r.testingSearchStats[-1][task][:1]

            from dreamcoder.domains.rb.main import makeTasks, makeOldTasks
            rbHardTasks = [ t.name for t in makeTasks()]

            challenge = [t.name for t in makeOldTasks(synth=False) ]
            print("length challenge,", len(challenge))

            delTasks = []
            seenBridges = False

            tnames = []
            for ins, outs in FILTER_OUT:
                examples = list(zip(ins, outs))
                name = str(examples[0])
                tnames.append(name)

            for task, results in r.testingSearchStats[-1].items():

                if "from bridges" in task.name: 
                    if seenBridges: delTasks.append(task)
                    seenBridges = True
                if r.testingSearchStats[-1][task] and r.testingSearchStats[-1][task][0].evaluations < 100:
                    print(task.name)
                if "pyramid on top" in task.name: delTasks.append(task)
                if task.name in exclude_lst: delTasks.append(task)


                if task.name in tnames:  delTasks.append(task)

                #if task.name not in challenge: delTasks.append(task)

            for task in delTasks: del r.testingSearchStats[-1][task]

            # print(path)
            # for task, results in r.testingSearchStats[-1].items():
            #     if "Max twoArches" in task.name and r.testingSearchStats[-1][task]:
            #         print(task.name, r.testingSearchStats[-1][task][0].evaluations)
            #         print(r.testingSearchStats[-1][task][0].program)


            print("ntasks", len(r.testingSearchStats[-1].items()))

            if hasattr(r, 'testingNumOfProg'):
                minN = float('inf')
                for task, results in r.testingSearchStats[-1].items():
                    if not results:
                        minN = min(minN, r.testingNumOfProg[-1][task])

                print("min of max N prog searched is", minN  )
                maxLens.append(minN)

            #import pdb; pdb.set_trace()
            res = r.searchStats[-1] if mode=='train' else r.testingSearchStats[-1]

            # for i, (task, v) in enumerate(res.items()):
            #     print(i, "task", task, "res", bool(v))

            # assert 0
            testResults.append( list(res.values()) )

        plotTestResults(testResults, timeout,
                        defaultLoss=1.,
                        names=names,
                        export=f"{outputDirectory}/{nameSalt}{ID}{mode}_curve.eps",
                        mode='fractionHit',
                        maxLens=maxLens if useMaxLens else [maxEvals] * len(testResults),
                        maxEvals=maxEvals)
