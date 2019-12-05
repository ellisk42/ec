## summarize, especialyl comparing model to human.
import sys

sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")

from analysis.getModelHumanDists import loadDistances, filterDistances
from analysis.parse import getBestFrontierProgram
from analysis.utils import *
from analysis.graphs import plotNumSolved
from analysis.graphs import plotAllTasks
from dreamcoder.domains.draw.main import visualizePrimitives
from dreamcoder.domains.draw.primitives import program_ink
import matplotlib.pyplot as plt
from analysis.getModelHumanDists import *
import random
   

sys.path.insert(0, "/Users/lucastian/tenen/TENENBAUM/drawgood/experiments")
sys.path.insert(0, "/home/lucast4/drawgood/experiments")
import segmentation as dgseg
import utils as dgutils
import plotsDatFlat as dgpflat
from segmentation import plotMultDrawingPrograms
import plotsSingleSubj as dgpsing
import preprocess as dgprep
import modelAnaly as dgmodel


REMOVELL=False # remove vertical line?

def printAllTasksSolutions(DAT, trainortest="train"):
    print("NOTE: whether is solved is by checking the last iteration")
    
    stringlist = []
        
    if trainortest=="train":
        tasks = DAT["tasks"]
        tasknames = DAT["programnames"]
    elif trainortest=="test":
        tasks = DAT["testtasks"]
        tasknames = DAT["programnames_test"]

    for i, (t, name) in enumerate(zip(tasks, tasknames)):
        print("===== {} - {} - {}:".format(i, t.name, name))
        stringlist.append("===== {} - {} - {}:".format(i, t.name, name))
        
        solved = DAT['taskresultdict'][name]
        # if len(DAT["result"].frontiersOverTime[t][-1])>0:
        #     solved=True
        # else:
        #     solved=False

        if solved:
#             ll = result.frontiersOverTime[t][-1].bestPosterior.logLikelihood
            frontier = getBestFrontierProgram(DAT["result"], t, lastKIter=15, returnfrontier=True)
            summary = f"HIT: ll={frontier.logLikelihood}, post={frontier.logPosterior}, prior={frontier.logPrior}: {frontier.program}"
            # summary = DAT["result"].frontiersOverTime[t][-1].summarize()
        
            if False:
                # print each in fritnier.
                for ii, f in enumerate(DAT["result"].frontiersOverTime[t][-1]):
                    print(dir(f.program))
                    ll = f.program.logLikelihood
                    p = f.program.betaNormalForm()
                    print("{} [{}] : {}".format(ii, ll, p))
        else:
            summary = DAT["result"].frontiersOverTime[t][-1].summarize()
#             print("NOT SOLVED")        
        
        print(summary)
        stringlist.append(summary)
    return stringlist


           

def summarize(ECTRAIN, SUMMARY_SAVEDIR = "", comparetohuman=True):
    # For a given slice of human/model/stim, plot all string distances
    # ECTRAIN = "S9.2"
    NPARSE = 10 # how many random parses to take for plotting for program
    useAggregateDistance = True


    # if False:
    # 1) load data 
    if comparetohuman:
        loadbehavior=True
    else:
        loadbehavior=False
    DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=False, suppressPrint=True, loadbehavior=loadbehavior)

    if len(SUMMARY_SAVEDIR)==0:
        SUMMARY_SAVEDIR = DAT["summarysavedir"]
        # SUMMARY_SAVEDIR = "{}/summary".format(DAT["analysavedir"])
        import os
        os.makedirs(SUMMARY_SAVEDIR, exist_ok=True)


    ################################################################
    # 2) number task solved
    fig = plotNumSolved(DAT["result"])
    fig.savefig("{}/{}_numsolved.pdf".format(SUMMARY_SAVEDIR, DAT["trainset"]))
    print("2: plotted n solved timecourse")

    # 3) print all primitives (and save)
    P = DAT["result"].grammars[-1].primitives
    _, stringlist = visualizePrimitives(P, export="{}/{}_primitives_".format(SUMMARY_SAVEDIR, DAT["trainset"]))
    fname = "{}/{}_primitives.txt".format(SUMMARY_SAVEDIR, DAT["trainset"])
    with open(fname, "w") as f:
        for s in stringlist:
            print(s)
            f.write(s+"\n")
    print("3: save figure of all invented primtiives")

    # 4) Plot all tasks (GROUND TRUTH), AND INDICATE IF SOLVED
    fig = plotAllTasks(DAT, trainortest="train")
    fig.savefig("{}/{}_alltasks_train.pdf".format(SUMMARY_SAVEDIR, DAT["trainset"]))

    if len(DAT["testtasks"])>0:
        fig = plotAllTasks(DAT, trainortest="test")
        fig.savefig("{}/{}_alltasks_test.pdf".format(SUMMARY_SAVEDIR, DAT["trainset"]))

    print("4: saved figure of all tasks (solvbd and unsolved)")
    plt.close('all')

    # 5) Print all task solutions
    stringlist = printAllTasksSolutions(DAT, trainortest="train")
    fname = "{}/{}_solutions_train.txt".format(SUMMARY_SAVEDIR, DAT["trainset"])
    with open(fname, "w") as f:
        for s in stringlist:
            f.write(s+"\n")

    if len(DAT["testtasks"])>0:
        stringlist = printAllTasksSolutions(DAT, trainortest="test")
        fname = "{}/{}_solutions_test.txt".format(SUMMARY_SAVEDIR, DAT["trainset"])
        with open(fname, "w") as f:
            for s in stringlist:
                f.write(s+"\n")

    print("5: saved text file of all task solutions (best posteiro program)")

    print("comapretohuman")
    print(comparetohuman)
    if comparetohuman:
        import pandas as pd
        import seaborn as sns
        print("COMPARING TO HUMAN - SHOULD ALREADY HAVE PREPROCESSED DATA")
        # DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=False, suppressPrint=True)
        DAT = DATloadDrawgoodData(DAT, dosegmentation=True)
        

        # distances = loadDistances(ECTRAIN, use_withplannerscore=False)
        # distances_medianparse = loadDistances(ECTRAIN, ver="medianparse")
        # distances_aggregate = loadDistances(ECTRAIN, ver="aggregate", use_withplannerscore=False)

        if useAggregateDistance:
            distances = loadDistances(ECTRAIN, ver="aggregate", use_withplannerscore=False)
            # distances=distances_aggregate
            label = "allparses_agg"
        else:
            distances = loadDistances(ECTRAIN, use_withplannerscore=False)
            label = "allparses"

        # remove all things like randomperms
        distances = [d for d in distances if d["model"]==ECTRAIN]
        print(len(distances))
        # print(len(distances_aggregate))
        # print(len(distances_medianparse))
        assert len(distances)>0, "huh?"

        # ==== PLOT DISTANCES FOR DIFFERENT SLICES OF MODEL/HUMAN/STIM
        stimlist = DATgetSolvedStim(DAT, intersectDrawgood=True, onlyifhasdatflat=True)
        print("Going one by one thru the stimuli that were both solved by dc and done by humans: {}".format(stimlist))
        for stim in stimlist:

            # DIAGNOSTICS
            print(stim)

            # 2) For model, plot random subset of parses drawings for thsi stim
            dflat = DATloadDatFlat(DAT, stim)
            if len(dflat)>0:
                dflat = dgseg.filterDat(dflat, stimlist=[stim])
                if len(dflat)>NPARSE:
                    dflat = random.sample(dflat, NPARSE)
                assert dflat is not None, "should have data since, only selected tasks that were sovled... whats going on."         
                plotMultDrawingPrograms(dflat, SUMMARY_SAVEDIR, ishuman=False, removeLL=REMOVELL)
                print("2: plotted drawing for {} random parses for model".format(NPARSE))
                plt.close('all')

                modelrends = [d["parsenum"] for d in dflat]
                dists = filterDistances(distances, stimlist=[stim], modelrend=modelrends)
                if len(dists)==0:
                    print("MISSING DATA!!")
                    
                # plt.figure(figsize=(40, 10))

                ax = sns.catplot(x="human", y="dist", hue="modelrend", data=pd.DataFrame(dists), 
                    jitter=0.17, dodge=False, alpha=0.5, height=5, aspect=10/5)
                # ax = sns.stripplot(x="human", y="dist", hue="modelrend", data=pd.DataFrame(dists), jitter=0.17, dodge=False, alpha=0.8, size=8)
                from modelPlanning import addLabel
                addLabel(ax)
                ax.savefig("{}/{}_hu{}_model{}_distances_randomparses.pdf".format(SUMMARY_SAVEDIR, stim, DAT["behaviorexpt"], DAT["trainset"]))
                plt.close('all')

    

            # 1) For human, plot drawings for this stim
            if "png" not in stim:
                dflat_hu = dgseg.filterDat(DAT["datflat_hu"], stimlist=[stim + ".png"])
            else:
                dflat_hu = dgseg.filterDat(DAT["datflat_hu"], stimlist=[stim])
            if len(dflat_hu)==0:
                print("WHY NO DATA")
                print(dflat_hu)
                raise
            plotMultDrawingPrograms(dflat_hu, SUMMARY_SAVEDIR, ishuman=True, removeLL=REMOVELL)
            print("1: plotted drawing steps for Humans")
            plt.close('all')

        # distances = loadDistances(ECTRAIN, use_withplannerscore=False)
        # distances_medianparse = loadDistances(ECTRAIN, ver="medianparse")
        # distances_aggregate = loadDistances(ECTRAIN, ver="aggregate", use_withplannerscore=False)

        # if useAggregateDistance:
        #     distances=distances_aggregate
        #     label = "allparses_agg"
        # else:
        #     label = "allparses"

        # # remove all things like randomperms
        # distances = [d for d in distances if d["model"]==ECTRAIN]
        # print(len(distances))
        # print(len(distances_aggregate))
        # print(len(distances_medianparse))
        # assert len(distances)>0, "huh?"

        # for stim in stimlist:
            # 3) Plot string edit distances between human and model
            # only the plotted parses (x the humans)
            # --- just the parses plotted
            # dflat = DATloadDatFlat(DAT, stim)
            # dflat = dgseg.filterDat(dflat, stimlist=[stim])
            # if len(dflat)>NPARSE:
            #     dflat = random.sample(dflat, NPARSE)
            # modelrends = [d["parsenum"] for d in dflat]
            # dists = filterDistances(distances, stimlist=[stim], modelrend=modelrends)
            # if len(dists)==0:
            #     print("MISSING DATA!!")
                
            # plt.figure(figsize=(40, 10))
            # ax = sns.stripplot(x="human", y="dist", hue="modelrend", data=pd.DataFrame(dists), jitter=0.17, dodge=True, alpha=0.8, size=8)
            # plt.savefig("{}/{}_hu{}_model{}_distances_randomparses.pdf".format(SUMMARY_SAVEDIR, stim, DAT["behaviorexpt"], DAT["trainset"]))
            # plt.close('all')


            # --- all parses for this stimulus
            dists = filterDistances(distances, stimlist=[stim])
            plt.figure(figsize=(40, 10))
            dat = pd.DataFrame(dists)
            ax = sns.violinplot(x="human", y="dist", data=dat, inner="quartile")
            sns.stripplot(ax=ax, x="human", y="dist", data=dat, jitter=0.17, dodge=True, alpha=0.3, size=8)
            ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
            plt.savefig("{}/{}_hu{}_model{}_distances_allparses.pdf".format(SUMMARY_SAVEDIR, stim, DAT["behaviorexpt"], DAT["trainset"]))
            print("3: Plotted all string edit distances for both random parses and all parses")
            plt.close('all')


            # --- aggregate, median over all parses (after aggregating)
            # import pdb
            # pdb.set_trace()
            try:
                if len(distances_medianparse)>0:
                    dists = filterDistances(distances_medianparse, stimlist=[stim])
                    plt.figure(figsize=(40, 10))
                    dat = pd.DataFrame(dists)
                    ax = sns.violinplot(x="human", y="dist", data=dat, inner="quartile")
                    ax = sns.stripplot(x="human", y="dist", data=dat, jitter=0.17, dodge=True, alpha=0.3, size=8)
                    plt.savefig("{}/{}_hu{}_model{}_distances_medianparse.pdf".format(SUMMARY_SAVEDIR, stim, DAT["behaviorexpt"], DAT["trainset"]))
                    print("4: Plotted all string edit distances aggregating over meausres, then taking median over parses")
                    plt.close('all')
            except:
                print("skippint miedian parse.")

            ############# SUMMARIZE FRONTRIERS
            if False:
                # currently skip since this just takes last timepoint, not all
                summarizeFrontiers()




from analysis.utils import *
from dreamcoder.domains.draw.primitives import program_ink
from analysis.importDrawgood import *

# print likelihoods of all frontiers

def summarizeFrontiers(ECTRAIN, k=20):
    """Plots top k frontiers, and save to text their scores and ink used.
    Also plots the ground truth task.
    Currently only does for test tasks but easy to modify to also do all trainign tasks.
    - check the amount of ink used by solutions vs. the ground truth programs"""    # ECTRAIN = "S12.10.test4"

    def F(stim, DAT, k):
        # find the program solution
        result = DAT["result"]
        t = DATgetTask(stim, DAT)[0]
        sdir = DAT["summarysavedir"]
        SDIR = "{}/frontiers".format(sdir)
        import os 
        os.makedirs(SDIR, exist_ok=True)
        print(SDIR)    
        
        # == plot best posterireo
        if False:
            p = result.frontiersOverTime[t][-1].bestPosterior.program.evaluate([])
            fig = dgutils.plotDrawingSteps(p)
            plt.title("ink {}".format(program_ink(p)))
        

        # == plot for all top k frontiers.
        def summarizeTopKFrontiers(k, frontiers, first_plot_best_post=True, skip_print_prog=False):

            frontiers = frontiers.topK(k)

            def print_f(f, skip_print_prog):
                if not skip_print_prog:
                    print(f.program)
                print("posterior: {}".format(f.logPosterior))
                print("likelihood: {}".format(f.logLikelihood))
                print("prior: {}".format(f.logPrior))
                print("ink used: {}".format(program_ink(f.program.evaluate([]))))
                print("---")
                return "post {}, ll {}, prior {}, ink {}".format(f.logPosterior, f.logLikelihood, f.logPrior, program_ink(f.program.evaluate([])))
            
            string_list = []
            if first_plot_best_post:
                st = print_f(frontiers.bestPosterior, skip_print_prog)
                fig = dgutils.plotDrawingSteps(frontiers.bestPosterior.program.evaluate([]))
    #             plt.title(st)
                fig.savefig("{}/{}_bestPost.pdf".format(SDIR, stim))
                string_list.append(st)
                
                
            for i, f in enumerate(frontiers.entries):
                st = print_f(f, skip_print_prog)
                fig = dgutils.plotDrawingSteps(f.program.evaluate([]))
    #             plt.title(st)
                fig.savefig("{}/{}_top_{}.pdf".format(SDIR, stim, i))
                string_list.append(st)
            
            fstring = "{}/{}_description.txt".format(SDIR, stim)
            with open(fstring, 'w') as f:
                for s in string_list:
                    f.write("{}\n".format(s))

        frontiers = result.frontiersOverTime[t][-1] # last timepoijnt
    #     print(frontiers)
        summarizeTopKFrontiers(k, frontiers, first_plot_best_post=False, skip_print_prog=True)


        # print all the likelihoods in sorted order
        print(sorted([f.logLikelihood for f in frontiers.entries]))
        
        # plot drawing steps for both ground truth and extracted program
        print(program_ink(t.strokes))
        fig = dgutils.plotDrawingSteps(t.strokes)
        plt.title("ink {}".format(program_ink(t.strokes)))
        fig.savefig("{}/{}_truth.pdf".format(SDIR, stim))
        
        plt.close(fig)
        
    DAT = loadCheckpoint(ECTRAIN, loadparse=False)
    for stim in [t.name for t in DAT["testtasks"]]:
    #     stim = "S12_13_test_4"
        F(stim, DAT, k)



if __name__=="__main__":
    # e..g, python analysis/summarize.py "S12.1" 0
    ECTRAIN = sys.argv[1]
    if len(sys.argv)>2:
        if int(sys.argv[2])==1:
            comparetohuman=True
        elif int(sys.argv[2])==0:
            comparetohuman=False
        else:
            raise
    else:
        comparetohuman=True

    summarize(ECTRAIN, SUMMARY_SAVEDIR = "", comparetohuman=comparetohuman)


