## summarize, especialyl comparing model to human.
import sys

sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")

from analysis.getModelHumanDists import loadDistances, filterDistances
from analysis.utils import *
from analysis.graphs import plotNumSolved
from analysis.graphs import plotAllTasks
from dreamcoder.domains.draw.main import visualizePrimitives
import pandas as pd
import seaborn as sns
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


REMOVELL=True # remove vertical line?

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
            
        if len(DAT["result"].frontiersOverTime[t][-1])>0:
            solved=True
        else:
            solved=False

        if solved:
#             ll = result.frontiersOverTime[t][-1].bestPosterior.logLikelihood
            summary = DAT["result"].frontiersOverTime[t][-1].summarize()
        
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
    NPARSE = 10 # how many random parses to take for plotting for programs.
    
    # 1) load data 
    DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=False, suppressPrint=True)

    if len(SUMMARY_SAVEDIR)==0:
        SUMMARY_SAVEDIR = "{}/summary".format(DAT["analysavedir"])
        import os
        os.makedirs(SUMMARY_SAVEDIR, exist_ok=True)

    # 2) number task solved
    fig = plotNumSolved(DAT["result"])
    fig.savefig("{}/{}_numsolved.png".format(SUMMARY_SAVEDIR, DAT["trainset"]))
    print("2: plotted n solved timecourse")

    # 3) print all primitives
    P = DAT["result"].grammars[-1].primitives
    visualizePrimitives(P, export="{}/{}_primitives_".format(SUMMARY_SAVEDIR, DAT["trainset"]))
    print("3: save figure of all invented primtiives")

    # 4) Plot all tasks (GROUND TRUTH), AND INDICATE IF SOLVED
    fig = plotAllTasks(DAT, trainortest="train")
    fig.savefig("{}/{}_alltasks_train.png".format(SUMMARY_SAVEDIR, DAT["trainset"]))

    fig = plotAllTasks(DAT, trainortest="test")
    fig.savefig("{}/{}_alltasks_test.png".format(SUMMARY_SAVEDIR, DAT["trainset"]))
    print("4: saved figure of all tasks (solvbd and unsolved)")
    plt.close('all')

    # 5) Print all task solutions
    stringlist = printAllTasksSolutions(DAT, trainortest="train")
    fname = "{}/{}_solutions_train.txt".format(SUMMARY_SAVEDIR, DAT["trainset"])
    with open(fname, "w") as f:
        for s in stringlist:
            f.write(s+"\n")

    stringlist = printAllTasksSolutions(DAT, trainortest="test")
    fname = "{}/{}_solutions_test.txt".format(SUMMARY_SAVEDIR, DAT["trainset"])
    with open(fname, "w") as f:
        for s in stringlist:
            f.write(s+"\n")
    print("5: saved text file of all task solutions (best posteiro program)")


    if comparetohuman:
        print("COMPARING TO HUMAN - SHOULD ALREADY HAVE PREPROCESSED DATA")
        DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True, suppressPrint=True)
        DAT = DATloadDrawgoodData(DAT, dosegmentation=True)
        distances = loadDistances(ECTRAIN)

        # ==== PLOT DISTANCES FOR DIFFERENT SLICES OF MODEL/HUMAN/STIM
        stimlist = DATgetSolvedStim(DAT, intersectDrawgood=True)
        print("Going one by one thru the stimuli that were both solved by dc and done by humans: {}".format(stimlist))
        for stim in stimlist:

            # DIAGNOSTICS
            print(stim)

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

            # 2) For model, plot random subset of parses drawings for thsi stim
            dflat = DATloadDatFlat(DAT, stim)
            dflat = dgseg.filterDat(dflat, stimlist=[stim])
            if len(dflat)>NPARSE:
                dflat = random.sample(dflat, NPARSE)
            assert dflat is not None, "should have data since, only selected tasks that were sovled... whats going on."         
            plotMultDrawingPrograms(dflat, SUMMARY_SAVEDIR, ishuman=False, removeLL=REMOVELL)
            print("2: plotted drawing for {} random parses for model".format(NPARSE))
            plt.close('all')
    
            # 3) Plot string edit distances between human and model
            # only the plotted parses (x the humans)
            # --- just the parses plotted
            modelrends = [d["parsenum"] for d in dflat]
            dists = filterDistances(distances, stimlist=[stim], modelrend=modelrends)
            if len(dists)==0:
                print("MISSING DATA!!")
                
            plt.figure(figsize=(40, 10))
            ax = sns.stripplot(x="human", y="dist", hue="modelrend", data=pd.DataFrame(dists), jitter=0.17, dodge=True, alpha=0.8, size=8)
            plt.savefig("{}/{}_hu{}_model{}_distances_randomparses.png".format(SUMMARY_SAVEDIR, stim, DAT["behaviorexpt"], DAT["trainset"]))
            plt.close('all')

            # --- all parses for this stimulus
            dists = filterDistances(distances, stimlist=[stim])
            plt.figure(figsize=(40, 10))
            dat = pd.DataFrame(dists)
            ax = sns.violinplot(x="human", y="dist", data=dat, inner="quartile")
            ax = sns.stripplot(x="human", y="dist", data=dat, jitter=0.17, dodge=True, alpha=0.3, size=8)
            plt.savefig("{}/{}_hu{}_model{}_distances_allparses.png".format(SUMMARY_SAVEDIR, stim, DAT["behaviorexpt"], DAT["trainset"]))
            print("3: Plotted all string edit distances for both random parses and all parses")
            plt.close('all')

if __name__=="__main__":
    ECTRAIN = sys.argv[1]
    summarize(ECTRAIN, SUMMARY_SAVEDIR = "", comparetohuman=True)


