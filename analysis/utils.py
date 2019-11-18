
from bin.graphs import loadfun
import matplotlib.pyplot as plt
from math import ceil
# === 4) Load tools to work with tasks libraries
# import dreamcoder.domains.draw.primitives as P
import os
import glob
import sys
import pickle
sys.path.append("../")

# from analysis.getModelHumanDists import DATloadDrawgoodData
sys.path.insert(0, "/Users/lucastian/tenen/TENENBAUM/drawgood/experiments")
sys.path.insert(0, "/home/lucast4/drawgood/experiments")
# from modelAnaly import distModelHumanAllStims

REMOVELL = False # remove vertical long line?


## ====== collect dreamcoder results and parse results
def loadCheckpoint(trainset="S9_nojitter", userealnames=True, loadparse=False, suppressPrint=False, loadbehavior=True):
    # userealnames = True, then names tasks by their stim names (eg., S8_2), otherwise names as train0, train1, ...
    ##### METADATA
    behaviorexpt = ""
    if trainset=="S8full":
        userealnames=False
        doshaping = True
        # == 1) Load pickle checkpoint
        jobname = "S8full_2019-10-10_09-45-18"
        exptdir = "2019-10-10T09:45:23.421494"
        checkpoint = "draw_aic=1.0_arity=3_BO=True_CO=True_doshaping=True_ES=1_ET=720_HR=0.5_it=6_MF=5_noConsolidation=False_pc=10_RT=1800_RR=False_RW=False_solver=ocaml_STM=True_L=1_TRR=default_K=2_topkNotMAP=False_trainset=S8full_graph=True.pickle"
        taskset = "S8full"
    elif trainset == "S9full":
        userealnames=False
        doshaping = True
        # == 1) Load pickle checkpoint
        jobname = "S9full_2019-10-10_09-45-29"
        exptdir = "2019-10-10T09:45:47.447037"
        checkpoint = "draw_aic=1.0_arity=3_BO=True_CO=True_doshaping=True_ES=1_ET=720_HR=0.5_it=6_MF=5_noConsolidation=False_pc=10_RT=1800_RR=False_RW=False_solver=ocaml_STM=True_L=1_TRR=default_K=2_topkNotMAP=False_trainset=S9full_graph=True.pickle"
        taskset = "S9full"
    elif trainset== "S8fixedprim":
        userealnames=False
        doshaping = True
        jobname = "S8fixedprim_2019-10-10_18-23-41"
        exptdir = "2019-10-10T18:24:59.632316"
        checkpoint = "draw_aic=1.0_arity=3_BO=True_CO=True_doshaping=True_ES=1_ET=1800_HR=0.5_it=6_MF=5_noConsolidation=False_pc=10_RT=2000_RR=False_RW=False_solver=ocaml_STM=True_L=1_TRR=default_K=2_topkNotMAP=False_trainset=S8full_graph=True.pickle"
        taskset = "S8full"
    elif trainset=="S9fixedprim":
        userealnames=False
        doshaping = True
        jobname = "S9fixedprim_2019-10-10_18-23-19"
        exptdir = "2019-10-10T18:24:13.326352"
        checkpoint = "draw_aic=1.0_arity=3_BO=True_CO=True_doshaping=True_ES=1_ET=1800_HR=0.5_it=5_MF=5_noConsolidation=False_pc=10_RT=2000_RR=False_RW=False_solver=ocaml_STM=True_L=1_TRR=default_K=2_topkNotMAP=False_trainset=S9full_graph=True.pickle"
        taskset = "S9full"
        
    elif trainset=="S8_nojitter":
        userealnames=False
        doshaping = True
        jobname = "S8_nojitter_2019-10-13_23-25-52"
        exptdir = "2019-10-13T23:26:02.364547"
        taskset = "S8_nojitter"
    elif trainset=="S9_nojitter":
        userealnames=False
        doshaping = True
        jobname = "S9_nojitter_2019-10-13_23-25-52"
        exptdir = "2019-10-14T08:33:57.806621"
        taskset = "S9_nojitter"

    elif trainset=="S8.2.2":
        userealnames=False
        doshaping = True
        jobname = "S8.2.2_2019-10-20_14-08-46"
        exptdir = "2019-10-20T14:09:07.358291"
        taskset = "S8_nojitter"
        behaviorexpt = "2.3"
    elif trainset=="S9.2":
        userealnames=False
        doshaping = True
        jobname = "S9.2_2019-10-17_20-00-40"
        exptdir = "2019-10-17T20:01:17.912264"
        taskset = "S9_nojitter"
        behaviorexpt = "2.3"

    elif trainset=="S12.1":
        userealnames=True
        doshaping = False
        # jobname = "S12.1_2019-11-01_10-49-52" # not actually used.
        exptdir = "2019-11-01T10:51:04.566148"
        taskset = "S12" 
        behaviorexpt = ""

    elif trainset=="S12.6.1":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-03T02:44:46.586584"
        taskset = "S12" 
        behaviorexpt = ""

    elif trainset=="S12.6.2":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-03T02:44:47.777239"
        taskset = "S12" 
        behaviorexpt = ""

    elif trainset=="S13.9":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-04T11:55:48.745857"
        taskset = "S13" 
        behaviorexpt = ""

    elif trainset=="S12.8.1":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-04T15:49:42.313470"
        taskset = "S12" 
        behaviorexpt = ""

    elif trainset=="S12.10":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-05T10:59:39.638549"
        taskset = "S12" 
        behaviorexpt = "2.4"

    elif trainset=="S13.10":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-05T10:59:39.800347"
        taskset = "S13" 
        behaviorexpt = "2.4"

    elif trainset=="S12.10.test4":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-10T22:47:09.031267"
        taskset = "S12" 
        behaviorexpt = "2.4"
    elif trainset=="S13.10.test4":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-10T22:47:09.086994"
        taskset = "S13" 
        behaviorexpt = "2.4"

    elif trainset=="S12.10.test5":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-16T19:03:45.745355"
        taskset = "S12" 
        behaviorexpt = "2.4"
    elif trainset=="S13.10.test5":
        userealnames=True
        doshaping = False
        exptdir = "2019-11-16T19:03:45.006020"
        taskset = "S13" 
        behaviorexpt = "2.4"
    
    else:
        print("PROBLEM did not find traiin set! ")
        assert False

        
    ###### FIRST, load the correct checkpoint
    print("Loading dreamcoder checkpoint")
    F = glob.glob("experimentOutputs/draw/{}/draw*.pickle".format(exptdir))
    iters = []
    for f in F:
        g = f.find("_graph")
        if not f[g+7:g+8]=="T":
            iters.append(0)
        else:
            a = f.find("_it")
            b = f.find("_MF")
            iters.append(int(f[a+4:b]))
        if not suppressPrint:
            print(f)
            print(iters[-1])
            print('----')

    # --- find the file with highest iteration
    # print(max(iters))
    ind = [i for i, j in enumerate(iters) if j==max(iters)]
    assert len(ind)==1
    checkpoint = F[ind[0]]
    checkpoint = checkpoint.split("/")[-1]
    f = "experimentOutputs/draw/{}/{}".format(exptdir, checkpoint)
    result = loadfun(f)

    ####### LOADING TASKS 
    def loadTasks(taskset, doshaping):
        print("Loading dreamcoder tasks")
        # == 2) Load tasks
        from dreamcoder.domains.draw.makeDrawTasks import makeSupervisedTasks
        tasks, testtasks, programnames, program_test_names = makeSupervisedTasks(trainset=taskset, doshaping=doshaping, userealnames=userealnames)
        return tasks, testtasks, programnames, program_test_names

    tasks, testtasks, programnames, program_test_names = loadTasks(taskset, doshaping)

    ######  LOAD PARSES IF EXIST
    def loadParses():
        import os
        F = glob.glob("experimentOutputs/draw/{}/parsesflat_*.pickle".format(exptdir))
        parses = []
        for f in F:
            if not suppressPrint:
                print("loading parse {}".format(f))
            if os.path.getsize(f)==0:
                print("skipping parse - size 0")
                continue

            name = f[f.find("parses")+11:f.find(".pickle")]

            with open(f, "rb") as ff:
                parse = pickle.load(ff)
            parses.append({
                "name":name,
                "parse":parse
                })  
        return parses        

    if loadparse:
        print("Loading parses")
        parses = loadParses()
        if len(parses)>0:
            print("FOUND {} pre-computed parses!".format(len(parses)))
    else:
        print("not loading parses")
        parses = []


    print("Num dreamcoder tasks {}".format(len(result.taskSolutions)))
    print("n supervised tasks {}".format(len(tasks)))
    assert len(result.taskSolutions)==len(tasks)
    print("Num dreamcoder TEST tasks {}".format(len(result.getTestingTasks())))
    print("n supervised TEST tasks {}".format(len(testtasks)))
    if len(result.getTestingTasks())==0:
        # then assume no testing, remove all test tasks
        testtasks=[]
    else:
        # make sure numbers 
        assert len(result.getTestingTasks())==len(testtasks)

    savedir = "experimentOutputs/draw/{}".format(exptdir)
    analysavedir = "analysis/saved/DAT_ec{}_dg{}".format(trainset, behaviorexpt)
    summarysavedir = "analysis/summary/DAT_ec{}_dg{}".format(trainset, behaviorexpt)
    summaryfigsavedir = "analysis/summaryfigs/DAT_ec{}_dg{}".format(trainset, behaviorexpt)

    import os
    os.makedirs(summaryfigsavedir, exist_ok=True)
    
    # ==== output a dict
    DAT = {
    "trainset":trainset,
    "result": result,
    "tasks": tasks,
    "testtasks":testtasks,
    "programnames": programnames,
    "programnames_test": program_test_names,
    "behaviorexpt": behaviorexpt,
    "savedir": savedir,
    "parses": parses,
    "analysavedir":analysavedir,
    "summarysavedir":summarysavedir,
    "loadparse":loadparse,
    }

    DAT["taskresultdict"] = getTaskResults(DAT)

    if loadbehavior:
        print("Loading behavior")
        loadBehavior(DAT)
        # print("Loaded behavior also")


    # === update savedirs
    DATupdateSaveDirs(DAT)

    return DAT


########### HELPER FUNCTIONS FOR DAT
from dreamcoder.domains.draw.primitives import program_ink
def _getAndRankAllFrontiers(results, task, SDIR=[], usell=True, K=10):
    """gets the K best programs (for each iteration) and combines across all iterations
    sorts by likelihood (decreasing)"""
    # t = DATgetTask(stim, DAT)[0]
    # results = DAT["result"]

    # 1) Collect all frontiers (top K each iteration) over all iterations.
    frontiers_over_time = []
    # print(type(results))
    # print(results.keys())
    # print(results.frontiersOverTime[task])
    for i, frontiers_thisiter in enumerate(results.frontiersOverTime[task]):
        print(frontiers_thisiter)
        if frontiers_thisiter.empty:
            print("emopt")
            continue

        frontierprogs = frontiers_thisiter.topK(K)
        for f in frontierprogs.entries:
            frontiers_over_time.append({
                "iteration":i,
                "frontier":f
            })
    
    # for each frontier get scores and ink
    for f in frontiers_over_time:
        # collect scores and ink used
        f["prior"] = f["frontier"].logPrior
        f["post"] = f["frontier"].logPosterior
        f["ll"] = f["frontier"].logLikelihood
        f["ink"] = program_ink(f["frontier"].program.evaluate([]))
        
    # sort by ll, then ink, then prior
    frontiers_over_time = sorted(frontiers_over_time, key=lambda x: (-x["ll"], x["ink"], -x["prior"]))

    
    # save as text
    def save(frontiers_over_time, path):
        import copy
        import json
        
        # first make copy that is stringified
        frontiers_over_time_string=copy.deepcopy(frontiers_over_time)
        for f in frontiers_over_time_string:
            f["frontier"]=str(f["frontier"])
        # save
        with open(path, "w") as f:
            json.dump(frontiers_over_time_string, f, indent=4)  
    
    if isinstance(SDIR, list):
        assert len(SDIR)==0, "should be empty"
        print("(getAndRankAllFrontiers) skipping saving, did not tell me a sdir")
    else:
        save(frontiers_over_time, "{}/{}.txt".format(SDIR, stim))

    return frontiers_over_time

def getAndRankAllFrontiers(DAT):
    """just wrapper that calls for each stim in test and training"""
    SDIR = "{}/frontiers_across_iter".format(DAT["summarysavedir"])
    import os
    os.makedirs(SDIR, exist_ok=True)
    stimlist = DAT["taskresultdict"]
    results = DAT["result"]

    for stim in stimlist:
        task = DATgetTask(stim, DAT)[0]
        frontiers_over_time = _getAndRankAllFrontiers(results, task, SDIR)


def getTaskResults(DAT):
    from analysis.parse import getLatestFrontierProgram
    """summary dict of each task and whether was solved (at any interation)"""
    # first get all stime
    stimnames = [t.name for t in DAT["tasks"]]
    stimnames.extend([t.name for t in DAT["testtasks"]])

    # solved?
    def solved(stim):
        solution = getLatestFrontierProgram(DAT["result"], DATgetTask(stim, DAT)[0])
        if isinstance(solution, list):
            assert len(solution)==0
            return False
        else:
            return True

    taskresultdict = {}
    for s in stimnames:
        # print("{} - {}".format(s, solved(s)))
        taskresultdict[s]=solved(s)
    return taskresultdict


def DATgetWorkerList(DAT):
    from analysis.importDrawgood import dgutils
    assert "datall_human" in DAT.keys(), "need to first load behavior."
    humanlist = dgutils.getWorkers(DAT["datall_human"])
    return humanlist
    

def DATupdateSaveDirs(DAT):
    import os
    """dirs for saving processed stimuli"""
    
    # -- datsegs:
    sdir = "{}/datsegs_ec".format(DAT["analysavedir"])
    os.makedirs(sdir, exist_ok=True)
    DAT["savedir_datsegs"] = sdir

    # -- datflat
    sdir = "{}/datflat_ec".format(DAT["analysavedir"])
    os.makedirs(sdir, exist_ok=True)
    DAT["datflatsavedir"] = sdir

    # -- model - human distance
    sdir = "{}/modelhudist".format(DAT["analysavedir"])
    os.makedirs(sdir, exist_ok=True)
    DAT["savedir_modelhudist"] = sdir


def DATloadDatFlat(DAT, stimname):
    # helper function to load datflat
    fname = "{}/{}.pickle".format(DAT["datflatsavedir"], stimname)
    from os import path
    if path.exists(fname):
        with open(fname, "rb") as f:
            datflat_ec = pickle.load(f)
        return datflat_ec
    else:
        print("CANT FIND FILE: {}".format(fname))
        return []

def DATsaveDatFlat(DAT, datflat, stimname):
    fname = "{}/{}.pickle".format(DAT["datflatsavedir"], stimname)
    with open(fname, "wb") as f:
        pickle.dump(datflat, f)
    

def DATloadDatSeg(DAT, stimname):
    # helper function to load datflat
    fname = "{}/{}.pickle".format(DAT["savedir_datsegs"], stimname)
    with open(fname, "rb") as f:
        datseg = pickle.load(f)
    return datseg

def DATsaveDatSeg(DAT, datseg, stimname):
    fname = "{}/{}.pickle".format(DAT["savedir_datsegs"], stimname)
    with open(fname, "wb") as f:
        pickle.dump(datseg, f)

def DATsaveDatSeg(DAT, datseg, stimname):
    fname = "{}/{}.pickle".format(DAT["savedir_datsegs"], stimname)
    with open(fname, "wb") as f:
        pickle.dump(datseg, f)


def loadBehavior(DAT):
    try:
        fname = "../TENENBAUM/drawgood/experiments/data/datall_{}.pickle".format(DAT["behaviorexpt"])
        with open(fname, "rb") as f:
            datall_drawgood = pickle.load(f)   
        DAT["datall_human"] = datall_drawgood 
    except:
        fname = "/home/lucast4/drawgood/experiments/data/datall_{}.pickle".format(DAT["behaviorexpt"])
        with open(fname, "rb") as f:
            datall_drawgood = pickle.load(f)   
        DAT["datall_human"] = datall_drawgood 

def DATloadBehavior(DAT):
    loadBehavior(DAT)


def getTask(stimname, DAT):
    # search thru train, then through test tasks, outputs the task object and whether is train or test
    for task, name in zip(DAT["tasks"], DAT["programnames"]):
        if name==stimname:
            return task, "train"
    for task, name in zip(DAT["testtasks"], DAT["programnames_test"]):
        if name==stimname:
            return task, "test"


def DATsaveModelHuDist(DAT, stim, human, dists, suffix=''):
    sdir = DAT["savedir_modelhudist"]
    fname = "{}/{}_{}_{}.pickle".format(sdir, stim, human, suffix)
    with open(fname, "wb") as f:
        pickle.dump(dists, f)

def DATloadModelHuDist(DAT, stim, human, suffix='', use_withplannerscore=False):
    if use_withplannerscore:
        # these are the same, but appending planner RL model scores.
        sdir = DAT["savedir_modelhudist"] + "_withplannerscore"
    else:
        sdir = DAT["savedir_modelhudist"]
    if len(suffix)>0:
        suffix="_"+suffix
    fname = "{}/{}_{}{}.pickle".format(sdir, stim, human, suffix)
    try:
        with open(fname, "rb") as f:
            dists = pickle.load(f)
    except:
        print("skipped loading: can't find {}".format(fname))
        dists = []
    return dists


def DATgetTask(stimname, DAT):
    return(getTask(stimname, DAT))


################# DRAWGOOD HELPER -  things that help with applying drawgood
def DATloadDrawgoodData(DAT, dosegmentation=True):
    from segmentation import getSegmentation
    from preprocess import getFlatData
    print("Loading human datflat data")
    DAT["datflat_hu"] = getFlatData(DAT["datall_human"])
    if dosegmentation:
        print("Doing segmentation")
        DAT["datseg_hu"] = getSegmentation(DAT["datflat_hu"], unique_codes=True, dosplits=True, removeLongVertLine=REMOVELL)                                      
    return DAT



def DATgetSolvedStim(DAT, removeshaping=True, intersectDrawgood=False, 
    onlyifhasdatflat=False):
    import re
    """gets stims that have parses"""

    # gets solved stim.
    stimnames = []
    for tname, solved in DAT["taskresultdict"].items():
        if solved:
            stimnames.append(tname)

    # assert DAT["loadparse"], print("no parses found - you have to load parses first")
    # stimnames = [P["name"] for P in DAT["parses"] if len(P["parse"])>0]

    if removeshaping:
        # then throw out shaping that did not get for humans:
        print("REMOVING SHAPING STIMULI (THOSE HUMANS WERE NOT GIVEN) [wil even remove things like S9_shaping_5]")

        def isShaping(name):
            # ouptuts true if is shaping
            # looks for something like S[], where [] is any numner. if can't fin then this is shaping.
            if False:
                # old version, not good since it thinks that things like S9_shaping_5 are not shaping
                if re.search("S[0-9]+_", name): # like S99_, must followed by underscore
                    # then is a real task
                    return False
                else:
                    print("removing, since I think is shaping: {}".format(name))
                    return True 
            else:
                if re.search("shaping", name): 
                    print("removing, since I think is shaping: {}".format(name))
                    # then is shaping
                    return True
                else:
                    return False 

        stimnames = [s for s in stimnames if not isShaping(s)]
    if intersectDrawgood:
        # then only keep if it is part of stim present in the drawgood stimuli
        DAT = DATloadDrawgoodData(DAT, dosegmentation=False)
        stimDG = set([d["stimname"][:d["stimname"].find(".png")] for d in DAT["datflat_hu"]])
        print(stimDG)
        # print(stimnames[0])
        stimnames = [s for s in stimnames if s in stimDG]
    if onlyifhasdatflat:
        # -- check if they have datflat
        from os import path
        def check(stim):
            fname = "{}/{}.pickle".format(DAT["datflatsavedir"], stim)
            return path.exists(fname)            
        stimnames = [stim for stim in stimnames if check(stim)]
        
    return stimnames


