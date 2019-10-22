from bin.graphs import *
import matplotlib.pyplot as plt
from math import ceil
# === 4) Load tools to work with tasks libraries
import dreamcoder.domains.draw.primitives as P
import os
import glob
import sys
sys.path.append("../")


## ====== collect dreamcoder results and parse results
def loadCheckpoint(trainset="S9_nojitter", userealnames=True):
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
    else:
        print("PROBLEM did not find traiin set! ")
        
        
    ###### FIRST, load the correct checkpoint
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
        # == 2) Load tasks
        from dreamcoder.domains.draw.makeDrawTasks import makeSupervisedTasks
        tasks, testtasks, programnames, program_test_names = makeSupervisedTasks(trainset=taskset, doshaping=doshaping, userealnames=userealnames)
        return tasks, testtasks, programnames, program_test_names


    tasks, testtasks, programnames, program_test_names = loadTasks(taskset, doshaping)

    print("Num dreamcoder tasks {}".format(len(result.taskSolutions)))
    print("n supervised tasks {}".format(len(tasks)))
    assert len(result.taskSolutions)==len(tasks)
    print("Num dreamcoder TEST tasks {}".format(len(result.getTestingTasks())))
    print("n supervised TEST tasks {}".format(len(testtasks)))
    assert len(result.getTestingTasks())==len(testtasks)

    savedir = "experimentOutputs/draw/{}".format(exptdir)
    return result, tasks, testtasks, programnames, program_test_names, behaviorexpt, savedir 


def getTask(stimname, tasks, programnames, testtasks, program_test_names):
    # search thru train, then through test tasks, outputs the task object and whether is train or test
    for task, name in zip(tasks, programnames):
        if name==stimname:
            return task, "train"
    for task, name in zip(testtasks, program_test_names):
        if name==stimname:
            return task, "test"
