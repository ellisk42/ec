## here taking dreamcoder outputs and parsing into sequences. 
## will integrate with behaviral data to do modeln

import sys

sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")
# print(sys.path)
from dreamcoder.domains.draw.drawPrimitives import Program
from dreamcoder.domains.draw.drawPrimitives import Parse
# from dreamcoder.domains.draw.primitives import _repeat, _line, _makeAffine, _circle,_connect
from dreamcoder.domains.draw.makeDrawTasks import makeSupervisedTasks, SupervisedDraw
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
import numpy as np
from analysis.utils import *


def getParses(dreamcoder_program):
    """ e.g., dreamcoder_program = result.allFrontiers[tasks[i]].bestPosterior.program"""
    # extracts all parses for a given program object
    print("getting parse...")
    # if len(str(dreamcoder_program))>150:
    # import pdb
    # pdb.set_trace()
    parses = Parse.ofProgram(Program.parse(str(dreamcoder_program)))
    return parses


def getBestFrontierProgram(result, task, lastKIter=4):
    """ gets best solution for a given task, restricting to 
    solutions from lastKIter before last iter, to last iter.
    ranks based on ll, then ink, then prior. 
    lastKIter counts from the largest iter for which there exists a solution,
    e.g., if you think that last 4 iterationsm might be bad, then
    make lastKIter>4"""
    from analysis.utils import _getAndRankAllFrontiers
    
    # stim = task.name
    # print("ASDas")
    # print(result.keys())
    frontiers_over_time = _getAndRankAllFrontiers(result, task)

    if len(frontiers_over_time)==0:
        print("did not find any programs across all frontiers and all iterations")
        return []
    else:
        last_iter = max([f["iteration"] for f in frontiers_over_time])
        frontiers_over_time = [f for f in frontiers_over_time if f["iteration"]>(last_iter-lastKIter)]
        frontier_to_take = frontiers_over_time[0]
        return frontier_to_take["frontier"].program


def getLatestFrontierProgram(result, task):
    """gives the most recent (latest iteration) 
    solutions (generally is the last one)
    Outputs the bestposterior program
    Works for both test or train tasks"""
    frontiers = result.frontiersOverTime[task]
    for f in reversed(frontiers):
        if not f.empty:
            return f.bestPosterior.program # this is the most recent frontier not empty
    return []

def getAndSaveParses(experiment="S9.2", debug=False, skipthingsthatcrash=False, useFindBestProgram=True, nrand_flat_parses=10000):
    """ gets the most recent, and the bestPosterior program
    gets all parses
    useFindBestProgram, means find best likelihood, lowest ink,
    over last few iterations. 
    nrand_flat_parses, will subsample randomly from parses to make the flat parses that saved. put [] to save all."""
    # DAT = loadCheckpoint(trainset=experiment)
    # for key in DAT.keys():
    #     key =

    # result, tasks, testtasks, programnames, program_test_names, behaviorexpt, savedir = loadCheckpoint(trainset=experiment)[:7]

    DAT = loadCheckpoint(trainset=experiment, loadparse=False, suppressPrint=True, loadbehavior=False)
    
    result=DAT["result"]
    tasks=DAT["tasks"]
    testtasks=DAT["testtasks"]
    programnames=DAT["programnames"]
    program_test_names=DAT["programnames_test"]
    behaviorexpt=DAT["behaviorexpt"]
    savedir=DAT["savedir"]

    def saveparses(T, P, skipthingsthatcrash=False):
        # === for each program, get the best posteiror and then all parses of that. 
        for t, name in zip(T, P):
            if name in ["shaping_1", "shaping_4", "shaping_8"]:
                continue
            print("Parsing {} ...".format(name))

            if debug:
                # for some reason crashes... was trying to figure out why.
                if name!="S12_235":
                    print(name)
                    continue
            print(skipthingsthatcrash)
            print(name)
            if skipthingsthatcrash:
                if name in ["S12_235", "S12_243", "S12_13_test_1"]:
                    print("SKIPPING - {} - since CRASHES...".format(name))
                    continue

            # Get bestPosterior solution (most recent, and best posterior)
            if useFindBestProgram:
                # then goes thru multipel itetatyions to find best
                p = getBestFrontierProgram(result, t)
            else:
                # then uses last iterations best program
                p = getLatestFrontierProgram(result, t)

            if isinstance(p, list):
                assert len(p)==0, "i thought a list means did not find a solution..."
                # then no solution
                parses = []
            else:
                # then get parses
                parses = getParses(p)
            # if result.frontiersOverTime[t][-1].empty:
            #     parses =[]
            # else:
            #     p = result.frontiersOverTime[t][-1].bestPosterior.program
            #     parses = getParses(p)
            #     # parses = [1,2,3]
            
            # 1) save parse object
            fname = "{}/parses_{}.pickle".format(savedir, name)
            with open(fname, "wb") as f:
                pickle.dump(parses, f, pickle.HIGHEST_PROTOCOL)

            # 2) save flattened parses
            fname = "{}/parsesflat_{}.pickle".format(savedir, name)
            flatparses = [p.flatten() for p in parses]
            if isinstance(nrand_flat_parses, int):
                import random
                if len(flatparses)>nrand_flat_parses:
                    flatparses = random.sample(flatparses, nrand_flat_parses)

            with open(fname, "wb") as f:
                pickle.dump(flatparses, f, pickle.HIGHEST_PROTOCOL)

            print("saved to :{}".format(fname))

    # ========= DO TRAIING TASKS
    saveparses(T=tasks, P=programnames, skipthingsthatcrash=skipthingsthatcrash)

    # ========= DO TEST TASKS
    if len(testtasks)>0:
        saveparses(T=testtasks, P=program_test_names, skipthingsthatcrash=skipthingsthatcrash)

    # === for each task and testtask
    return result, tasks, testtasks, programnames, program_test_names, behaviorexpt


def program2strokes(program):
    """given program convert into "strokes", so that can pass into same behavioral aalysis as for subjects"""
    # program is list of numpy arrays (flattened)
    # for nowwill put down times in order. will be in fake milliseconds. 
    
    on = 1
    off = 300
    strokes = []
    for p in program:
        times = np.linspace(on, off, p.shape[0])
        p = np.concatenate((p, times[:,None]), axis=1)
        on+=500
        off+=500
        strokes.append(p)
    return strokes
    
    # strokes = program2strokes(dreams[1].evaluate([]))
    # strokes = program2strokes(result.allFrontiers[tasks[1]].bestPosterior.program.evaluate([]))
    # strokes = [s.tolist() for s in strokes]


"""Given a list of flattened program (i.e, list of numpy) converts to datflat"""
def parses2datflat(programs, stimname="", condition="", randomsubsample=[]):
    # ideally stimname is the name of stim for all parses (i.e progs)
    # ideally condition indicates what the training schedule was.

    # === A HACK, to make this work with code written fro beahviroal analysis, 
    # convert this list of programs --> list of strokes --> one datflat object
    # this datflat can then be treated just like a human subject's data.

    datflat = []
    from pythonlib.tools.listtools import permuteRand
    print("getting datflat for {}".format(stimname))

    if isinstance(randomsubsample, int)>0:
        # takes a random subsample of all programs, without repalcement
        if len(programs)>randomsubsample:
            import random
            print("(parses --> datflat) doing random subsample of {}{} from {} to {}".format(stimname, condition, len(programs), randomsubsample))
            programs = random.sample(programs, randomsubsample)
    else: assert isinstance(randomsubsample, list), "expect either int or empty list..."


    for i, prog in enumerate(programs):
        # print("prog {}".format(i))
        if isinstance(prog, Parse):
            prog = prog.flatten()
        # append fake timestamps, so that prog become a strokes list:
        strokes = program2strokes(prog)

        # create a new entry in datflat, 
        datflat.append({
            "parsenum": i, # this is unique to dreamcoder; each program has multiple parses
            "trialstrokes":strokes,
            "trialonset": 0,
            "stimname": stimname, 
            "trialprimitives":[],
            "trialcircleparams":[],
            "condition":condition
        })
    return datflat


def parses2datflatAll(DAT, randomsubsample=[]):
    # === this gets all into one datflat (i.e., stim x parses)
    datflat_ec = []
    for P in DAT["parses"]:
        # print(P)
        stimname = P["name"]
        parses = P["parse"]
        datflat_ec.extend(parses2datflat(parses, stimname=stimname, condition=DAT["trainset"], randomsubsample=randomsubsample))
    return datflat_ec


def parses2datflatAllSave(DAT, randomsubsample=[]):
    print("NOTE: will skip if find that has already beeen done")
    if "datflatsavedir" not in DAT.keys():
        savedir = "{}/datflat_ec".format(DAT["analysavedir"])
        DAT["datflatsavedir"] = savedir
    savedir = DAT["datflatsavedir"]
    os.makedirs(savedir, exist_ok=True)
    print(savedir)
    for P in DAT["parses"]:
        stimname = P["name"]
        parses = P["parse"]
        print(stimname)
        
        from os import path
        savename = "{}/{}.pickle".format(savedir, stimname)

        if not path.exists(savename):
            datflat_ec = parses2datflat(parses, stimname=stimname, condition=DAT["trainset"], randomsubsample=randomsubsample)    
            with open(savename, "wb") as f:
                pickle.dump(datflat_ec, f)
        else:
            print("SKIPPING {} since already done!".format(savename))

def getAndSaveRandomParses(DAT, Nperm=1000):
    """gets N random permutations. uses the sequence, and ignores the actual aprses
    Have to first get parses before run this"""
    from pythonlib.tools.listtools import permuteRand

    # ==== for each stimulus (solved), get N random permutations 
    stimlist = DATgetSolvedStim(DAT, intersectDrawgood=True, onlyifhasdatflat=True)
    for stim in stimlist:
        # get datsegs for this stim (and just keep the first parse)
        datseg_single = DATloadDatSeg(DAT, stim)[0]
        # collect N permutation
        print(type(permuteRand))
        datseg_randomperm = permuteRand(datseg_single, Nperm, includeOrig=False, not_enough_ok=True)
        # save
        DATsaveDatSeg(DAT, datseg_randomperm, "{}_randomperm".format(stim))
        print("saved {}".format("{}_randomperm".format(stim)))



def updateParsesMirrorSymmetry(DAT):
    """update set of parses so that has mirror symmetry"""    
    # load parses

    # for each parse get 
    pass

# ==== do segmentation
if __name__=="__main__":
    IMPORT_DRAWGOOD=False
else:
    IMPORT_DRAWGOOD=True

IMPORT_DRAWGOOD = False
if IMPORT_DRAWGOOD:
    import sys
    sys.path.append("/Users/lucastian/tenen/TENENBAUM/drawgood/experiments")
    import segmentation as dgseg
    import utils as dgutils
    import plotsDatFlat as dgpflat
    import plotsSingleSubj as dgpsing
    import preprocess as dgprep
    import modelAnaly as dgmodel
    from segmentation import getSegmentation

if __name__=="__main__":
    import sys

    # 1) experiment name
    experiment = sys.argv[1]
    # 2) do parse? {2}=only get random perm {1, empty}=yes, {0}=no
    # if len(sys.argv)>2:
    #     doparse = int(sys.argv[2])
    # else:
    #     doparse = 1

    skipthingsthatcrash=False
    REMOVELL = False    

    print("getting all parses (may take a while")
    getAndSaveParses(experiment=experiment, skipthingsthatcrash=skipthingsthatcrash)

    #     # === get datflat
    #     print("GETTING DATFLAT (computing and then saving")
    #     DAT = loadCheckpoint(trainset=experiment, loadparse=True, suppressPrint=True)
    #     parses2datflatAllSave(DAT)

    #     # === get datseg
    #     # -- for each stim, load datflat, do segmentation, save..
    #     from segmentation import getSegmentation

    #     print("GETTING DATSEGS (computing and then saving)")
    #     stims = DATgetSolvedStim(DAT, intersectDrawgood=True)
    #     for s in stims:
    #         print("getting datsegs for {}".format(s))
    #         # load datflat
    #         datflat = DATloadDatFlat(DAT, s)
            
    #         # 1) get datseg
    #         datseg = getSegmentation(datflat, unique_codes=True, dosplits=True, removebadstrokes=True, removeLongVertLine=REMOVELL) 
                
    #         # save datflat
    #         DATsaveDatSeg(DAT, datseg, s)

    # if doparse in [0,1,2]:
    #     # === get RAndom per:mutations
    #     print("GETTING RANDOM PERMUTATIONS")
    #     DAT = loadCheckpoint(trainset=experiment, loadparse=True, suppressPrint=True)
    #     getAndSaveRandomParses(DAT, Nperm=1000)
    # 