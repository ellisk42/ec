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

def getAndSaveParses(experiment="S9.2", debug=False, skipthingsthatcrash=False):
    """ gets the most recent, and the bestPosterior program
    gets all parses """
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

    def saveparses(T, P):
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
            if skipthingsthatcrash:
                if name=="S12_235":
                    print("SKIPPING - {} - since CRASHES...".format(name))
                    continue

            # Get bestPosterior solution (most recent, and best posterior)
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
                pickle.dump(parses, f)

            # 2) save flattened parses
            fname = "{}/parsesflat_{}.pickle".format(savedir, name)
            with open(fname, "wb") as f:
                pickle.dump([p.flatten() for p in parses], f)

            print("saved to :{}".format(fname))

    # ========= DO TRAIING TASKS
    saveparses(T=tasks, P=programnames)

    # ========= DO TEST TASKS
    if len(testtasks)>0:
        saveparses(T=testtasks, P=program_test_names)

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
def parses2datflat(programs, stimname="", condition=""):
    # ideally stimname is the name of stim for all parses (i.e progs)
    # ideally condition indicates what the training schedule was.

    # === A HACK, to make this work with code written fro beahviroal analysis, 
    # convert this list of programs --> list of strokes --> one datflat object
    # this datflat can then be treated just like a human subject's data.

    datflat = []
    print("getting datflat for {}".format(stimname))
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


def parses2datflatAll(DAT):
    # === this gets all into one datflat (i.e., stim x parses)
    datflat_ec = []
    for P in DAT["parses"]:
        # print(P)
        stimname = P["name"]
        parses = P["parse"]
        
        datflat_ec.extend(parses2datflat(parses, stimname=stimname, condition=DAT["trainset"]))
    return datflat_ec


def parses2datflatAllSave(DAT):
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
        
        datflat_ec = parses2datflat(parses, stimname=stimname, condition=DAT["trainset"])
        
        savename = "{}/{}.pickle".format(savedir, stimname)
        with open(savename, "wb") as f:
            pickle.dump(datflat_ec, f)


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

if __name__=="__main__":
    import sys
    experiment = sys.argv[1]
    if len(sys.argv)>2:
        doparse = sys.argv[2]
    else:
        doparse = 1

    REMOVELL = False    

    # === Get all parses, if desired
    if doparse==1:
        print("getting all parses (may take a while")
        getAndSaveParses(experiment=experiment)
    else:
        print("skipping parse as requested")

    # === get datflat
    print("GETTING DATFLAT (computing and then saving")
    DAT = loadCheckpoint(trainset=experiment, loadparse=True, suppressPrint=True)
    parses2datflatAllSave(DAT)

    # === get datseg
    # -- for each stim, load datflat, do segmentation, save...
    print("GETTING DATSEGS (computing and then saving)")
    stims = DATgetSolvedStim(DAT, intersectDrawgood=True)
    for s in stims:
        # print("getting datsegs for {}".format(s))
        # load datflat
        datflat = DATloadDatFlat(DAT, s)
        
        # 1) get datseg
        datseg = getSegmentation(datflat, unique_codes=True, dosplits=True, removebadstrokes=True, removeLongVertLine=REMOVELL) 
            
        # save datflat
        DATsaveDatSeg(DAT, datseg, s)
