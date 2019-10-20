## here taking dreamcoder outputs and parsing into sequences. 
## will integrate with behaviral data to do modeln

import sys
sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
# print(sys.path)
from dreamcoder.domains.draw.drawPrimitives import *
from dreamcoder.domains.draw.primitives import _repeat, _line, _makeAffine, _circle,_connect
from dreamcoder.domains.draw.makeDrawTasks import makeSupervisedTasks, SupervisedDraw
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
import numpy as np
from analysis.utils import *



def getParses(dreamcoder_program):
    """ e.g., dreamcoder_program = result.allFrontiers[tasks[i]].bestPosterior.program"""
    # extracts all parses for a given program object
    print("getting parse...")
    parses = Parse.ofProgram(Program.parse(str(dreamcoder_program)))
    return parses


def getAndSaveParses(experiment="S9.2"):
    result, tasks, testtasks, programnames, program_test_names, behaviorexpt, savedir = loadCheckpoint(trainset=experiment)

    # === for each program, get the best posteiror and then all parses of that. 
    for t, name in zip(tasks, programnames):
        print("Parsing {} ...".format(name))
        fname = "{}/parses_{}.pickle".format(savedir, name)
        if result.frontiersOverTime[t][-1].empty:
            parses =[]
        else:
            p = result.frontiersOverTime[t][-1].bestPosterior.program
            parses = getParses(p)
            # parses = [1,2,3]
        with open(fname, "wb") as f:
            pickle.dump(parses, f)
        print("saved to :{}".format(fname))

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
    for i, prog in enumerate(programs):
        print("prog {}".format(i))
        if isinstance(prog, Parse):
            prog = prog.flatten()
        # append fake timestamps, so that prog become a strokes list:
        strokes = program2strokes(prog)

        # create a new entry in datflat, 
        datflat.append({
            "parsenum": i, # this is unique to dreamcoder; each program has multiple parses
            "trialstrokes":strokes,
            "trialonset": 0,
            "stimname": stimname, # TODO, replace with actual stim name
            "trialprimitives":[],
            "trialcircleparams":[],
            "condition":"",
        })
    return datflat



# ==== do segmentation
import sys
sys.path.append("/Users/lucastian/tenen/TENENBAUM/drawgood/experiments")
import segmentation as dgseg
import utils as dgutils
import plotsDatFlat as dgpflat
import plotsSingleSubj as dgpsing
import preprocess as dgprep


if __name__=="__main__":
    import sys
    experiment = sys.argv[1]
    print(experiment)
    print(type(experiment))
    getAndSaveParses(experiment=experiment)