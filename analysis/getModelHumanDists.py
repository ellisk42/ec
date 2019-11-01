## === script that, given a model dataset and human dataset, does comparison
# ==== GET ALL STRING DISTANCES, BETWEEN ALL STIM X SUBJECTS
import sys

sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")
from analysis.utils import *

sys.path.insert(0, "/Users/lucastian/tenen/TENENBAUM/drawgood/experiments")
sys.path.insert(0, "/home/lucast4/drawgood/experiments")
from modelAnaly import distModelHumanAllStims
from segmentation import getSegmentation
from preprocess import getFlatData

import segmentation as dgseg
import utils as dgutils
import plotsDatFlat as dgpflat
import plotsSingleSubj as dgpsing
import preprocess as dgprep
import modelAnaly as dgmodel



REMOVELL = True # remove vertical long line?



########################## STUFF TO DO AFTER SAVING DISTANCES
def loadDistances(ECTRAIN):
    # For a given slice of human/model/stim, plot all string distances
    # ACTUALLY: loads all into one list of dicts
    
    # load model
    DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True, suppressPrint=True)

    # load each stim/human
    stimlist = DATgetSolvedStim(DAT, intersectDrawgood=True)
    DAT = DATloadDrawgoodData(DAT, dosegmentation=True)
    humans = set([d["workerID"] for d in DAT["datflat_hu"]])

    # distances = []
    # for stim, hum in zip(stimlist, humans):
    #     print({"getting {}, {}".format(stim, hum)})
    #     distances.extend(DATloadModelHuDist(DAT, stim, hum))

    distances = []
    for d in DAT["datflat_hu"]:
        stim = d["stimname"]
        stim = stim[:stim.find(".png")]
        if stim in stimlist: # then model sucecsffuly found soolution
            hum = d["workerID"]
            print("getting {}, {}".format(stim, hum))
            distances.extend(DATloadModelHuDist(DAT, stim, hum))
    return distances


def filterDistances(distances, stimlist=[], humans=[], models=[], modelrend=[]):
    if len(stimlist)==0:
        stimlist = set(d["stim"] for d in distances)
    if len(humans)==0:
        humans = set(d["human"] for d in distances)
    if len(models)==0:
        models = set(d["model"] for d in distances)
    if len(modelrend)==0:
        modelrend = set(d["modelrend"] for d in distances)
    
    dist =[]
    for d in distances:
        if d["stim"] in stimlist and d["human"] in humans and d["model"] in models and d["modelrend"] in modelrend:
            dist.append(d)
    return dist


if __name__=="__main__":
    ################## SCRIPT TO EXTRACT ALL DISTANCES AND SAVE

    # === INPUT ARGUMENTS
    ECTRAIN = sys.argv[1]
    # ECTRAIN = "S8.2.2"
    # ECTRAIN = "S9.2"
    REMOVELL = True # this must match with preprocessing of model


    # load DAT
    DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True, suppressPrint=True)
    print("Loaded checkpoint DAT, for {}".format(ECTRAIN))

    # get list of stimuli
    stimlist = DATgetSolvedStim(DAT, removeshaping=True)

    # Extract segmented data for humans
    DAT["datflat_hu"] = getFlatData(DAT["datall_human"])
    DAT["datseg_hu"] = getSegmentation(DAT["datflat_hu"], unique_codes=True, dosplits=True, removeLongVertLine=REMOVELL)                                      

    # for each human, get data structure containing all distances
    # distances = []
    for i, (dflat, dseg) in enumerate(zip(DAT["datflat_hu"], DAT["datseg_hu"])):
            
        stimthis = dflat["stimname"]
        stimthis = stimthis[:stimthis.find(".png")]
        humanname = dflat["workerID"]

        print("getting {} of stim x subject, out of {}, ({})({})".format(i, len(DAT["datflat_hu"]), stimthis, humanname))
        
        # skip if doesn't have modeling data
        if stimthis not in stimlist:
            print("SKIPPED, not in stimlist that model got")
            continue
        
        def seqgetter_hu(stim):
            sequence_human = [d["codes_unique"] for d in dseg]
            return sequence_human
        
        def seqgetter_ec(stim):
            datseg = DATloadDatSeg(DAT, stim)
            return [[d["codes_unique"] for d in dat] for dat in datseg]
        
        # DATloadDatSeg(DAT, stim)
        dflat = DATloadDatFlat(DAT, stimthis)
        modelparsenums = [d["parsenum"] for d in dflat]

        distances = distModelHumanAllStims([stimthis], seqgetter_ec, seqgetter_hu, modelname = ECTRAIN, humanname = humanname, distancelabel="dist", modelrends=modelparsenums)
        DATsaveModelHuDist(DAT, stimthis, humanname, distances)