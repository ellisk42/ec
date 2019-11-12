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
from segmentation import getSegmentation, codeUniqueFeatures

from preprocess import getFlatData

import segmentation as dgseg
import utils as dgutils
import plotsDatFlat as dgpflat
import plotsSingleSubj as dgpsing
import preprocess as dgprep
import modelAnaly as dgmodel



REMOVELL = False # remove vertical long line?



########################## STUFF TO DO AFTER SAVING DISTANCES
def loadDistances(ECTRAIN, ver="multiple", modelkind="parse"):
    # For a given slice of human/model/stim, plot all string distances
    # ACTUALLY: loads all into one list of dicts
    if ver=="legacy":
        suffixes = [""]
    elif ver=="multiple":
        # now calculating amny different distnaces
        suffixes = ["codes_unique", "codes", "row", "col"]
    elif ver=="aggregate":
        suffixes = ["aggregate"]
    elif ver=="medianparse":
        suffixes = ["medianparse"]
    else:
        assert False, "what version?"

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
    for suf in suffixes:
        for d in DAT["datflat_hu"]:
            stim = d["stimname"]
            stim = stim[:stim.find(".png")]
            if stim in stimlist: # then model sucecsffuly found soolution
                hum = d["workerID"]
                print("getting {}, {}".format(stim, hum))
                # print(d)
                # import pdb
                # pdb.set_trace()
                if modelkind=="parse":
                    d = DATloadModelHuDist(DAT, stim, hum, suf)
                elif modelkind=="randomperm":
                    d = DATloadModelHuDist(DAT, "{}_randomperm".format(stim), hum, suf)
                if isinstance(d, dict):
                    distances.append(d)    
                elif isinstance(d, list):
                    distances.extend(d)
    return distances


def filterDistances(distances, stimlist=[], humans=[], models=[], modelrend=[]):
    assert "stim" in distances[0].keys(), "code assumes stim is a key"
    assert "human" in distances[0].keys()
    assert "model" in distances[0].keys()

    if len(stimlist)==0:
        stimlist = set(d["stim"] for d in distances)
    if len(humans)==0:
        humans = set(d["human"] for d in distances)
    if len(models)==0:
        models = set(d["model"] for d in distances)
    skipmodelrend=False
    if len(modelrend)==0:
        if "modelrend" in distances[0].keys():
            modelrend = set(d["modelrend"] for d in distances)
        else:
            skipmodelrend=True
    
    dist =[]
    for d in distances:
        if skipmodelrend:
            if d["stim"] in stimlist and d["human"] in humans and d["model"] in models:
                dist.append(d)
        else:    
            if d["stim"] in stimlist and d["human"] in humans and d["model"] in models and d["modelrend"] in modelrend:
                dist.append(d)
    return dist




def updateWithPosNegControls(ECTRAIN):
    """positive control: for each human, gets distance from all other huamns"""
    pass


import pandas as pd
from pythonlib.tools.pandastools import *
def aggregateDistances(ECTRAIN, modelkindlist=["parse", "randomperm"]):
    """if many distances calcualted (e.g., using codes, codes_unique..., (1) extracrts, and aggregates, fefault is to take mean. (2) saves all back as "aggregat", then (3) takes median over all parses tp summarize"""
    print("NOTE: nonnumercols below assumes that codes_unique will be the first thin encountered, and therefore will be extracted in aggregationp.. this likely depend on the order of the distances kinds used in calcualting huamn-model distances. Is only important if use these codes later, e.g., if taking min over reversed sequences.")
    
    # modelkind=="parse":
    # modelkind=="randomperm"
    # ======== LOAD ALL MODELS AND COMBINE IN THIS ANALYSIS. Is fine since diff models are flagged with diff values for "model" key.
    distances_all = []
    for modelkind in modelkindlist:
        distances_all.append(loadDistances(ECTRAIN, modelkind=modelkind))
    distances = [d for dist in distances_all for d in dist] # fglatten

    DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True, suppressPrint=True)

    # 1) aggregate over distances, get some sort of aggregated mean
    df = aggregMean(pd.DataFrame(distances), ["stim", "human", "model", "modelrend"], values=["dist"], nonnumercols=["sequence_human", "sequence_model"])

    # 2) save, aggregate over distance measures
    print("=== DOING AGGREGATE")
    distances_agg = df.to_dict("records")
    humanlist = set(df["human"])
    stimlist = set(df["stim"])
    modellist = set(df["model"])
    assert len(modellist)==1, ""
    # --- Below is just for saving.
    for human in humanlist:
        for stim in stimlist:
            d = filterDistances(distances_agg, stimlist=[stim], humans=[human])
            if len(d)>0:
                DATsaveModelHuDist(DAT, stim, human, d, "aggregate")



    # 3) aggregate over parses by taking median
    print("=== DOING MEDIANS")
    df = aggreg(df, group=["stim", "human", "model"], values=["dist"], aggmethod=["median"])
    df["dist"]=df["dist_median"]
    df = df.drop(columns=["dist_median"])
    distances_median = df.to_dict("records")
    suff="medianparse"
    for d in distances_median:
        # print("---")
        # print(d)
        # try:
        #     if len(d["stim"])==0:
        #         import pdb
        #         pdb.set_trace()
        # except:
        #     import pdb
        #     pdb.set_trace()
        DATsaveModelHuDist(DAT, d["stim"], d["human"], [d], suff)






if __name__=="__main__":
    ################## SCRIPT TO EXTRACT ALL DISTANCES AND SAVE

    # === INPUT ARGUMENTS
    ECTRAIN = sys.argv[1]
    # ECTRAIN = "S8.2.2"
    # ECTRAIN = "S9.2"
    REMOVELL = False # this must match with preprocessing of model
    PARSEVERSIONLIST = ["parse", "randomperm"] # parse is kevin ellis code. random is random permutation.

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

        for parseversion in PARSEVERSIONLIST:
            # --- load datseg model data for this stim.
            if parseversion=="parse":
                datseg_ec = DATloadDatSeg(DAT, stimthis)
                dflat = DATloadDatFlat(DAT, stimthis)    
                modelparsenums = [d["parsenum"] for d in dflat]
                modelname=ECTRAIN
            elif parseversion=="randomperm":
                datseg_ec = DATloadDatSeg(DAT, "{}_randomperm".format(stimthis))
                modelparsenums = list(range(len(datseg_ec)))
                modelname="{}_randomperm"
            else:
                assert False, "dont unerstand..."

            def getSeqGetters(labelkind="codes_unique"):
                if labelkind=="codes_unique":
                    def seqgetter_hu(stim):
                        sequence = [d["codes_unique"] for d in dseg]
                        return sequence
                    def seqgetter_ec(stim):
                        sequence = [[d["codes_unique"] for d in dat] for dat in datseg_ec]
                        return sequence    
                elif labelkind=="codes":
                    def seqgetter_hu(stim):
                        sequence = [d["codes"] for d in dseg]
                        return sequence
                    def seqgetter_ec(stim):
                        sequence = [[d["codes"] for d in dat] for dat in datseg_ec]
                        return sequence
                elif labelkind=="col":
                    def seqgetter_hu(stim):
                        sequence = [codeUniqueFeatures(d["codes_unique"])[1] for d in dseg]
                        return sequence
                    def seqgetter_ec(stim):
                        sequence = [[codeUniqueFeatures(d["codes_unique"])[1] for d in dat] for dat in datseg_ec]
                        return sequence
                elif labelkind=="row":
                    def seqgetter_hu(stim):
                        sequence = [d["row"] for d in dseg]
                        return sequence
                    def seqgetter_ec(stim):
                        sequence = [[d["row"] for d in dat] for dat in datseg_ec]
                        return sequence    
                return seqgetter_hu, seqgetter_ec


            # =========== go thru each type of sequence
            for seqkind in ["codes_unique", "codes", "row", "col"]:
                seqgetter_hu, seqgetter_ec = getSeqGetters(labelkind=seqkind)

                distances = distModelHumanAllStims([stimthis], seqgetter_ec, seqgetter_hu, modelname = modelname, humanname = humanname, distancelabel=seqkind, modelrends=modelparsenums)
                DATsaveModelHuDist(DAT, stimthis, humanname, distances, seqkind)

    # ==== aggregate all distance measures, and get medians across parases.
    aggregateDistances(ECTRAIN, modelkindlist=PARSEVERSIONLIST)