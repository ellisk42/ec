"""Use parse.py first to get parses. then run this to process the parses"""
import sys

sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")

from analysis.utils import *
from parse import parses2datflatAllSave, getAndSaveRandomParses
from segmentation import getSegmentation




if __name__=="__main__":

    ############################### INPUT PARAMS
    experiment = sys.argv[1]
    subsampleparses=10000 # parse-->datflat, subsample randomly with replacement.
    randomparses = 1000

    ################################ HARD PARAMS
    REMOVELL = False    

    ################################ RUN
    # === get datflat
    print("GETTING DATFLAT (computing and then saving")
    if isinstance(subsampleparses, int):
        print("GETTING SUBSAMPLE (PARSE-->DATFLAT) of {}".format(subsampleparses))
        
    DAT = loadCheckpoint(trainset=experiment, loadparse=True, suppressPrint=True)
    parses2datflatAllSave(DAT, randomsubsample=subsampleparses)

    # === get datseg
    # -- for each stim, load datflat, do segmentation, save..
    print("GETTING DATSEGS (computing and then saving)")
    stims = DATgetSolvedStim(DAT, intersectDrawgood=True, onlyifhasdatflat=True)
    for s in stims:
        print("getting datsegs for {}".format(s))
        from os import path
        if path.exists("{}/{}.pickle".format(DAT["savedir_datsegs"], s)):
            print("Skipping {}. since already done".format(s))
            continue
        # load datflat
        datflat = DATloadDatFlat(DAT, s)

        # 1) get datseg
        datseg = getSegmentation(datflat, unique_codes=True, dosplits=True, removebadstrokes=True, removeLongVertLine=REMOVELL) 
        assert len(list(set([d["codes_unique"] for dseg in datseg for d in dseg])))==len([d["codes_unique"] for d in datseg[0]]), "i expected all permutations to have exact same tokens, just in different order. must be error in extracting after parsing?"            
        assert len(datflat)==len(datseg), "assuming they are perfectly matched"
  
        # save datflat
        DATsaveDatSeg(DAT, datseg, s)

    # === get RAndom per:mutations
    print("GETTING RANDOM PERMUTATIONS")
    DAT = loadCheckpoint(trainset=experiment, loadparse=True, suppressPrint=True)
    getAndSaveRandomParses(DAT, Nperm=randomparses)
