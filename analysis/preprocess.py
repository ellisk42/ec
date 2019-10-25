# ===== for a given dreamcoder exeriment, script to preprocess all parses
import sys

sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")
# print("ADAS1")
from analysis.utils import loadCheckpoint, DATgetSolvedStim, DATloadDatFlat, DATsaveDatSeg
# print("ADAS2")
from analysis.parse import parses2datflatAllSave
# from analysis.analy import *
# print("ADAS3")

# == import things from drawgood.
sys.path.insert(0, "/Users/lucastian/tenen/TENENBAUM/drawgood/experiments")
sys.path.insert(0, "/home/lucast4/drawgood/experiments")
from segmentation import getSegmentation
# import utils as dgutils
# import plotsDatFlat as dgpflat
# import plotsSingleSubj as dgpsing
# import preprocess as dgprep
# import modelAnalyadsfads as dgmodel





if __name__=="__main__":
	ECTRAIN = sys.argv[1]
	print(ECTRAIN)
	REMOVELL=True	# if true, removes vertical bars from analyses.
	# ECTRAIN = "S8.2.2"
	# ECTRAIN = "S9.2"

	# 1) Get parses, if not already done
	print("NEED TO FIRST RUN: python analysis/parse.py {}".format(ECTRAIN))

	# 2) Load experiment data structure
	DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True)
	print("Loaded checkpoint DAT, for {}".format(ECTRAIN))

	# 4) for each stimulus, make a datflat (dreamcoder). save in DAT the path to that datflat
	print("Making datflat for each stimulus for dreamcoder (may take a while...)")
	parses2datflatAllSave(DAT)

	# 5) do segmentation of all datfalts
	# Dreamcoder
	print("Making datseg for each stimulus for dreamcoder (may take a while...)")
	stims = DATgetSolvedStim(DAT)	# -- for each stim, load datflat, do segmentation, save...
	for s in stims:
	    print("getting datsegs (dremacoder) for {}".format(s))
	    datflat = DATloadDatFlat(DAT, s)	    # load datflat
	    datseg = getSegmentation(datflat, unique_codes=True, dosplits=True, removebadstrokes=True, removeLongVertLine=REMOVELL) # get datseg
	    DATsaveDatSeg(DAT, datseg, s) # save datflat



	# # Human
	# DATFLAT["datseg_hu"] = dgseg.getSegmentation(DAT["datflat_hu"], unique_codes=True, dosplits=True, removeLongVertLine=REMOVELL)                                      

	# # 3) Load human data
	# DAT["datflat_hu"] = dgprep.getFlatData(DAT["datall_human"])
