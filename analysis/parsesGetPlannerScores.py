## for each dc model datsegs (i.e. parses, before comapring to huamns), scores those parses
# using different planner models. Currently agnostically takes params over all planmner models
# that are in a given directory, so this could average over 3-param models, 4-param models, etc.
# then saves, next to datsegs, a score for each datsets.

import sys
sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")

from analysis.utils import *
from analysis.parse import *
from analysis.analy import *
from analysis.modelAnalyses import *
from analysis.importDrawgood import dgplan as D
from scipy.special import logsumexp

Planner = D.Planner

# ==== Dreamcoder-huamn params.

modelkind_list = ["parse", "randomperm"]
# ver="aggregate"
# randomsubset=10000 # if [], then will get all parses. if an int, then will
# subsample without replacement.

# ===== Planner model to load
# NOTE: this doesnt necesasrily have to include all the subjects, since
# will take the average params and use those to weigh behavior.
# BATCHNAME = "191116"
# BATCHNAME = "200118"
BATCHNAME = str(sys.argv[1])

if sys.argv[2]=="S12":
    ECTRAINlist = ["S12.10.test5"]
elif sys.argv[2]=="S13":
    ECTRAINlist = ["S13.10.test5"]
elif sys.argv[2]=="S12_S13":
    ECTRAINlist = ["S12.10.test5", "S13.10.test5"]
else:
    assert False, "which dreamcoder models to use?"

# ECTRAINlist = ["S12.10.test4", "S13.10.test4"]
# ECTRAINlist = ["S12.10.test4"]
# ECTRAINlist = ["S13.10.test4"]

EXPT = "2.4"

# ===== Params for current -rescoring
# PLANVERDICT={
#     "motor":['start', 'motor_dist', 'motor_dir'],
#     "motorplusVH":['start', 'motor_dist', 'motor_dir', 'cog_vertchunker'],
#     "full":['start', 'motor_dist', 'motor_dir', 'cog_primtype', 'cog_vertchunker', 'cog_vertchunker_LL']}
PLANVERDICT={
    "motorplusVH":['start', 'motor_dist', 'motor_dir', 'cog_vertchunker']}

# === OTHER GENERAL PARAMS THAT APPLY TO ALL PLANERS
REMOVELL=[False]
REGULARIZE=["l2"]
hasLLparam = [False, True]
STIMSET = ["train"]

# --- LOAD PLANNER PARAMS
Planner, summarydict_all, datall, savedir = loadPlannerData(EXPT, BATCHNAME)
summaryflat = D.flattenSummary(summarydict_all)

# planner_agg_version = "common" # then takes average across all subjects
# planner_agg_version = "bycondition" # then averages over wrokers within each condition. need conditiondict for this to work.
for planner_agg_version in ["common", "bycondition"]:
    if planner_agg_version=="common":
        # 4) Load Planner data
        print("LOADING planner model fit data")

        planner_objects = {}
        for planver, params_list in PLANVERDICT.items():
            # 5) Get Planner using mean data across subjects
            # NOTE: this avearges over all planner models..

            # 1) Get params for just the experiments that fit this plannerlist.
            F = {
               "REMOVELL":REMOVELL,
               "REGULARIZE":REGULARIZE,
               "hasLLparam":hasLLparam,
               "STIMSET":STIMSET,
               "PARAMSLISTstr":[", ".join(params_list)]}

            params = D.getParamsAggregate(summaryflat, filterdict=F, returnnames=True, meanoverworkers=True)
            assert len(params)==len(params_list), f"likely do not have a planner model fit with exactly these {len(params_list)} params"

            # 2) Set up planner object
            print("Settuing up a planner object using previously fit params")
            param_vals = getParamValues(params, params_list)
            planner = Planner(paramslist=params_list, handparams=param_vals)
            print("--- Got params {} for {}, for planver {}".format(getParamValues(params, params_list), params_list, planver))
            planner_objects[planver]=planner

    elif planner_agg_version=="bycondition":
        # - get worker list
        from analysis.importDrawgood import dgutils
        workerlist = dgutils.getWorkers(datall)

        # - get one planner vector for each behavioral condition. ectrainname:condition for drawgood subjects whos
        # training matches the training that the ecmodel got.
        conditiondict = {
            "S12.10.test5":0, 
            "S13.10.test5":1
        }
        planner_objects={}
        for planver, params_list in PLANVERDICT.items():
            F = {
               "REMOVELL":REMOVELL,
               "REGULARIZE":REGULARIZE,
               "hasLLparam":hasLLparam,
               "STIMSET":STIMSET,
               "PARAMSLISTstr":[", ".join(params_list)]}

            for ectrain, condition in conditiondict.items():
                # - get list of subjects in this condition
                # get average params over these subjects
                workersthis = [w["workerID"] for w in workerlist if w["condition"]==condition]
                summaryflat_this = [s for s in summaryflat if s["workerID"] in workersthis]
                
                # get params
                params = D.getParamsAggregate(summaryflat_this, filterdict=F, returnnames=True, meanoverworkers=True)
                assert len(params)==len(params_list), f"likely do not have a planner model fit with exactly these {len(params_list)} params"

                param_vals = getParamValues(params, params_list)

                # make planner
                planner = Planner(paramslist=params_list, handparams=param_vals)

                # - save
                keyname = f"{planver}_{ectrain}"
                planner_objects[keyname]=planner



    ## go 1 by 1 thre previously saved segmented pickel files.
    for ECTRAIN in ECTRAINlist:
        for modelkind in modelkind_list:
            
                DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=False, suppressPrint=True, loadbehavior=False)
                stimlist = DATgetSolvedStim(DAT, onlyifhasdatflat=True)

                assert len(stimlist)>0, "no data in datflat?"

                for stim in stimlist:
                    
                    # 1) load datsegs
                    if modelkind == "randomperm":
                        stimname = "{}_randomperm".format(stim)
                    else:
                        stimname = stim
                    datsegs = DATloadDatSeg(DAT, stimname)
                    
                    # == get scores for each planner
                    # for planner_name, planner in planner_objects.items():
                    for planner_name in PLANVERDICT.keys():
                        
                        if planner_agg_version=="bycondition":
                            planner_key = f"{planner_name}_{ECTRAIN}"
                        elif planner_agg_version=="common":
                            planner_key = planner_name

                        planner = planner_objects[planner_key]

                        # print("scoring using {}".format(planner_key))
                        print("Getting planner scores for {}, {}, {}, {}".format(ECTRAIN, modelkind, stim, planner_key))        

                        scores = planner.scoreMultTasks(datsegs, Nperm=[], getRawScore=True, returnAllScores=True)

                        assert len(scores)>0, "not sure why..."

                        # save
                        fname= "{}/{}_planscores_{}.pickle".format(DAT["savedir_datsegs"], stimname, planner_key)
                        with open(fname, "wb") as f:
                            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
                            
