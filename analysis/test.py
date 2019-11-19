## updated, scoring datsegs, before gotten distances
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
ECTRAINlist = ["S12.10.test4", "S13.10.test4"]
modelkind_list = ["parse", "randomperm"]
# ver="aggregate"
# randomsubset=10000 # if [], then will get all parses. if an int, then will
# subsample without replacement.

# ===== Planner model to load
BATCHNAME = "191108"
EXPT = "2.4"

# ===== Params for current -rescoring
PLANVERDICT={
    "motor":['start', 'motor_dist', 'motor_dir'],
    "motorplusVH":['start', 'motor_dist', 'motor_dir', 'cog_vertchunker'],
    "full":['start', 'motor_dist', 'motor_dir', 'cog_primtype', 'cog_vertchunker', 'cog_vertchunker_LL']}


# 4) Load Planner data
print("LOADING planner model fit data")
Planner, summarydict_all, datall, savedir = loadPlannerData(EXPT, BATCHNAME)
planner_objects = {}
for planver, params_list in PLANVERDICT.items():
    # 5) Get Planner using mean data across subjects
    print("Settuing up a planner object using previously fit params")
    params = D.getParamsAggregate(D.flattenSummary(summarydict_all), returnnames=True, meanoverworkers=True)
    planner = Planner(paramslist=params_list, handparams=getParamValues(params, params_list))
    print("--- Got params {} for {}, for planver {}".format(getParamValues(params, params_list), params_list, planver))
    planner_objects[planver]=planner


## go 1 by 1 thre previously saved segmented pickel files.
for ECTRAIN in ECTRAINlist:
    for modelkind in modelkind_list:
        
            DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True, suppressPrint=True, loadbehavior=False)
            stimlist = DATgetSolvedStim(DAT, onlyifhasdatflat=True)

            for stim in stimlist:
                print("Getting planner scores for {}, {}, {}".format(ECTRAIN, modelkind, stim))        
                
                # 1) load datsegs
                if modelkind == "randomperm":
                    stimname = "{}_randomperm".format(stim)
                else:
                    stimname = stim
                datsegs = DATloadDatSeg(DAT, stimname)
                
                # == get scores for each planner
                for planner_name, planner in planner_objects.items():
                    print("scoring using {}".format(planner_name))
                    
                    scores = planner.scoreMultTasks(datsegs, Nperm=[], getRawScore=True, returnAllScores=True)

                    # save
                    fname= "{}/{}_planscores_{}.pickle".format(DAT["savedir_datsegs"], stimname, planner_name)
                    with open(fname, "wb") as f:
                        pickle.dump(scores, f)
                        
