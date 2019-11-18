"""Code to plot and analyze model results,. inclding combining
pl;anner and dreamcoder models"""
"""NOTE: this is newer version thatn modelAnalyses. Use this. Here using the same params across everyone (i.e, getting average param)
Also updated a few functions. To merge with modelAnalyses, should add ability here to use subject specific motor params."""





# ===== load multiple models and concatenate
import sys
sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")
from analysis.importDrawgood import dgplan
addLabel = dgplan.addLabel
# from analysis.modelPlanning import addLabel
from pythonlib.tools.dicttools import printOverviewKeyValues
from analysis.importDrawgood import *
from analysis.parse import *
from analysis.getModelHumanDists import *
import os

def loadMultDCHumanDistances(ECTRAINlist, modelkind_list, ver, use_withplannerscore=False):
    """loads mutlpel and contatenates into a heirarchical list"""
    # use_withplannerscore, if True, then loads the version that already has
    # planner score appended. This is gotten from modelParsesGet...py
#     modelkind_list = ["parse", "randomperm"]
    # ver="codes_unique"

    # 1) load all experimetn dta
    DAT_all, workerlist, SAVEDIR = loadMultDCdata(ECTRAINlist)
    
    distances_all = []
    for DAT in DAT_all:
        for modelkind in modelkind_list:
            if ver=="aggregate" and modelkind=="randomperm":
                if not use_withplannerscore:
                    print("then skip, this doesn't exist, since aggregation code combines randomperm and parse into one summary dict")
                    continue
                else:
                    # just a hack, since I mistakenly saved random
                    # perm as a separate file after appending the distances. so should load it now.
                    assert True
            distances = loadDistances2(DAT, ver=ver, modelkind=modelkind, use_withplannerscore=use_withplannerscore)
            # distances = loadDistances(ECTRAIN, ver=ver, modelkind=modelkind, use_withplannerscore=use_withplannerscore)
            distances_all.append(distances)
    
    # Combine all models into one flat list of dicts
    distances_flat = [d for distances in distances_all for d in distances]
    return distances_flat, DAT_all, workerlist, SAVEDIR



# def loadMultDCHumanDistances(ECTRAINlist, modelkind_list, ver, use_withplannerscore=False):
#     """loads mutlpel and contatenates into a heirarchical list"""
#     # use_withplannerscore, if True, then loads the version that already has
#     # planner score appended. This is gotten from modelParsesGet...py
# #     modelkind_list = ["parse", "randomperm"]
#     # ver="codes_unique"
#     distances_all = []
#     for ECTRAIN in ECTRAINlist:
#         for modelkind in modelkind_list:
#             if ver=="aggregate" and modelkind=="randomperm":
#                 if not use_withplannerscore:
#                     print("then skip, this doesn't exist, since aggregation code combines randomperm and parse into one summary dict")
#                     continue
#                 else:
#                     # just a hack, since I mistakenly saved random
#                     # perm as a separate file after appending the distances. so should load it now.
#                     assert True
#             distances = loadDistances(ECTRAIN, ver=ver, modelkind=modelkind, use_withplannerscore=use_withplannerscore)
#             distances_all.append(distances)
    
#     # Combine all models into one flat list of dicts
#     distances_flat = [d for distances in distances_all for d in distances]
#     return distances_flat

    
def loadMultDCdata(ECTRAINlist):
    # figure out the experimental condition for each subject
    DAT_all = []
    for ECTRAIN in ECTRAINlist:
        DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=False, suppressPrint=True, loadbehavior=True)
        DAT_all.append(DAT)
    
    # get list of workers
    workerlist_all =[]
    for DAT in DAT_all:
        workerlist_all.append(dgutils.getWorkers(DAT["datall_human"]))
    workerlist = [ww for w in workerlist_all for ww in w]

    # make savedir
    SAVEDIR = "analysis/summaryfigs/acrossexpt/ec{}_{}-dg{}_{}".format(
        DAT_all[0]["trainset"], DAT_all[1]["trainset"], DAT_all[0]["behaviorexpt"], DAT_all[1]["behaviorexpt"])
    os.makedirs(SAVEDIR, exist_ok=True)
    
    return DAT_all, workerlist, SAVEDIR

def savedirDat(ECTRAINlist):
    """dir to save data (diff from dir to save figures)"""
    SAVEDIRDAT = "analysis/summary/acrossexpt/ec{}_{}-dg{}_{}".format(
        DAT_all[0]["trainset"], DAT_all[1]["trainset"], DAT_all[0]["behaviorexpt"], DAT_all[1]["behaviorexpt"])
    os.makedirs(SAVEDIRDAT, exist_ok=True)
    return SAVEDIRDAT
    


def addWorkerCond(distances_flat, workerlist):
    # include expt condition for each worker
    for d in distances_flat:
        W = [w["condition"] for w in workerlist if w["workerID"]==d["human"]]
        # assert all(x == W[0] for x in W), "why different conditions?"
        assert len(W)>0, "did not load the correct behaviroal expt"
        d["human_cond"]=W[0]


def loadPlannerData(EXPT, BATCHNAME):
    ## load planner model parameters
    from analysis.importDrawgood import dgplan as D
    Planner = D.Planner
    
    # 1) load model fit
    summarydict_all, datall, savedir = D.loadModelFitsIndiv(EXPT, BATCHNAME)
    
    return Planner, summarydict_all, datall, savedir


def getParamValues(params, params_list, workerID=[]):
    """given list of param names outputs the params for thsi owrke in order
    e.g, params_list=['start', 'motor_dist', 'motor_dir']
    if workerID=[] then assumes that params is one column, e.g, the average over workers."""
    import numpy as np
    paramvals = []
    for name in params_list:
        if workerID:
            paramvals.append(params.loc[params["xname"]==name, workerID].values)
        else:
            paramvals.append(params.loc[params["xname"]==name, "xmean"].values)
#         paramvals.append(params[params["xname"]==name][workerID].values)
#     print(paramvals)
    assert len(paramvals)==len(params_list)
    return np.array(paramvals)
    

############################# combining with planner


if __name__=="__main__":
    # NOTE: This is not advised, since this takes a really long time, since
    # does each person separately, this is not necessary if I am using the same parametres
    # for everyone. even if using diff params for each person, is still too costly.
    # since makes a new planner every iteration... should cache it.
    
    from analysis.importDrawgood import dgplan as D
    from scipy.special import logsumexp

    Planner = D.Planner

    # ==== Dreamcoder-huamn params.
    ECTRAINlist = ["S12.10.test4", "S13.10.test4"]
    modelkind_list = ["parse", "randomperm"]
    ver="aggregate"
    randomsubset=10000 # if [], then will get all parses. if an int, then will
    # subsample without replacement.

    # ===== Planner model to load
    BATCHNAME = "191108"
    EXPT = "2.4"

    # ===== Params for current -rescoring
    PLANVERDICT={
        "motor":['start', 'motor_dist', 'motor_dir'],
        "motorplusVH":['start', 'motor_dist', 'motor_dir', 'cog_vertchunker'],
        "full":['start', 'motor_dist', 'motor_dir', 'cog_primtype', 'cog_vertchunker', 'cog_vertchunker_LL']}


    for planver, params_list in PLANVERDICT.items():
        ###################### RUN PREPROECESS
        # 1) load human-dc distances
        distances_flat = loadMultDCHumanDistances(ECTRAINlist, modelkind_list, ver)
        # 2) load DC data
        DAT_all, workerlist, SAVEDIR = loadMultDCdata(ECTRAINlist)
        SAVEDIRDAT = savedirDat(ECTRAINlist)
        print("SAVING AT {}".format(SAVEDIRDAT))
        # 3) add worker conditions
        addWorkerCond(distances_flat, workerlist)
        # 4) Load Planner data
        print("LOADING planner model fit data")
        Planner, summarydict_all, datall, savedir = loadPlannerData(EXPT, BATCHNAME)
        # params = D.getParamsAggregate(D.flattenSummary(summarydict_all), returnnames=True)

        # 5) Get Planner using mean data across subjects
        print("Settuing up a planner object using previously fit params")
        params = D.getParamsAggregate(D.flattenSummary(summarydict_all), returnnames=True, meanoverworkers=True)
        planner = Planner(paramslist=params_list, handparams=getParamValues(params, params_list))
        print("--- Got params {} for {}, for planver {}".format(params, params_list, planver))
        

        ##################### RUN, ITERATING OVER ALL MODELS/HUMANS/STIMS
        # NOTE: probabilities will only be accurate if comparing within grouyp of stim x model x human
        # stim = "S12_10"
        # model = "S12.10"
        humanlist = set([d["human"] for d in distances_flat])
        stimlist = set([d["stim"] for d in distances_flat])
        modellist = set([d["model"] for d in distances_flat])

        # map model to dc trainset [just for converting names...]
        modelmap = {
            "S13.10.test4_randomperm":"S13.10.test4",
            "S13.10_randomperm":"S13.10",
            "S12.10.test4_randomperm":"S12.10.test4",
            "S12.10_randomperm":"S12.10"
        }

        for model in modellist:
            distances_flat_new = []
            for stim in stimlist:
                for human in humanlist:
                    # get planner model

                    # get distances for just this human/model/stim
                    distances_this = [d for d in distances_flat if d["stim"]==stim and d["model"]==model and d["human"]==human]
                    idx_original = [i for i,d in enumerate(distances_flat) if d["stim"]==stim and d["model"]==model and d["human"]==human]
                    
                    if len(distances_this)==0:
                        continue

                    if isinstance(randomsubset, int):
                        # get random subsample
                        import random
                        if len(distances_this)>randomsubset:
                            distances_this = random.sample(distances_this, randomsubset)
                        idx_original = [] # this is not accurate since sunsampled, so erase.
                        
                    # load datseg
                    if "randomperm" in model:
                        trainset = modelmap[model]
                    else:
                        trainset = model
                    DAT = [d for d in DAT_all if d["trainset"]==trainset][0]
                    datsegs = DATloadDatSeg(DAT, stim)
                    # make a dict for datseg primtiives
                    datsegs_dict = {d["codes_unique"]:d for d in datsegs[0]}

                    # sanity check, find all the sequencse for this model
                    if False:
                        # skip, since may take time and assertion always been passing.
                        if "randomperm" not in model:
                            seqset_parses = set(["_".join(d["sequence_model"]) for d in distances_this])
                            seqset_all = set(["_".join([d["codes_unique"] for d in dseg]) for dseg in datsegs])
                            assert seqset_parses.issubset(seqset_all)

                    # 2) get datsegs that corresponds only, and exactly, to the parses in this set of scores.
                    datsegs_sub = []
                    for d in distances_this:
                        seqthis = d["sequence_model"]
                        datsegs_sub.append([datsegs_dict[s] for s in seqthis])

                    # === run planner model 
                    # 1) use subject specific params
                #     planner = Planner(paramslist=params_list, handparams=getParamValues(params, human, params_list) + np.array([10,1,1]).reshape(-1,1))
                    scores = planner.scoreMultTasks(datsegs_sub, Nperm=[], getRawScore=True, returnAllScores=True)
                    # convert scores to log probabilities takig into account all scores
                    
                    if False:
                        scores_norm = []
                        for s in scores:
                            scores_norm.append(planner.scoreSoftmax(s, scores))
                        scores_norm = np.exp(scores_norm)
                        scores = scores_norm
                    else:
                        scoressum = logsumexp(scores)
                        scores = [s-scoressum for s in scores]

                    # out scores back into dict in order
                    if False:
                        # old version, putthing score  back into originnal distances. doesn't work now because have option to subsamp.e
                        for i, sc in zip(idx_original, scores):
                            distances_flat[i]["planner_prob"]=sc.item()
                    else:
                        for d, sc in zip(distances_this, scores):
                            d["planner_prob"]=sc.item()
                        distances_flat_new.extend(distances_this)
                        

                    print("DONE for {}, {}, {}, {}".format(planver, stim, model, human))

            import pickle
            fname = "{}/modhudist_planner{}_model{}.pickle".format(SAVEDIRDAT, planver, model)
            with open(fname, "wb") as f:
                if True:
                    pickle.dump(distances_flat_new, f)
                else:
                    pickle.dump(distances_flat, f)
