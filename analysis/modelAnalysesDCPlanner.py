"""analyses that combines Dreamcoder output wiht planner"""
## NOTE: THIS MAY BE OBSOLETE?

import sys
sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")
from analysis.getModelHumanDists import * 


# ===== load multiple models and concatenate
from pythonlib.tools.dicttools import printOverviewKeyValues
from analysis.modelAnalyses import *


def reweightDistsByPlanner(distances_flat, planner_model):
    """each dict should have <planner_model>_prob as a key.
    this shouidl be a probabilty
    will mulitply the dist by this probabilty to reweight. will 
    aggregagte over all dataponts."""

    stimlist = [d["stim"] for d in distances_flat]
    humanlist = [d["human"] for d in distances_flat]
    modellist = [d["model"] for d in distances_flat]

    import numpy as np

    count=0
    distances_weighted = []
    for stim in stimlist:
        for human in humanlist:
            for model in modellist:
                # -- get all datapoints
                print(count)
                count+=1
                dists = filterDistances(distances_flat, [stim], [human], [model])

                # print("SUM: shouldnt this be 1?")
                print(np.sum([d["{}_prob".format(planner_model)] for d in dists]))
                # print("getting weighted average using planner model scores")
                
                D = [d["dist"] for d in dists]
                P = [d["{}_prob".format(planner_model)] for d in dists]            
                weighted_d = np.average(D, weights=P)
                
                distances_weighted.append({
                    "stim":stim,
                    "human":human,
                    "model":model,
                    "dist":weighted_d,
                })
    return distances_weighted

def aggregateOverParses(df, ver="median"):
    """various methods to aggregate over parses"""
    if ver=="median":
        PRCTILE=50
    else:
        assert False, "NOT CODED"
    am = lambda x: np.percentile(x, PRCTILE)
    df = aggregGeneral(df, group=["stim", "human", "model"], values=["dist"], nonnumercols=["human_cond"],
                  aggmethod=[am])
    return df




from analysis.importDrawgood import dgplan as D
Planner = D.Planner
def extractAndSaveReweightedDists(ECTRAINlist = ["S12.10.test4", "S13.10.test4"],
    modelkind_list = ["parse", "randomperm"], ver="aggregate", planner_params={"BATCHNAME":"191116", "EXPT":"2.4", "planner_model":"motor"},
    ):
    """reweights distances by planner scores, can then do things like take
    median of scores to quantify fit"""
    from analysis.importDrawgood import dgplan as D

    ######################### PARAMS
    ## DREAMCODER
    # ECTRAINlist = ["S12.10.test4", "S13.10.test4"]
    # modelkind_list = ["parse", "randomperm"]
    # ver="aggregate"
    use_withplannerscore=True # this flags so that loads from distances that have planner scores appended.
    
    ## PLANNER
    # BATCHNAME = "191108"
    BATCHNAME = planner_params["BATCHNAME"]
    EXPT = planner_params["EXPT"]
    ## FOR REWEIGHTING DC BY PLANNER PARAMS
    planner_model=planner_params["planner_model"]

    # 2) Load Planner model data
    print("Loading params from Planner model")
    Planner = D.Planner
    Planner, summarydict_all, datall, savedir = loadPlannerData(EXPT, BATCHNAME)
    params = D.getParamsAggregate(D.flattenSummary(summarydict_all), returnnames=False)

    ########################## RUN
    # 1) Lopad Dreamcoder models
    distances_flat, DAT_all, workerlist, SAVEDIR = loadMultDCHumanDistances(ECTRAINlist, modelkind_list, ver, use_withplannerscore)
    addWorkerCond(distances_flat, workerlist)
    print("Extracted {} workers".format(len(set([d["workerID"] for d in workerlist]))))


    # 3) Reweight

    distances_weighted = reweightDistsByPlanner(distances_flat, planner_model)

    # 4) Save
    SD = "{}/summarydict_planner_{}.pickle".format(SAVEDIR, planner_model)
    print("Svaing at{}".format(SD))
    allsummarydict = {
        "distances_weighted":distances_weighted,
        "planner_params":params
    }
    with open(SD, "wb") as f:
        pickle.dump(allsummarydict, f, pickle.HIGHEST_PROTOCOL)


####################################### PLOTS
def plotPointOverview(df, SAVEDIR, suffix=""):
    """PLOT EACH STIM [ONE SUBPLOT EACH STIM, SHOWING THE POINT ESTIMATE FOR EACH SUBECT"""
    import seaborn as sns
    ax = sns.catplot(data=df, hue="model", x="human", y="dist", col="stim", col_wrap=6)
    addLabel(ax)
    fname = "{}/point_overview_{}.pdf".format(SAVEDIR, suffix)
    print("SAVING AT {}".format("{}/point_overview_{}.pdf".format(SAVEDIR, suffix)))
    ax.savefig(fname)


if __name__=="__main__":
    from analysis.importDrawgood import dgplan as D
    Planner = D.Planner
    extractAndSaveReweightedDists()