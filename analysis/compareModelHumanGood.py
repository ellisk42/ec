""" 4/14/20, latest detailed comparison of model and human.
should also be general for other systems"""


from analysis.getModelHumanDists import * 
from analysis.modelAnalyses import *
from analysis.importDrawgood import *


def plotParses(distances_flat_this, datsegs, sort_by, priorver=None, 
              nplot = [0,1,-2,-1]):
    """ plots drawing steps for model parses 
    - distances_flat_this is list of dicts, with each dcit correstponding to 
    one parse version (e.g., permutation). this list only has entries
    for a single model of interest. 
    - also must be for only one stim of interest.
    - order must match datsegs (and same length). this is tru
    by default 
    - sort_by says whether to order parses by prior or likelihood score
    - priorver is diff combinations of DC and Planner model
    - likeli ver currently is always the aggregated string edit distances, so 
    not a variable yet. 
    - nplot is which parses to plot. e.g., [0,1,-2,-1] plots the 2 best and wrost.
    """
    if sort_by=="prior":
        # sort in descending order
        distances_flat_this = sorted(distances_flat_this, key = lambda x: -x[priorver])
    elif sort_by=="likeli":
        # sort ascending distance (which is descending likeli)
        distances_flat_this = sorted(distances_flat_this, key = lambda x: x["dist"])
        
    assert len(nplot)<len(distances_flat_this), "can't plot this many since not enough parses abailable"
#     assert len(distances_flat_this)==len(datsegs), "they do not match..."
    
    # - plot parse sequence
    figs = []
    for n in nplot:
        D = distances_flat_this[n]
        idx = D["modelrend"]
        dist = D["dist"]
        stim = D["stim"]
        model = D["model"]
        prior = D[priorver]
        
        if False:
            dseg = datsegs[idx]
        else:
            dseg = extractDatSeg(datsegs[0], D["sequence_model"])
        
        # -- confirm that the datsegs sequence is identical to the sorted distances_flat
    #     print([d["codes_unique"] for d in dseg])
    #     print(D["sequence_model"])
    #     print([d["codes_unique"] for d in dseg]==D["sequence_model"])
    #     print([d["codes_unique"] for d in dseg][::-1]==D["sequence_model"])
        assert [d["codes_unique"] for d in dseg]==D["sequence_model"], "need to match the origianl datsegs object to the desired parse"
        dflat = dgseg.segmentedScores2NumpyStrokes([dseg])
        pp = priorver if sort_by=="prior" else sort_by
        f = dgutils.plotDrawingSteps(dflat[0], title=[f"stim\n{stim}", f"model\n{model}", 
                                                  f"prob using:\n {pp}", 
                                                  f"rank {n}",
                                                  f"prior {prior:.2}",
                                                 f"dist 2 hu\n{dist:.2}"]);
        figs.append(f)
    return figs
    
    
from drawgood.experiments.modelAnaly import stringDist
from drawgood.experiments.segmentation import codeUniqueFeatures

# ==== GET HUMAN-HUMAN DISTANCES
def getSeq(dseg, labelkind="codes_unique"):
    """ given one list of dicts - dseg, output
    a list that is the sequence represnettaiton desired (labelkind)"""
#     print(dseg[0])
#     print(labelkind)
#     print(dseg[0].keys())
    if labelkind=="col":
        return [codeUniqueFeatures(d["codes_unique"])[1] for d in dseg]
    else:
        return [d[labelkind] for d in dseg]
    
def _dist(dseg1, dseg2):
    """ dseg1 and 2 are list of dicts. just one
    sequence each"""
    distances = []
    for labelkind in ["codes_unique", "codes", "row", "col"]:
        distances.append(stringDist(getSeq(dseg1, labelkind), getSeq(dseg2, labelkind)))
    return np.mean(distances)

        
def extractHumanDatseg(DAT_all, task, human, getdatflat=False):
    df = dgutils.filterDat(DAT_all[0]["datflat_hu"], workerIDlist=[human], stimlist=[f"{task}.png"])
    if getdatflat:
        return df
    if len(df)==0:
        print(f"[dist] no data for {human} for {task} - skipping")
        return None
    elif len(df)>1:
        print("WHY?")
        print(f"[dist] {human} for {task}")
        assert False
    else:
        dseg = getSegmentation(df, unique_codes=True, dosplits=True)[0]
    return dseg

def extractHumanCond(human, workerlist):
    cond = list(set([w["condition"] for w in workerlist if w["workerID"]==human]))[0]
    return cond    


def plotLikeliPrior(likeli, prior):
    # -- if any of them are none then remove
    prior = np.array([p for l, p in zip(likeli, prior) if l!=None])
    likeli = np.array([l for l in likeli if l!=None])
    x = np.arange(len(prior))
    fig = plt.figure(figsize=(7,3))
    plt.subplot(2,1,2)
    plt.plot(x, likeli, '-k', label="likeli")
    plt.ylim([0,1])
    plt.legend()
    plt.subplot(2,1,1)
    plt.plot(x, prior, '-r', label="prior")
    plt.legend()
    
    # weighted average of likeli (weighted by prior)
    prior = prior/np.sum(prior)
    assert(len(prior)==len(likeli))
    return np.average(likeli, weights=prior), fig

def extractDatSeg(datsegs, code_sequence):
    """ given list of codes_unique (e./g, [C1, LL2, ...])
    and datseg (list of dicts, each dict being one stroke
    that corresponding to one unique code), pull out datseg 
    that matches your code sequence """
    
    dseg_out = []
    for c in code_sequence:
        dseg_out.append([d for d in datsegs if d["codes_unique"]==c][0])
    return dseg_out
    
   

######################### get HUMAN-HUMAN DISTANCSE
def getHumanHumanDists(cleanup=True):
    print("SHOULD modify code so that gets triangular matrix of human x human, to save memory. But then would have to modify downstream code, so nevermind.")
    def dist(DAT_all, task, human1, human2):
        # get datseg for each human on this task

        dsegs = []
        for hu in [human1, human2]:
            
            df = dgutils.filterDat(DAT_all[0]["datflat_hu"], workerIDlist=[hu], stimlist=[f"{task}.png"])
            if len(df)==0:
                print(f"[dist] no data for {hu} for {task} - skipping")
                return None
            elif len(df)>1:
                print("WHY?")
                print(f"[dist] {hu} for {task}")
                assert False
            else:
                dsegs.append(getSegmentation(df, unique_codes=True, dosplits=True)[0])
            
        # compute distances
            
        return _dist(dsegs[0], dsegs[1])

        
    # def checkBothHumansDidThisTask(DAT_all, task, human1, human2):
    #     """ self explanatory"""
    #     for hu in [human1, human2]:
    #         df1 = dgutils.filterDat(DAT_all[0]["datflat_hu"], workerIDlist=[hu], stimlist=[f"{task}.png"])
    #         df2 = dgutils.filterDat(DAT_all[1]["datflat_hu"], workerIDlist=[hu], stimlist=[f"{task}.png"])
    #         if len(df1)==0 or len(df2)==0:
    #             return False
    #     return True
        
        
    ##  TRY TO LOAD
    try:
        SAVEDIR = f"analysis/summaryfigs/acrossexpt/ecS12.10.test5_S13.10.test5-dg2.4_2.4/closer_analysis/notebook_comparing_human_model_parses"

        # ==
        fname = f"{SAVEDIR}/human_human_dist.pkl"
        import pickle
        with open(fname, "rb") as f:
            outdict = pickle.load(f)            
    except:
        # === RECALCUALTE
        # -- for each task, for each human get each other human's distance
        # task_list = set([d["stim"] for d in distances_flat])
        task_list = [d.name for d in DAT_all[0]["testtasks"]] # only does test tasks. (those have shared)
        human_list = set([d["human"] for d in distances_flat])
        outdict = []
        for task in task_list:
            print(task)
            # for i, human1 in enumerate(human_list):
            #     for human2 in human_list[i:]:
            for human1 in human_list:
                for human2 in human_list:
                    if human1!=human2:
                        d = dist(DAT_all, task, human1, human2)
                        outdict.append({
                            "task":task,
                            "human1":human1,
                            "human2":human2,
                            "dist":d})
        ##  save
        SAVEDIR = f"analysis/summaryfigs/acrossexpt/ecS12.10.test5_S13.10.test5-dg2.4_2.4/closer_analysis/notebook_comparing_human_model_parses"
        import os
        os.makedirs(SAVEDIR, exist_ok=True)

        # ==
        fname = f"{SAVEDIR}/human_human_dist.pkl"
        import pickle
        with open(fname, "wb") as f:
            pickle.dump(outdict, f)       
                
    if cleanup:
        outdict = [o for o in outdict if o["dist"]!=None] 
    return outdict


    

############## RUN
if __name__=="__main__":
    ## ===== PARAMS
    # stim = "S12_13_test_1"
    # stim = "S12_247"
    # # human = "A2VLTSW6CXIUMR"
    # # humans_to_plot = ["A2VLTSW6CXIUMR", "A2P53AN2M8IWFP"]
    # humans_to_plot = ["AI36LV7AATYWF", "AIK9IRPT4M848"]

    if False:
        STIMS = [
        "S13_182",
        "S12_13_test_9",
        "S12_13_test_9",
        "S12_13_test_7",
        "S12_13_test_4"
        ]
    else:
        STIMS = [] # if empty, then will do all

    if False:
        HUMANSTOPLOT = [
        ["A29I0O9V6N1CY", "A2P53AN2M8IWFP"],
        ["A2RLY2I4U06ETD", "A2BBDH8DZD77AU"],
        ["A2ZJAEL03VTZ8", "A2BBDH8DZD77AU"],
        ["A2JDYN6QM8M5UN" , "AMPMTF5IAAMK8"],
        ["A2RLY2I4U06ETD", "A1Y82LKWQQP90M"]
        ]
    else:
        HUMANSTOPLOT = [
        ["A2ZJAEL03VTZ8", "AIK9IRPT4M848"]
        ] # if only one entry, then will use this for all STIMS

    #########################################
    outdict = getHumanHumanDists()
    print("Loaded Human-Human distances")
    outdict = [o for o in outdict if o["dist"]!=None]

    if len(STIMS)==0:
        STIMS = sorted(list(set([o["task"] for o in outdict])))
    if len(HUMANSTOPLOT)!=len(STIMS):
        assert len(HUMANSTOPLOT)==1, "either have matching hum and stim, or if want to repeat humans, then only have one pair"
        HUMANSTOPLOT = [HUMANSTOPLOT[0] for _ in range(len(STIMS))]
    assert len(STIMS)==len(HUMANSTOPLOT)

    print("STIMS:")
    print(STIMS)
    print("HUMANS")
    print(HUMANSTOPLOT)
    
    ## ==== LOAD ALL AT ONCE INTO MEMEORY, NOT PIECEMEITL
    # 1) Load dreamcoder
    # ECTRAINlist = ["S12.10.test4", "S13.10.test4"]
    ECTRAINlist = ["S12.10.test5", "S13.10.test5"]
    modelkind_list = ["parse", "randomperm"]
    ver="aggregate"
    use_withplannerscore=True

    distances_flat, DAT_all, workerlist, SAVEDIR = loadMultDCHumanDistances(
        ECTRAINlist, modelkind_list, ver, use_withplannerscore)

    for stim, humans_to_plot in zip(STIMS, HUMANSTOPLOT):

        # ===== FOR A STIM, PULL OUT TOP PARSES MODELS
        assert len(humans_to_plot)==2, "should be opposite conditions also"

        figs_all = []

        for human_to_compare_model in humans_to_plot:

            # == plot human 
            df = dgutils.filterDat(DAT_all[0]["datflat_hu"], workerIDlist=[human_to_compare_model], stimlist=[f"{stim}.png"])
            assert len(df)==1, "should be one human on one task..."
            df=df[0]
            condition = df["condition"]
            fig = dgutils.plotDrawingSteps(df["trialstrokes"], title=[f"stim\n{stim}", f"human\n{human_to_compare_model}", 
                                                      f"cond {condition}"])
            figs_all.append(fig)

            # given a set of humans, get set of distances.
            for cond in [0, 1]:
                humanlist = set([w["workerID"] for w in workerlist if w["condition"]==cond])
                print("NOTE: need to change this once change otudict to be traingular")
                distvals = [o for o in outdict if o["human1"]==human_to_compare_model 
                            and o["human2"] in humanlist and o["task"]==stim]
                distvals = sorted(distvals, key=lambda x: x["dist"])
                distvals = [d["dist"] for d in distvals]
                # - sort by likeli
                likeli = np.array(distvals)
                prior = (1/len(likeli))*np.ones(likeli.size)
                l, fig = plotLikeliPrior(likeli, prior)
                plt.title(f"vs. humans, cond {cond}; l = {l:.2}")
                figs_all.append(fig)


            # ====== 2) Plot model
            # sort_by = "prior"
            # sort_by = "likeli"
            for model in ["S12.10.test5", "S12.10.test5_randomperm",
                         "S13.10.test5", "S13.10.test5_randomperm"]:

                for priorver in ["motor_prob", "full_S12.10.test5_prob", "full_S13.10.test5_prob", "equal_prob"]:
                    
                    if priorver=="full_S12.10.test5_prob" and model!="S12.10.test5_randomperm":
                        continue
                    elif priorver=="full_S13.10.test5_prob" and model!="S13.10.test5_randomperm":
                        continue

                    distances_flat_this = [d for d in distances_flat 
                                           if d["stim"]==stim and d["model"]==model and d["human"]==human_to_compare_model]
                    print(f"this many parses found: {len(distances_flat_this)}")

                    if priorver=="equal_prob":
                        # then assign same prob to each parse
                        prob = 1/len(distances_flat_this)
                        for d in distances_flat_this:
                            d["equal_prob"] = prob

                    # to map backwards from distances to a given datseg object
                    if "S12.10" in model:
                        i=0
                    elif "S13.10" in model:
                        i=1
                    datsegs = DATloadDatSeg(DAT_all[i], stim)

                    for sort_by in ["likeli", "prior"]:
                        fig = plotParses(distances_flat_this, datsegs, sort_by, priorver, nplot=[0, 1, 2, -1])
                        figs_all.extend(fig)

                    # === PLOT HISTROGRAM OF PRIOR SCORES
                    # sort distances
                    distances_flat_this_sorted = sorted(distances_flat_this, key= lambda x:x["dist"])
            #         distances_flat_this_sorted = sorted(distances_flat_this, key= lambda x:x[priorver])
                    likeli = [d["dist"] for d in distances_flat_this_sorted]
                    prior = [d[priorver] for d in distances_flat_this_sorted]
            #         likeli = np.array(likeli
                    prior = np.array(prior)/np.sum(prior)

                    l, fig = plotLikeliPrior(likeli, prior)
                    plt.title(f"mod {model}, prior {priorver}, weighted ave dist = {l:.2}")
                    figs_all.append(fig)


            ## = for a given human, rank all the other humans and plot
            # given a set of humans, get set of distances.
            conds_to_include = [0,1]
            # humanlist = set([w["workerID"] for w in workerlist if w["condition"] in conds_to_include])
            human_others = set([w["workerID"] for w in workerlist if w["workerID"]!=human_to_compare_model 
                                and w["condition"] in conds_to_include])

            outdict_this = [o for o in outdict if o["human1"]==human_to_compare_model and o["human2"] in human_others and o["task"]==stim]
            outdict_this = sorted(outdict_this, key=lambda x: x["dist"])

            # 1) get human datsegs
            df1 = extractHumanDatseg(DAT_all, stim, human_to_compare_model, getdatflat=True)[0]
            # print(df1["trialstrokes"])
            fig = dgutils.plotDrawingSteps(df1["trialstrokes"])
            figs_all.append(fig)
            plt.title(f"starting human-human plots; this: {human_to_compare_model},cond{extractHumanCond(human_to_compare_model, workerlist)}")


            for n in np.arange(0, len(outdict_this),2):
            #     dseg1 = extractHumanDatseg(DAT_all, stim, human_to_compare_model)
                hu2 = outdict_this[n]["human2"]
                df2 = extractHumanDatseg(DAT_all, stim, hu2, getdatflat=True)[0]
                dist = outdict_this[n]["dist"]
                cond = extractHumanCond(hu2,workerlist)
                fig = dgutils.plotDrawingSteps(df2["trialstrokes"])
                if cond==0:
                    col = 'b'
                else:
                    col='r'

                plt.title(f"other: {hu2}\ndist{dist:.2},cond{cond}", color=col)
                figs_all.append(fig)


        SAVEDIR = f"analysis/summaryfigs/acrossexpt/ecS12.10.test5_S13.10.test5-dg2.4_2.4/closer_analysis/notebook_comparing_human_model_parses/allstim"
        fname = f"{SAVEDIR}/{stim}_h1-{humans_to_plot[0]}_h2-{humans_to_plot[1]}.pdf"

        # === save all figs
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        with PdfPages(fname) as pdf:
            # As many times as you like, create a figure fig and save it:
            for fig in figs_all:
                if not fig is None:
                    pdf.savefig(fig)

        # pdf = matplotlib.backends.backend_pdf.PdfPages("/tmp/output.pdf")
        # for fig in range(1, len(figs_all)): ## will open an empty extra figure :(
        #     pdf.savefig( fig )
        # pdf.close()


        ############ SAVE
        ##  save
        import os
        os.makedirs(SAVEDIR, exist_ok=True)

        # ==
        fname = f"{SAVEDIR}/human_human_dist.pkl"
        import pickle
        with open(fname, "wb") as f:
            pickle.dump(outdict, f)                
