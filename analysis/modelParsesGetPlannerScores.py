"""
Takes the dreamcoder parses (already processed to datsegs) and appends planner scores.
- Requires first preprocessing dreamcoder parses so that datsegs exists.
- Can specify which planner params to use to get planner scores.
- saves back into the same datsegs folder.
- This is independent from model-human distance. (i.e., do not need to have already run that)
- NOTE (5/10/2020) updated so that it forces to use planner models that were trained on ONLY those variables. 
Previous versions had taken all pmodels and extracted just those variables. I believe this should have
enough sanity checks that it will fail unless it runs properly.
"""

import sys
sys.path.insert(0, "/Users/lucastian/Dropbox/CODE/Python/Tenenbaum/ec/")
sys.path.insert(0, "/om/user/lyt/ec")
sys.path.insert(0, "/home/lucast4/dc")
# print(sys.path)

from analysis.utils import *
from scipy.special import logsumexp
import numpy as np
from analysis.importDrawgood import dgutils

# 3) Load planner scores
def DATloadPlannerScore(DAT, stimname, planver):
    """ loads presaved planner score for this stimulus, from
    parsesGetPlannerScores"""
    fname = "{}/{}_planscores_{}.pickle".format(DAT["savedir_datsegs"], stimname, planver)
    from os import path
    if path.exists(fname):
        with open(fname, "rb") as f:
            scores = pickle.load(f)
        return scores
    else:
        return None

# from analysis.model
import sys
if sys.argv[1]=="both":
    ECTRAIN_list = ["S13.10.test4", "S12.10.test4"]
elif sys.argv[1]=="12":
    ECTRAIN_list = ["S12.10.test4"]
elif sys.argv[1]=="13":
    ECTRAIN_list = ["S13.10.test4"]
elif sys.argv[1]=="test5":
    ECTRAIN_list = ["S12.10.test5", "S13.10.test5"]

#ECTRAIN_list = ["S13.10.test4"]
modelver_list = ["parse", "randomperm"]
planver_list = ["motor", "motorplusVH", "full"]
distver ="aggregate"
convert_to_prob=True # across parses for this stim x human x DCmodel x modelver x planner

debug_allow_skip_if_dists_and_seg_dont_match=False

for ECTRAIN in ECTRAIN_list:

    planver_list_this = [f"{p}_{ECTRAIN}" for p in planver_list] + planver_list
    print(f"planverlist for this ectrain is: {planver_list_this}")

    DAT = loadCheckpoint(ECTRAIN, loadparse=False, loadbehavior=True)
    SDIR = "{}/modelhudist_withplannerscore".format(DAT["analysavedir"])
    import os
    os.makedirs(SDIR, exist_ok=True)
    print("Saving at : {}".format(SDIR))

    # get lists for this experiment
    stimlist = DATgetSolvedStim(DAT, onlyifhasdatflat=True)
    humanlist = [h["workerID"] for h in DATgetWorkerList(DAT)]

    for stim in stimlist:
        for human in humanlist:

            # 1) Load distances object (one object covers all modelvers)
            dists_allmodelvers = DATloadModelHuDist(DAT, stim, human, suffix=distver)
            if len(dists_allmodelvers)==0:
                print("skipping {}, {}, {}, since no human-model distance found (shouidle be OK, different trainins et?".format(ECTRAIN, stim, human))
                continue

            for modelver in modelver_list:
                # - iterate through each model version. they all share same planner
                if modelver=="parse":
                    stimname = stim
                    modelname=ECTRAIN
                else:
                    modelname = "{}_{}".format(ECTRAIN, modelver)
                    stimname = "{}_{}".format(stim, modelver)


                # 1) dists for this
                dists = [d for d in dists_allmodelvers if d["model"]==modelname]
                dists = sorted(dists, key=lambda x:x["modelrend"])
                assert dists[-1]["modelrend"]==len(dists)-1, "expect continuous from 0 to K..."
                
                # 2) Load datseg
                datsegs = DATloadDatSeg(DAT, stimname)

                # 3) load planner score
                for planver in planver_list_this:
                    print("Getting palnner score for: {} (model), {}, {}, {} (modelver), {} (plannerver)".format(ECTRAIN, stim, human, modelver, planver))

                    scores = DATloadPlannerScore(DAT, stimname, planver)
                    if scores is None:
                        print("skipping, since did not find previously saved planscores. this is not unexpectexed")
                        # then files doesnt exist
                        continue

                    if convert_to_prob:
                        scoressum = logsumexp(scores)
                        scores = np.exp([s-scoressum for s in scores])

                    # 4) sanity checks make sure they are aligned
                    # if len(dists)!=len(datsegs):
                    #     import pdb
                    #     pdb.set_trace()
                    # print(dists[0])
                    # print(len(dists))
                    # print(len(datsegs))
                    # print(len(scores))

                    assert len(dists)==len(datsegs)
                    assert len(datsegs)==len(scores)

                    
                    for dst, dseg, sc in zip(dists, datsegs, scores):
                        assert "_".join([d["codes_unique"] for d in dseg]) == "_".join(dst["sequence_model"]), "model parse sequence of tokens are not exaclt the same. need to be" 
                        # NOTE: no need to check scores. I know that scores is matched perfectly to datseg. 

                        # -- append to dist the planner score
                        if isinstance(sc, list):
                            assert len(sc)==1
                            sc = sc[0]
                        dst["{}_prob".format(planver)]=sc.item()
                # save
                import pickle
                fname = "{}/{}_{}_{}.pickle".format(SDIR, stimname, human, distver)
                with open(fname, "wb") as f:
                    pickle.dump(dists, f)



            
                        



                    
                    
                    
