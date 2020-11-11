#blended graph

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import dill

from dreamcoder.showTowerTasks import computeConfusionMatrixFromScores, graphPrecisionRecall

useRNNPolicy = True
with open(f"precisionRecallAllDataRNNPolicy={useRNNPolicy}MAX.p", 'rb') as h:
    saved = dill.load(h)
symbolicDataLst, _,rnnDataLst, _, _ = saved    


useRNNPolicy = False
with open(f"precisionRecallAllDataRNNPolicy={useRNNPolicy}MAX.p", 'rb') as h:
    saved = dill.load(h)
symbolicDataLst2, neuralDataLst,_, _, _ = saved    


path = 'plots/precisionRecallit3samp10iclrblended.png'
graphPrecisionRecall(symbolicDataLst, neuralDataLst, rnnDataLst, path, otherSymbolicDataLst=symbolicDataLst2,  nSamp=500)

