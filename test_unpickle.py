
from ec import *
from regexes import *
import dill
import numpy as np

import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines


#path = "experimentOutputs/regex_activation=sigmoid_aic=1.0_arity=3_ET=5_helmholtzBatch=5000_HR=0.5_it=2_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_steps=100_L=1.0_K=5_useNewRecognitionModel=False_rec=True_feat=LearnedFeatureExtractor.pickle"

#path = "experimentOutputs/regex_aic=1.0_arity=3_ET=2_helmholtzBatch=5000_it=2_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_L=1.0_K=5_useNewRecognitionModel=False_rec=False.pickle"

#path = "experimentOutputs/regex_activation=sigmoid_aic=1.0_arity=3_ET=10_helmholtzBatch=5000_HR=0.5_it=2_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_steps=100_L=1.0_K=5_useNewRecognitionModel=False_rec=True_feat=LearnedFeatureExtractor.pickle"

path = "experimentOutputs/list_activation=sigmoid_aic=1.0_arity=3_ET=5_helmholtzBatch=5000_HR=0.5_it=1_likelihoodModel=all-or-nothing_MF=5_baseline=False_pc=10.0_steps=250_L=1.0_K=5_useNewRecognitionModel=False_rec=True_feat=LearnedFeatureExtractor.pickle"

path = "experimentOutputs/regex_aic=1.0_arity=3_ET=3_helmholtzBatch=5000_it=1_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_L=1.0_K=5_useNewRecognitionModel=True_rec=False_feat=MyJSONFeatureExtractor.pickle"

file3="experimentOutputs/regex_activation=sigmoid_aic=1.0_arity=3_ET=3_helmholtzBatch=5000_HR=0.5_it=10_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_steps=250_L=1.0_K=5_useNewRecognitionModel=False_rec=True_feat=LearnedFeatureExtractor.pickle"

with open(file3, 'rb') as handle:
    result = dill.load(handle)

print(result)
