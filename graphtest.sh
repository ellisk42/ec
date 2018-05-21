#python graphs.py "experimentOutputs/regex_aic=1.0_arity=3_ET=2_helmholtzBatch=5000_it=2_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_L=1.0_K=5_useNewRecognitionModel=False_rec=False.pickle" "testimg.png"


#this is a vanilla regex ec 1.0 with the unigram cutoff

file="experimentOutputs/regex_aic=1.0_arity=3_ET=3_helmholtzBatch=5000_it=1_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_L=1.0_K=5_useNewRecognitionModel=True_rec=False_feat=MyJSONFeatureExtractor.pickle"

file2="experimentOutputs/regex_aic=1.0_arity=3_ET=10_helmholtzBatch=5000_it=10_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_L=1.0_K=5_useNewRecognitionModel=False_rec=False.pickle"

file3="experimentOutputs/regex_activation=sigmoid_aic=1.0_arity=3_ET=3_helmholtzBatch=5000_HR=0.5_it=10_likelihoodModel=probabilistic_MF=5_baseline=False_pc=10.0_steps=250_L=1.0_K=5_useNewRecognitionModel=False_rec=True_feat=LearnedFeatureExtractor.pickle"

python graphs.py $file3 "testimg.png" 
