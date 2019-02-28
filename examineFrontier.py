#all the imports
import pregex as pre
import pickle

from regexPrimitives import *
from program import Abstraction, Application

class ConstantVisitor(object):
    def __init__(self, stringConst):
        self.const = stringConst

    def primitive(self, e):
        if e.name == "r_const":
            e.value = PRC(pre.String(self.const))
        return e

    def invented(self, e): return e.body.visit(self)

    def index(self, e): return e

    def application(self, e):
        return Application(e.f.visit(self), e.x.visit(self))

    def abstraction(self, e):
        return Abstraction(e.body.visit(self))

#checkpoint_file = "experimentOutputs/regex/2019-02-21T00:45:20.505815/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=10_HR=0.5_it=11_mask=True_MF=10_pc=30.0_RT=1_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
checkpoint_file = "experimentOutputs/regex/2019-02-21T21:43:37.181850/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=3_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"

checkpoint_file = "experimentOutputs/regex/2019-02-21T21:43:37.181850/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=5_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"

checkpoint_file = "experimentOutputs/regex/2019-02-21T21:47:52.176434/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=5_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"

checkpoint_file = "experimentOutputs/regex/2019-02-23T23:41:20.015912/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=6_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"

#strConst
checkpoint_file = "/om2/user/ellisk/ec/experimentOutputs/regex/2019-02-26T14:49:18.044767/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=6_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"

#strConst, no training cutoff
checkpoint_file = "/om2/user/ellisk/ec/experimentOutputs/regex/2019-02-26T14:49:43.594106/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=6_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"


print("started:", flush=True)

with open(checkpoint_file, 'rb') as file:
        checkpoint = pickle.load(file)



tasks = checkpoint.testSearchTime.keys() #recognitionTaskMetrics.keys()
from likelihoodModel import add_cutoff_values
tasks = add_cutoff_values(tasks, "gt") #could be "unigram" or "bigram"


print("TESTING ONLY:")
#print loop?

for task in tasks:
        try:
                frontier = checkpoint.recognitionTaskMetrics[task]['frontier']
        except KeyError:
                continue
        print(task.name)
        print("\t", ["".join(example[1]) for example in task.examples])
        def examineProgram(entry):
            program = entry.program
            ll = entry.logLikelihood
            program = program.visit(ConstantVisitor(task.str_const))
            print(program)
            preg = program.evaluate([])(pre.String(""))
            string = preg.str().replace('[ABCDEFGHIJKLMNOPQRSTUVWXYZ]','\\u').replace("[0123456789]","\\d").replace("[abcdefghijklmnopqrstuvwxyz]","\\l")
            print("\t", string)
            print("\t", "samples:")
            print("\t", [preg.sample() for i in range(5)])
            if ll > task.gt:
                    print(f"\t HIT, Ground truth: {task.gt}, found ll: {ll}")
            else:
                    print(f"\t MISS, Ground truth: {task.gt}, found ll: {ll}")

            
        print("\t", "best Posterior:")
        examineProgram(max(frontier.entries, key=lambda e: e.logLikelihood + e.logPrior))
        print("\t", "best Likelihood:")
        examineProgram(max(frontier.entries, key=lambda e: e.logLikelihood))
        print()
