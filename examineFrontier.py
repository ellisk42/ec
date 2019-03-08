#all the imports
import pregex as pre
import pickle

from groundtruthRegexes import *
from regexPrimitives import *
from program import Abstraction, Application

from makeRegexTasks import regexHeldOutExamples
from regexPrimitives import PRC

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
#checkpoint_file = "/om2/user/ellisk/ec/experimentOutputs/regex/2019-02-26T14:49:43.594106/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=6_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"


print("started:", flush=True)

with open(checkpoint_file, 'rb') as file:
        checkpoint = pickle.load(file)



tasks = checkpoint.testSearchTime.keys() #recognitionTaskMetrics.keys()
from likelihoodModel import add_cutoff_values
tasks = add_cutoff_values(tasks, "gt") #could be "unigram" or "bigram"



print("TESTING ONLY:")
#print loop?
posteriorHits = 0
likelihoodHits = 0
posteriorHits_test = 0
likelihoodHits_test = 0

totalTasks = 0
for task in tasks:
        try:
                frontier = checkpoint.recognitionTaskMetrics[task]['frontier']
        except KeyError:
                continue
        print(task.name)
        totalTasks += 1
        print("\tTRAIN\t", ["".join(example[1]) for example in task.examples])

        testingExamples = regexHeldOutExamples(task)
        print("\tTEST\t", [example[1] for example in testingExamples])

        gt_preg = gt_dict[int(task.name.split(" ")[-1])]
        print("\tHuman written regex:",gt_preg)
        gt_preg = pre.create(gt_preg)
        def examineProgram(entry):
            program = entry.program
            ll = entry.logLikelihood
            program = program.visit(ConstantVisitor(task.str_const))
            print(program)
            preg = program.evaluate([])(pre.String(""))
            testing_likelihood = sum(preg.match(testingString)
                                     for _,testingString in testingExamples)
            ground_truth_testing = sum(gt_preg.match(testingString)
                                     for _,testingString in testingExamples)
            string = preg.str().replace('[ABCDEFGHIJKLMNOPQRSTUVWXYZ]','\\u').replace("[0123456789]","\\d").replace("[abcdefghijklmnopqrstuvwxyz]","\\l")
            print("\t", string)
            print("\t", "samples:")
            print("\t", [preg.sample() for i in range(5)])
            if ll >= task.gt:
                    print(f"\t HIT (train), Ground truth: {task.gt}, found ll: {ll}")
            else:
                    print(f"\t MISS (train), Ground truth: {task.gt}, found ll: {ll}")
            if testing_likelihood >= ground_truth_testing:
                    print(f"\t HIT (test), Ground truth: {ground_truth_testing}, found ll: {testing_likelihood}")
            else:
                    print(f"\t MISS (test), Ground truth: {ground_truth_testing}, found ll: {testing_likelihood}")

            
        print("\t", "best Posterior:")
        examineProgram(max(frontier.entries, key=lambda e: e.logLikelihood + e.logPrior))
        if max(frontier.entries, key=lambda e: e.logLikelihood + e.logPrior).logLikelihood >= task.gt:
            posteriorHits += 1
        print("\t", "best Likelihood:")
        examineProgram(max(frontier.entries, key=lambda e: e.logLikelihood))
        if max(frontier.entries, key=lambda e: e.logLikelihood).logLikelihood >= task.gt:
            likelihoodHits += 1
        print()

print(f"Best posteriorc hits task {posteriorHits}/{totalTasks} = {posteriorHits/totalTasks}")
print(f"Best likelihood hits task {likelihoodHits}/{totalTasks} = {likelihoodHits/totalTasks}")

