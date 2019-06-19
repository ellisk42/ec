try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import pregex as pre

from dreamcoder.utilities import *
from dreamcoder.domains.regex.groundtruthRegexes import *
from dreamcoder.program import Abstraction, Application

from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
from dreamcoder.domains.regex.regexPrimitives import PRC

import torch
from torch.nn import Parameter
from torch import optim

import torch.nn.functional as F
import string

AUTODIFF_ITER = 100
autodiff = True
SMOOTHING = None #.1


#used in rendering
def map_fun(diff_lookup):
    #step one: normalize 

    def fun(x):
        if type(x) is pre.CharacterClass and x.values == pre.d.values: #string.digits
            char = string.digits
            ps = F.softmax( diff_lookup['d'] )
            x2 = pre.CharacterClass(char, ps=ps, normalised=True)
            #x2 = x._replace(ps=ps, normalised=True)
            if SMOOTHING:
                ps = ps + SMOOTHING
                ps = ps / torch.sum(ps)
            return x2
        elif type(x) is pre.CharacterClass and x.values == pre.u.values:
            char = string.ascii_uppercase
            ps = F.softmax( diff_lookup['u'] )
            if SMOOTHING:
                ps = ps + SMOOTHING
                ps = ps / torch.sum(ps)
            x2 = pre.CharacterClass(char, ps=ps, normalised=True)
            return x2.map( fun )
        elif type(x) is pre.CharacterClass and x.values == pre.l.values:
            char = string.ascii_lowercase
            ps = F.softmax( diff_lookup['l'] )
            if SMOOTHING:
                ps = ps + SMOOTHING
                ps = ps / torch.sum(ps)
            x2 = pre.CharacterClass(char, ps=ps, normalised=True)
            return x2.map( fun )
        elif type(x) is pre.CharacterClass and x.values == pre.dot.values:
            char = string.printable[:-4]
            ps = F.softmax( diff_lookup['.'] )
            if SMOOTHING:
                ps = ps + SMOOTHING
                ps = ps / torch.sum(ps)
            x2 = pre.CharacterClass(char, ps=ps, normalised=True)
            return x2.map( fun )
        elif type(x) is pre.KleeneStar:
            p = F.sigmoid(diff_lookup['*'])
            return pre.KleeneStar(x.val, p=p).map( fun )
        elif type(x) is pre.CharacterClass:
            print("x=", x)
            print("x.name=", x.name)
            raise NotImplementedError()
        else:
            #print("x", x)
            return x.map(fun)
        #TODO 

    return fun

def create_params():
    diff_lookup = {
    'd': Parameter(torch.zeros( len(string.digits)) ),
    'u': Parameter(torch.zeros(len(string.ascii_uppercase) ) ),
    'l': Parameter(torch.zeros(len(string.ascii_lowercase) ) ),
    '.': Parameter(torch.zeros(len(string.printable[:-4]) ) ),
    '*': Parameter(torch.zeros(1) ), #geometric
    }

    return diff_lookup.values(), diff_lookup

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

checkpoint_file = "experimentOutputs/regex/2019-03-04T19:30:01.252339/regex_aic=1.0_arity=3_BO=True_CO=False_ES=1_ET=720_HR=0.5_it=25_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
#strConst, no training cutoff
#checkpoint_file = "/om2/user/ellisk/ec/experimentOutputs/regex/2019-02-26T14:49:43.594106/regex_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=6_mask=True_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=40_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"

import sys
if len(sys.argv) > 1:
    checkpoint_file = sys.argv[1]
REGEXCACHINGTABLE = {}
def testingRegexLikelihood(task, program):
    global REGEXCACHINGTABLE
    from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
    import pregex as pre
    
    testing = regexHeldOutExamples(task)
    program = program.visit(ConstantVisitor(task.str_const))
    r = program.evaluate([])(pre.String(""))

    ll = 0.
    for _,s in testing:
        if (r,s) not in REGEXCACHINGTABLE:
            REGEXCACHINGTABLE[(r,s)] = r.match(s)
        ll += REGEXCACHINGTABLE[(r,s)]
    return ll

def verbatim(s):
    delimiters = "|.@#"
    for d in delimiters:
        if d not in s:
            return f"\\verb{d}{s}{d}"
    assert False, f"could not turn into verbatim {s}"

def verbatimTable(strings, columns=1):
    if columns == 1:
        strings = [verbatim(s) if s is not None else "\\\\midrule " for s in strings]
    else:
        strings = [" & ".join(verbatim(s) for s in ss)  for ss in strings]
    return """
\\begin{tabular}{%s}
    %s
\\end{tabular}
"""%("l"*columns, "\\\\\n".join(strings))

def prettyRegex(preg):
    return preg.str().replace('[ABCDEFGHIJKLMNOPQRSTUVWXYZ]','\\u').replace("[0123456789]","\\d").replace("[abcdefghijklmnopqrstuvwxyz]","\\l").replace("[0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t]",".")
    
if __name__ == "__main__":
    
    print("started:", flush=True)

    with open(checkpoint_file, 'rb') as file:
        checkpoint = pickle.load(file)



    tasks = checkpoint.testSearchTime.keys() #recognitionTaskMetrics.keys()
    from dreamcoder.likelihoodModel import add_cutoff_values
    tasks = add_cutoff_values(tasks, "gt") #could be "unigram" or "bigram"



    print("TESTING ONLY:")
    #print loop?
    posteriorHits = 0
    likelihoodHits = 0
    posteriorHits_test = 0
    likelihoodHits_test = 0
    marginalHits = 0
    marginalHits_test = 0

    totalTasks = 0
    for task in tasks:
        #if task.name in badRegexTasks: continue

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

        eprint(verbatimTable(["".join(example[1]) for example in task.examples] + [None] + \
                             [gt_preg,None] + \
                             [example[1] for example in testingExamples]))
        eprint("&")

        gt_preg = pre.create(gt_preg)
        def examineProgram(entry):
            global preg
            global diff_lookup
            program = entry.program
            ll = entry.logLikelihood
            program = program.visit(ConstantVisitor(task.str_const))
            print(program)
            preg = program.evaluate([])(pre.String(""))
            Pstring = prettyRegex(preg)

            if autodiff:
                params, diff_lookup = create_params()  #TODO

                opt = optim.Adam(params, lr=0.1)
                for i in range(AUTODIFF_ITER):
                    opt.zero_grad()
                    #normalize_diff_lookup(diff_lookup) #todo, using softmax and such
                    preg = preg.map(map_fun(diff_lookup)) #preg.map(fun)
                    score = sum( preg.match( "".join(example[1])) for example in task.examples)
                    (-score).backward(retain_graph=True)
                    opt.step()
                    # if i%10==0:
                    #     print(i, params)

                #post-optimization score
                #normalize_diff_lookup(diff_lookup) #todo, using softmax and such
                preg = preg.map(map_fun(diff_lookup))
                #print("parameters:")
                ll = sum( preg.match( "".join(example[1]))  for example in task.examples )

            testing_likelihood = sum(preg.match(testingString)
                         for _,testingString in testingExamples)
            ground_truth_testing = sum(gt_preg.match(testingString)
                         for _,testingString in testingExamples)
            
            eprint("&")
            eprint(verbatimTable([Pstring] + [preg.sample() for i in range(5)]))
            print("\t", Pstring)
            print("\t", "samples:")
            print("\t", [preg.sample() for i in range(5)])
            entry.trainHit = ll >= task.gt
            if ll >= task.gt:
                print(f"\t HIT (train), Ground truth: {task.gt}, found ll: {ll}")
            else:
                print(f"\t MISS (train), Ground truth: {task.gt}, found ll: {ll}")
            entry.testHit = testing_likelihood >= ground_truth_testing
            if testing_likelihood >= ground_truth_testing:
                print(f"\t HIT (test), Ground truth: {ground_truth_testing}, found ll: {testing_likelihood}")
            else:
                print(f"\t MISS (test), Ground truth: {ground_truth_testing}, found ll: {testing_likelihood}")


        print("\t", "best Posterior:")
        entry = max(frontier.entries, key=lambda e: e.logLikelihood + e.logPrior)
        examineProgram(entry)
        posteriorHits += int(entry.trainHit)
        posteriorHits_test += int(entry.testHit)

        print("\t", "best Likelihood:")
        entry = max(frontier.entries, key=lambda e: e.logLikelihood)
        examineProgram(entry)
        likelihoodHits += int(entry.trainHit)
        likelihoodHits_test += int(entry.testHit)
        print()

        print("\t","Posterior predictive samples...")
        programSamples = [frontier.sample().program for _ in range(5)]
        programSamples = [p.visit(ConstantVisitor(task.str_const)).evaluate([])(pre.String(""))
                          for p in programSamples ]
        stringSamples = [p.sample()
                         for p in programSamples ]
        programSamples = [prettyRegex(p)
                          for p in programSamples ]
        eprint("&")
        eprint(verbatimTable(list(zip(stringSamples, programSamples)),columns=2))
        
        eprint("\\\\")
        
        eprint()

        posterior = [(e.logPosterior, e.program.visit(ConstantVisitor(task.str_const)).evaluate([])(pre.String("")))
                     for e in frontier.normalize() ]
        testingExamples = [te for _,te in testingExamples]
        print("testingExamples",testingExamples)
        testingLikelihood = lse([lp + sum(r.match(te) for te in testingExamples)
                                 for lp,r in posterior])
        trainingExamples = ["".join(example[1]) for example in task.examples]
        print("trainingExamples",trainingExamples)
        trainingLikelihood = lse([lp + sum(r.match(te) for te in trainingExamples)
                                 for lp,r in posterior])

        if testingLikelihood >= sum(gt_preg.match(te) for te in testingExamples):
            marginalHits_test += 1
        if trainingLikelihood >= sum(gt_preg.match(te) for te in trainingExamples):
            marginalHits += 1
        

        
        

    print(f"Best posteriorc hits training task {posteriorHits}/{totalTasks} = {posteriorHits/totalTasks}")
    print(f"Best likelihood hits training task {likelihoodHits}/{totalTasks} = {likelihoodHits/totalTasks}")

    print(f"Best posteriorc hits testing task {posteriorHits_test}/{totalTasks} = {posteriorHits_test/totalTasks}")
    print(f"Best likelihood hits testing task {likelihoodHits_test}/{totalTasks} = {likelihoodHits_test/totalTasks}")

    print(f"Posterior predictive hits training task {marginalHits}/{totalTasks} = {marginalHits/totalTasks}")
    print(f"Posterior predictive hits testing task {marginalHits_test}/{totalTasks} = {marginalHits_test/totalTasks}")

