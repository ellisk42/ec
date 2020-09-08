

#train robustfill baseline


#simpleEval.py
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *

from dreamcoder.domains.tower.towerPrimitives import *
import time
import torch
import dill

from dreamcoder.domains.rb.rbPrimitives import *

from dreamcoder.domains.rb.main import makeOldTasks, makeTasks





from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead, REPLPolicyHead
from dreamcoder.domains.tower.makeTowerTasks import makeNewMaxTasks
from dreamcoder.SMC import SMC

from dreamcoder.valueHead import RBPrefixValueHead
#"rbPolicyOnlyBigram_SRE=True.pickle"
from dreamcoder.frontier import Frontier, FrontierEntry

from syntax_robustfill import SyntaxCheckingRobustFill


prims = robustFillPrimitives()
g = Grammar.uniform(prims)

def stringify(line):
    lst = []
    string = ""
    for char in line+" ":
        if char == " ":
            if string != "":
                lst.append(string)
            string = ""
        elif char in '()':
            if string != "":
                lst.append(string)
            string = ""
            lst.append(char)
        else:
            string += char      
    return lst

def makeBatch(batchsize):
    
    return inps, tgts


def getDatum(n_ex):
    while True:
        tsk = random.choice(tasks)
        tp = tsk.request


        p, task = r.recognitionModel.featureExtractor.sampleHelmholtzTask(arrow(texpression, texpression))
        #p = g.sample(tp, maximumDepth=6)
        #task = fe.taskOfProgram(p, tp)

        if task is None:
            #print("no taskkkk")
            continue

        del task.examples[n_ex:]
        #print(len(task.examples))
        ex = makeExamples(task)
        if ex is None: continue
        return ex, stringify(str(p))

def makeExamples(task):    
    if hasattr(fe,'tokenize'):
        examples = []
        #print(task.examples)
        tokens = fe.tokenize(task.examples) #todo
        if tokens is None: return None
        for xs,y in tokens:
            i = []
            for x in xs:
                i.extend(x)
                i.append('EOE')
            examples.append((i,y))
        return examples


parser = argparse.ArgumentParser()
parser.add_argument('--num_pretrain_episodes', type=int, default=100000, help='number of episodes for training')
parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate', dest='adam_learning_rate')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
parser.add_argument('--max_length_eval', type=int, default=50, help='maximum generated sequence length when evaluating the network')
parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
parser.add_argument('--dropout_p', type=float, default=0.1, help=' dropout applied to embeddings and LSTMs')
parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
parser.add_argument('--episode_type', type=str, default='scan_simple_original', help='what type of episodes do we want')
parser.add_argument('--batchsize', type=int, default=None )
parser.add_argument('--type', type=str, default="miniscanRBbase")
parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
parser.add_argument('--save_path', type=str, default='robustfill_baseline0.p')
parser.add_argument('--parallel', type=int, default=None)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--save_old_freq', type=int, default=1000)
parser.add_argument('--positional', action='store_true')
args = parser.parse_args()



#sys.setrecursionlimit(50000)
graph = ""
ID = 'rb'
runType = "PolicyOnly" #"Policy"
#runType =""
model = "Bigram"
useREPLnet = False
path = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'

print(path)
with open(path, 'rb') as h:
    r = dill.load(h)

fe = r.recognitionModel.featureExtractor

extras = ['(', ')', 'lambda'] + ['$'+str(i) for i in range(10)]
input_vocabularies = [fe.lexicon + ['EOE'], fe.lexicon]
target_vocabulary = [str(p) for p in g.primitives] + extras

m = SyntaxCheckingRobustFill(input_vocabularies=input_vocabularies,
                            target_vocabulary=target_vocabulary)



t = time.time()
for i in range(1, max_iter):

    #inps, tgts = makeBatch(batchsize)

    batch = [getDatum(n_ex) for _ in range(BATCHSIZE)]
    inputs, targets = zip(*batch)

    score, syntax_score = m.optimiser_step(inputs,targets) #syntax or not, idk
    m.iter += 1

    print(f"total time: {time.time() - t}, total num ex processed: {(i+1)*batchsize}, avg time per ex: {(time.time() - t)/((i+1)*batchsize)}, score: {score}")

    if i%args.save_freq==0:
        torch.save(m, args.save_path)
        print('saved model')
    if i%args.save_old_freq==0:
        torch.save(m, args.save_path+str(m.iter))