try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module


from json.decoder import JSONDecodeError
from syntax_robustfill import SyntaxCheckingRobustFill
from image_robustfill import Image_RobustFill
import time

from dreamcoder.grammar import Grammar
from dreamcoder.domains.text.makeTextTasks import *
import dreamcoder.domains.text.main as Text
from dreamcoder.task import Task
from dreamcoder.frontier import Frontier,FrontierEntry
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.type import tpregex

from dreamcoder.domains.arithmetic.arithmeticPrimitives import real, real_division, real_addition, real_multiplication

from rational import RandomParameterization, FeatureExtractor, drawFunction

import rational
#text:
import dreamcoder.domains.text.textPrimitives as text_primitives
from dreamcoder.domains.list.listPrimitives import bootstrapTarget,bootstrapTarget_extra
from string import printable

import torch

# from dreamcoder.domains.list.listPrimitives import josh_primitives
# from dreamcoder.domains.list.makeListTasks import joshTasks

from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS
from dreamcoder.domains.list.main import retrieveJSONTasks
import dreamcoder.domains.list.main as List

import dreamcoder.domains.tower.main as Tower
import dreamcoder.domains.tower.makeTowerTasks
from dreamcoder.domains.tower.towerPrimitives import new_primitives, ttower
import dreamcoder.domains.logo.main as LOGO
import dreamcoder.domains.logo.logoPrimitives
from dreamcoder.domains.logo.logoPrimitives import turtle
from dreamcoder.domains.logo.makeLogoTasks import drawLogo

from pregex import pregex
import pregex as pre
from dreamcoder.domains.regex.makeRegexTasks import makeOldTasks, makeLongTasks, makeShortTasks, makeWordTasks, makeNumberTasks, makeHandPickedTasks, makeNewTasks, makeNewNumberTasks #, add_string_constants
from dreamcoder.likelihoodModel import add_string_constants
from dreamcoder.domains.regex.regexPrimitives import reducedConcatPrimitives
import dreamcoder.domains.regex.main as Regex
Regex.ConstantInstantiateVisitor.SINGLE = Regex.ConstantInstantiateVisitor()
Text.ConstantInstantiateVisitor.SINGLE = \
        Text.ConstantInstantiateVisitor(list(map(list, list({tuple([c for c in s])
                                                        for t in loadPBETasks("PBE_Strings_Track")[0] + dreamcoder.domains.text.makeTextTasks.makeTasks()
                                                        for s in t.stringConstants}))))

#from scipy.misc import imresize # deprecated
from PIL import Image
def imresize(data, size):
    return np.array(Image.fromarray(data).resize(size=size, resample=Image.BICUBIC).getdata()).reshape(size)
import numpy as np

from dreamcoder.utilities import ParseFailure,lse
#import other stuff

extras = ['(', ')', 'lambda'] + ['$'+str(i) for i in range(10)]




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

#print(stringify("(foo (bar)) (foo fooo)"))



def getDatum():
    while True:
        tsk = random.choice(tasks)
        tp = tsk.request
        p = g.sample(tp, maximumDepth=6)
        task = fe.taskOfProgram(p, tp)
        if task is None:
            #print("no taskkkk")
            continue
        ex = makeExamples(task)
        if ex is None: continue
        
        return ex, stringify(str(p))

def makeExamples(task):
    if arguments.domain == 'tower':
        i = task.getImage().transpose(2,0,1)
        return i

    if arguments.domain == "regex":
        return [([],y)
                for xs,y in task.examples ]

    if arguments.domain == 'rational':
        i = imresize(task.features,(256,256))
        i = np.array([i]*3)
        return i
    if arguments.domain == 'logo':
        i = np.array([float(xx)/256. for xx in task.highresolution])        
        i = i.reshape((128,128))
        i = imresize(i,(256,256))
        i = np.array([i]*3)
        return i
    
    if hasattr(fe,'tokenize'):
        examples = []
        tokens = fe.tokenize(task.examples)
        if tokens is None: return None
        for xs,y in tokens:
            i = []
            for x in xs:
                i.extend(x)
                i.append('EOE')
            examples.append((i,y))
        return examples

def test_task(m, task, timeout):
    start = time.time()
    failed_cands = set()

    print(task.examples)

    frontier = []

    sampleFrequency = {}
    
    while time.time() - start < timeout:
        query = makeExamples(task)
        #import pdb; pdb.set_trace()
        candidates = m.sample([query]*BATCHSIZE) #i think this works
        #print('len failed', len(failed_cands))
        for cand in candidates:
            try:
                p = Program.parse(" ".join(cand))
            except ParseFailure: continue
            except IndexError: continue
            except AssertionError: continue
            if p not in failed_cands:
                if "STRING" in str(p):
                    assert arguments.domain == 'text'
                    if len(task.stringConstants) == 0: ll = float('-inf')
                    else:
                        ci = Text.ConstantInstantiateVisitor([[cc for cc in sc ]
                                                              for sc in task.stringConstants],
                                                             sample=False)
                        ll = min(task.logLikelihood(pp, timeout=0.1 if arguments.domain != 'rational' else None)
                                 for pp in p.visit(ci))
                if arguments.domain == 'regex':
                    # regex is handled specially
                    # we just collect all of the candidates and then marginalize over them
                    # but we have to make sure that each candidate is well typed and well formed
                    ll = float('-inf')
                    if not p.canHaveType(task.request): p = None
                    else:
                        from examineFrontier import ConstantVisitor
                        p = p.visit(ConstantVisitor(task.str_const))
                        try:
                            regex = p.evaluate([])(pre.String(""))
                            if arguments.sampleLikelihood:
                                sampleFrequency[regex] = 1 + sampleFrequency.get(regex)
                                p = None
                            else:
                                dataLikelihood = sum(regex.match("".join(y))
                                                     for _,y in task.examples )
                                logPrior = g.logLikelihood(task.request,p)
                                frontier.append(FrontierEntry(p,
                                                              logPrior=logPrior,
                                                              logLikelihood=dataLikelihood))
                            #print("sampled program",p,
                            #      "which translates into regex",regex,
                            #      "and which assigns the following likelihood to the test data",
                            #      dataLikelihood,
                            #      "and which has prior probability",logPrior)
                        except: p = None
                            
                elif arguments.domain != 'logo':
                    ll = task.logLikelihood(p, timeout=0.1 if arguments.domain != 'rational' else None)
                else:
                    try:
                        yh = drawLogo(p, timeout=1., resolution=28)
                        if isinstance(yh,list) and list(map(int,yh)) == task.examples[0][1]:
                            ll = 0.
                        else:
                            ll = float('-inf')
                        #print("no warning, we are cool.jpeg")
                    except JSONDecodeError:
                        eprint("WARNING: Could not decode json. If this occurs occasionally it might be because the neural network is producing invalid code. Otherwise, if this occurs frequently, then this is a bug.")
                        ll = float('-inf')
                        
                #print(ll)
                if ll > float('-inf'):
                    #print(p)
                    #print(task.name)
                    return True

                elif p is not None: failed_cands.add(p)

    if arguments.domain != 'regex':
        return False

    from examineFrontier import testingRegexLikelihood
    if arguments.sampleLikelihood:
        return lse([ math.log(frequency) + testingRegexLikelihood(task, regex)
                     for regex, frequency in sampleFrequency.items() ])
    # calculate that thing that we have to for regex
    frontier = Frontier(frontier,
                        task)
    from graphs import addStupidRegex
    frontier = addStupidRegex(frontier,g)
    print("for this task I think that the following is the map estimate:\n",
          frontier.topK(1))
    if arguments.taskLikelihood:
        return lse([ e.logPrior + e.logLikelihood
                     for e in frontier ])
    return lse([ e.logPosterior + testingRegexLikelihood(task, e.program)
                 for e in frontier ])



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--domain",'-d',default="text")
    parser.add_argument("--taskLikelihood",default=False,action='store_true')
    parser.add_argument("--sampleLikelihood",default=False,action='store_true')
    parser.add_argument("--test", type=str, default=False)
    parser.add_argument("--timeout", type=float, default=600)
    arguments = parser.parse_args()

    if arguments.domain == "text":
        tasks = makeTasks()
        g = Grammar.uniform(text_primitives.primitives + [p for p in bootstrapTarget()])
        input_vocabularies = [list(printable[:-4]) + ['EOE'], list(printable[:-4])]
        test = loadPBETasks("PBE_Strings_Track")[0]
        fe = Text.LearnedFeatureExtractor(tasks=tasks,
                                          testingTasks=test)

        BATCHSIZE = 16
        
    elif arguments.domain == "regex":
        g = Grammar.uniform(reducedConcatPrimitives(),
                            continuationType=tpregex)
        tasks = makeNewTasks()
        fe = Regex.LearnedFeatureExtractor(tasks)
        
        eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
        seed = 42 # previously this was hardcoded and never changed
        maxTasks = 256
        random.seed(seed)
        random.shuffle(tasks)
        maxTasks = 256
        del tasks[maxTasks:]    
        test, _ = testTrainSplit(tasks, 0.5)
        test = add_string_constants(test)
        
        input_vocabularies = [["dummy"], list(printable) + ["LIST_END","LIST_START"]]
        BATCHSIZE = 64
    elif arguments.domain == "tower":
        g = Grammar.uniform(new_primitives, continuationType=ttower)
        tasks = dreamcoder.domains.tower.makeTowerTasks.makeSupervisedTasks()
        test, _ = testTrainSplit(tasks, 0.5)
        fe = Tower.TowerCNN([])
        BATCHSIZE = 64
    elif arguments.domain == "logo":
        g = Grammar.uniform(dreamcoder.domains.logo.logoPrimitives.primitives,
                            continuationType=turtle)
        tasks = dreamcoder.domains.logo.makeLogoTasks.makeTasks(['all'],proto=False)
        test, _ = testTrainSplit(tasks, 0.5)
        fe = LOGO.LogoFeatureCNN([])
        BATCHSIZE = 64
    elif arguments.domain == "rational":
        tasks = rational.makeTasks()
        for t in tasks:
            t.features = drawFunction(200,10.,t.f)
            delattr(t, 'f')
        test, _ = testTrainSplit(tasks, 100)
        random.seed(42)
        random.shuffle(test)
        test = test[:100]
        g = Grammar.uniform([real, real_division, real_addition, real_multiplication])
        fe = FeatureExtractor([])
        BATCHSIZE = 64
    elif arguments.domain == "list":
        BATCHSIZE = 16
        tasks = retrieveJSONTasks("data/list_tasks.json") + sortBootstrap()
        tasks.extend([
            Task("remove empty lists",
                 arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
                 [((ls,), list(filter(lambda l: len(l) > 0, ls)))
                  for _ in range(15)
                  for ls in [[[random.random() < 0.5 for _ in range(random.randint(0, 3))]
                              for _ in range(4)]]]),
            Task("keep squares",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: int(math.sqrt(x)) ** 2 == x,
                                      xs)))
                  for _ in range(15)
                  for xs in [[random.choice([0, 1, 4, 9, 16, 25])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
            Task("keep primes",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: x in {2, 3, 5, 7, 11, 13, 17,
                                                      19, 23, 29, 31, 37}, xs)))
                  for _ in range(15)
                  for xs in [[random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
        ])
        for i in range(4):
            tasks.extend([
                Task("keep eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x == i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x != i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("keep gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: not x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]])
            ])
            fe = List.LearnedFeatureExtractor(tasks)

        def isIdentityTask(t):
            return all( len(xs) == 1 and xs[0] == y for xs, y in t.examples  )
        eprint("Removed", sum(isIdentityTask(t) for t in tasks), "tasks that were just the identity function")
        tasks = [t for t in tasks if not isIdentityTask(t) ]
        test, train = testTrainSplit(tasks, .5)
        test = [t for t in test
                if t.name not in EASYLISTTASKS]
        g = Grammar.uniform(bootstrapTarget_extra())
        input_vocabularies = [fe.lexicon + ['EOE'], fe.lexicon]

    elif arguments.domain == "josh":
        tasks = joshTasks()
        fe = List.LearnedFeatureExtractor(tasks)
        # test, train = testTrainSplit(tasks, .5)
        # test = [t for t in test
        #         if t.name not in EASYLISTTASKS]
        g = Grammar.uniform(josh_primitives())
        input_vocabularies = [fe.lexicon + ['EOE'], fe.lexicon]

    if not arguments.test:

        target_vocabulary = [str(p) for p in g.primitives] + extras
        
        if arguments.domain in ['tower', 'rational', 'logo']:
            m = Image_RobustFill(target_vocabulary=target_vocabulary)
        else:
            m = SyntaxCheckingRobustFill(input_vocabularies=input_vocabularies,
                                    target_vocabulary=target_vocabulary)


        if torch.cuda.is_available():
            print("CUDAfying net...")
            m.cuda()
        else:
            print("Not using CUDA")


        start=time.time()
        max_n_iterations = 10000000000
        #batch = [getDatum() for _ in range(BATCHSIZE)]
        for i in range(max_n_iterations):
            batch = [getDatum() for _ in range(BATCHSIZE)]
            inputs, targets = zip(*batch)
            score = m.optimiser_step(inputs, targets)

            if i%10==0: print(f"Iteration {i}/{max_n_iterations}, Score {score}, ({(time.time()-start)/(i+1)} seconds per iteration)", flush=True) 

            if i%100==0:
                PATH = f"{arguments.domain}_robustfill_baseline_6.p"
                torch.save(m, PATH)
                print("saved at", PATH)

    else:
        path = arguments.test

        m = torch.load(path)
        m.max_length=50
        print("domain: ", arguments.domain)
        print('testing model:')
        print(path)

        #assumes `test` is a list of the testing tasks

        if arguments.domain == 'regex':
            from dreamcoder.domains.regex.groundtruthRegexes import badRegexTasks
            test = [t for t in test
                    if t.name not in badRegexTasks]
        
        n_tasks = len(test)
        print(n_tasks, "testing tasks")
        n_hits = 0
        total_likelihood = 0
        for i, task in enumerate(test):
            hit = test_task(m, task, arguments.timeout)
            if arguments.domain == 'regex':
                total_likelihood += hit
            else:
                if hit: n_hits += 1
            print("for task ",i, ", hit=", hit, flush=True)
            

        if arguments.domain == 'regex':
            from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
            totalCharacters = sum( len(s)
                                   for t in test 
                                   for _,s in regexHeldOutExamples(t))
            if arguments.taskLikelihood:
                print("average marginal likelihood of held out task (normalized per character)",
                      total_likelihood/totalCharacters)
            else:
                print("average likelihood of held out examples (normalized per character)",
                      total_likelihood/totalCharacters)
        else:
            print("final score:")
            print(n_hits/float(n_tasks))







