try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module


from syntax_robustfill import SyntaxCheckingRobustFill
from image_robustfill import Image_RobustFill
import time

from dreamcoder.grammar import Grammar
from dreamcoder.domains.text.makeTextTasks import *
import dreamcoder.domains.text.main as Text 
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.type import tpregex

from dreamcoder.domains.arithmetic.arithmeticPrimitives import real, real_division, real_addition, real_multiplication

from rational import RandomParameterization, FeatureExtractor

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

from pregex import pregex
from dreamcoder.domains.regex.makeRegexTasks import makeOldTasks, makeLongTasks, makeShortTasks, makeWordTasks, makeNumberTasks, makeHandPickedTasks, makeNewTasks, makeNewNumberTasks
from dreamcoder.domains.regex.regexPrimitives import reducedConcatPrimitives
import dreamcoder.domains.regex.main as Regex
Regex.ConstantInstantiateVisitor.SINGLE = Regex.ConstantInstantiateVisitor()

from scipy.misc import imresize
import numpy as np

from dreamcoder.utilities import ParseFailure
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

tasks = makeTasks()



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
        return task.getImage().transpose(2,0,1)

    if arguments.domain == "regex":
        return [([],y)
                for xs,y in task.examples ]

    if arguments.domain == 'rational':
        return imresize(np.array([task.features]*3),(256,256)).transpose(2,0,1)
    if arguments.domain == 'logo':
        i = np.array([float(xx)/256. for xx in task.highresolution])
        i = i.reshape((128,128))
        return imresize(np.array([i]*3),(256,256)).transpose(2,0,1)
    
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
    while time.time() - start < timeout:
        query = makeExamples(task)
        #import pdb; pdb.set_trace()
        candidates = m.sample([query]*BATCHSIZE) #i think this works
        #print('len failed', len(failed_cands))
        for cand in candidates:
            try:
                p = Program.parse(" ".join(cand))
                #print(p)
            except ParseFailure: continue
            except IndexError: continue
            except AssertionError: continue
            if p not in failed_cands:
                ll = task.logLikelihood(p, timeout=10)
                #print(ll)
                if ll > float('-inf'): return True
                else: failed_cands.add(p)

    return False



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--domain",'-d',default="text")
    parser.add_argument("--test", type=str, default=False)
    parser.add_argument("--timeout", type=float, default=1200)
    arguments = parser.parse_args()

    if arguments.domain == "text":
        g = Grammar.uniform(text_primitives.primitives + [p for p in bootstrapTarget()])
        input_vocabularies = [list(printable[:-4]) + ['EOE'], list(printable[:-4])]
        fe = Text.LearnedFeatureExtractor(tasks=tasks,
                                          testingTasks=loadPBETasks("PBE_Strings_Track")[0])

        BATCHSIZE = 16
    elif arguments.domain == "regex":
        g = Grammar.uniform(reducedConcatPrimitives(),
                            continuationType=tpregex)
        tasks = makeNewTasks()
        fe = Regex.LearnedFeatureExtractor(tasks)
        input_vocabularies = [["dummy"], list(printable) + ["LIST_END","LIST_START"]]
        BATCHSIZE = 64
    elif arguments.domain == "tower":
        g = Grammar.uniform(new_primitives, continuationType=ttower)
        tasks = dreamcoder.domains.tower.makeTowerTasks.makeSupervisedTasks()
        fe = Tower.TowerCNN([])
        BATCHSIZE = 64
    elif arguments.domain == "logo":
        g = Grammar.uniform(dreamcoder.domains.logo.logoPrimitives.primitives,
                            continuationType=turtle)
        tasks = dreamcoder.domains.logo.makeLogoTasks.makeTasks(['all'],proto=False)
        fe = LOGO.LogoFeatureCNN([])
        BATCHSIZE = 64
    elif arguments.domain == "rational":
        tasks = rational.makeTasks()
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

        #assumes tasks are the testing tasks

        n_tasks = len(tasks)
        print(n_tasks, "tasks")
        n_hits = 0
        for i, task in enumerate(tasks):
            hit = test_task(m, task, arguments.timeout)
            print("for task ",i, ", hit=", hit)
            if hit: n_hits += 1

        print("final score:")
        print(n_hits/float(n_tasks))







