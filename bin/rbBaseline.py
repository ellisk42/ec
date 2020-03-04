try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module


from syntax_robustfill import SyntaxCheckingRobustFill
#from image_robustfill import Image_RobustFill
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

from dreamcoder.domains.list.listPrimitives import josh_primitives
from dreamcoder.domains.list.makeListTasks import joshTasks

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



def getDatum(n_ex):
    while True:
        tsk = random.choice(tasks)
        tp = tsk.request
        p = g.sample(tp, maximumDepth=6)
        task = fe.taskOfProgram(p, tp)
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
        tokens = fe.tokenize(task.examples)
        if tokens is None: return None
        for xs,y in tokens:
            i = []
            for x in xs:
                i.extend(x)
                i.append('EOE')
            examples.append((i,y))
        return examples

def test_task(m, task, i, timeout):
    start = time.time()
    #failed_cands = set()

    #print(task.examples)
    while time.time() - start < timeout:
        
        allExamples = task.examples
        for i_run in range(5):


            #print(f"i_run: {i_run}, i_N: 0, task ID: {i}, test input: {task.examples[0][0]}, test output: {task.examples[0][1]}, prediction: None, correct: False", flush=True )
            print(f"\"CSVc{task.name.split('_')[0]}\",{i_run},{task.name.split('_')[1]},{1},\"(lambda $0)\",0,0")
            
            task.examples = [allExamples[0]]
            for trial in range(1,11):
                #test_example = random.choice( [ex for ex in allExamples if ex not in task.examples])
                test_example = allExamples[trial]
                query = makeExamples(task)

                n_cands = -1
                t = time.time()
                hit = False
                while time.time() - t < timeout and not hit:
                    candidates = m.sample([query]*BATCHSIZE) #i think this works
                    for cand in candidates:
                        n_cands += 1
                        try:
                            p = Program.parse(" ".join(cand))
                            #print(p)
                        except ParseFailure: continue
                        except IndexError: continue
                        except AssertionError: continue

                        #samples.csv:
                        prettyP = str(p).replace("fix1","fix").replace("gt?",">").replace("-n99","-").replace("-n9","-").replace("+n99","+").replace("+n9","+").replace("car","head").replace("cdr","tail").replace("empty?","is_empty").replace("eq?","is_equal")
                        eprint(f"\"CSVc{task.name.split('_')[0]}\",{i_run},{task.name.split('_')[1]},{trial+1},\"{prettyP}\",{time.time() - t},{n_cands}")

                        #if p not in failed_cands:
                        ll = task.logLikelihood(p, timeout=1)
                        #print(ll)
                        if ll > float('-inf'): 
                            #print("found program:")
                            #print(p, flush=True)
                            #print(ll)
                            try:
                                prog = p.evaluate([])
                                #import pdb; pdb.set_trace()
                                pred_out = prog(test_example[0][0])
                            except AttributeError as e:
                                print(e)
                                continue
                            except IndexError:
                                continue
                            except:
                                continue
                            correct = pred_out == test_example[1]

                            #print(f"i_run: {i_run}, i_N: {i_N}, task ID: {i}, test input: {test_example[0]}, test output: {test_example[1]}, prediction: {pred_out}, correct: {correct}" , flush=True)

                            searchTime = time.time() - t
                            
                            print(f"\"CSVc{task.name.split('_')[0]}\",{i_run},{task.name.split('_')[1]},{trial},\"{prettyP}\",{searchTime},{n_cands}")
                            hit = True
                            break
                        
                if not hit:
                    print(f"\"CSVc{task.name.split('_')[0]}\",{i_run},{task.name.split('_')[1]},{trial+1},\"(lambda $0)\",0,0", flush=True)
                    #print(f"i_run: {i_run}, i_N: {i_N}, task ID: {i}, test input: {test_example[0]}, test output: {test_example[1]}, prediction: None, correct: False", flush=True )
                task.examples.append(test_example)
    return False



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--domain",'-d',default="josh")
    parser.add_argument("--test", type=str, default=False)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("-w", type=int)
    parser.add_argument("--type", type=str, default='9') #'9' or '99'
    parser.add_argument("--tasks", type=int, default=0)
    arguments = parser.parse_args()

    assert arguments.domain == "josh"
    assert arguments.w == 3

    BATCHSIZE = 16

    tasks = joshTasks(arguments.w)
    fe = List.LearnedFeatureExtractor(tasks)

    if not arguments.test:
        #TRAINING CODE:


        if arguments.type == '9':
            tasks = [t for t in tasks if int(int(t.name.split("_")[0]) < 81)]
            p = josh_primitives(arguments.w)[0]
            g = Grammar.uniform(p)

        elif arguments.type == '99':
            tasks = [t for t in tasks if int(int(t.name.split("_")[0]) >= 81)] 
            p = josh_primitives(arguments.w)[1]
            g = Grammar.uniform(p)

        else: assert False



        input_vocabularies = [fe.lexicon + ['EOE'], fe.lexicon]

        target_vocabulary = [str(p) for p in g.primitives] + extras
        
        m = SyntaxCheckingRobustFill(input_vocabularies=input_vocabularies,
                                    target_vocabulary=target_vocabulary)

        if torch.cuda.is_available():
            print("CUDAfying net...")
            m.cuda()
        else:
            print("Not using CUDA")


        start=time.time()
        max_n_iterations = 10000000000
        for i in range(max_n_iterations):
            n_ex = random.choice(range(1,11))
            batch = [getDatum(n_ex) for _ in range(BATCHSIZE)]
            inputs, targets = zip(*batch)
            try:
                score = m.optimiser_step(inputs, targets)
            except KeyError: continue

            if i%10==0: print(f"Iteration {i}/{max_n_iterations}, Score {score}, ({(time.time()-start)/(i+1)} seconds per iteration)", flush=True) 

            if i%100==0:
                PATH = f"{arguments.domain}_wave_{arguments.w}.p_g={arguments.type}"
                torch.save(m, PATH)
                print("saved at", PATH)

    else:
        path = arguments.test


        #TO batch, we are going to 

        m9 = torch.load(path+"_g=9", map_location=lambda storage, loc: storage)
        m99 = torch.load(path+"_g=99", map_location=lambda storage, loc: storage)
        m9.max_length=50
        m99.max_length=50
        print('testing model:')
        print(path)

        #assumes tasks are the testing tasks
        taskToModel = {t: m99 if (int(t.name.split("_")[0]) >= 81) else m9
                         for t in tasks }
        n_tasks = len(tasks)
        print(n_tasks, "tasks")
        n_hits = 0

        start = arguments.tasks

        for i in range(start, start + 10):
        #for i, task in enumerate(tasks):
            task = tasks[i]
            test_task(taskToModel[task], task, i, arguments.timeout)

            #print("for task ",i, ", hit=", hit)
            #if hit: n_hits += 1

        #print("final score:")
        #print(n_hits/float(n_tasks))







