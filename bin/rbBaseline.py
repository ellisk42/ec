try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module


from syntax_robustfill import SyntaxCheckingRobustFill
import time

from dreamcoder.grammar import Grammar
from dreamcoder.domains.text.makeTextTasks import *
from dreamcoder.domains.text.main import LearnedFeatureExtractor

#text:
from dreamcoder.domains.text.textPrimitives import primitives
from dreamcoder.domains.list.listPrimitives import bootstrapTarget
from string import printable


BATCHSIZE = 32
#import other stuff
input_vocabularies = [list(printable[:-4]) + ['EOE'], list(printable[:-4])]

extras = ['(', ')', 'lambda'] + ['$'+str(i) for i in range(10)]

g = Grammar.uniform(primitives + [p for p in bootstrapTarget()])
target_vocabulary = [str(p) for p in g.primitives] + extras

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

fe = LearnedFeatureExtractor(tasks=tasks)

def getDatum():
    tsk = random.choice(tasks)
    tp = tsk.request
    p = g.sample(tp) #TODO
    task = fe.taskOfProgram(p, tp)  


    examples = []

    for ex in tsk.examples:
        I, o = ex
        i = []
        for inp in I:
            i.extend(inp)
            i.append('EOE')
        examples.append((i, o))

    import pdb; pdb.set_trace()
    return tsk.examples, stringify(str(p))


if __name__=='__main__':
    m = SyntaxCheckingRobustFill(input_vocabularies=input_vocabularies,
                                target_vocabulary=target_vocabulary)



    start=time.time()
    max_n_iterations = 10000000000
    for i in range(max_n_iterations):
        batch = [getDatum() for _ in range(BATCHSIZE)]
        inputs, targets = zip(*batch)

        score = m.optimiser_step(inputs, targets)

        if i%10==0: print("Iteration %d/%d" % (i, max_n_iterations), "Score %3.3f" % score, "(%3.3f seconds per iteration)" % ((time.time()-start)/(i+1)))




