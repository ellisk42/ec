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


import dill
import numpy as np

ID = "listBaseIT=1" #"listBaseIT=1Long"
path = f'experimentOutputs/{ID}REPL_SRE=True.pickle'
with open(path, 'rb') as h:
    result = dill.load(h)

bootstrapTarget()
g = result.grammars[0]
tasks = result.taskSolutions.keys()

# print("n tasks", len(tasks))
# valueHead = result.recognitionModel.valueHead
# e = np.array([ valueHead.featureExtractor.featuresOfTask(task).cpu().detach().numpy() for task in tasks])
# print ("e shape", e.shape)
# names = [task.name + "_HIT" if len(result.taskSolutions[task].entries) > 0 else task.name + "_MISS" for task in tasks ] 

##############
# task = list(tasks)[1]
# frontier = result.taskSolutions[task]
# print(task.name)
# print(frontier)

# entry = frontier.sample()
# fullProg = entry.program
# posTrace, negTrace =  getTracesFromProg(fullProg, frontier.task.request, g)
# negTrace = [sk for sk in negTrace if (sk not in posTrace) ]

# valueHead = result.recognitionModel.valueHead
# xs, _ = task.examples[-1]
# n_ex = 5

# embeddings = [valueHead._computeSketchRepresentation(sketch, xs).cpu().detach().numpy() for sketch in posTrace+negTrace for xs,_ in task.examples[:n_ex]]
# print("num embeddings", len(embeddings))
# e = np.array(embeddings)
# print ("e shape", e.shape)
# names = [ "POS "+str(sketch)+"INPUTS:"+str(xs) for sketch in posTrace for xs,_ in task.examples[:n_ex] ]
# names.extend([ "NEG" + str(sketch)+"INPUTS:"+str(xs) for sketch in negTrace for xs,_ in task.examples[:n_ex] ] )


#################
s_xs = [
('(lambda (+ 1 <HOLE>))', (0,) ),
('(lambda (+ <HOLE> 1))', (0,) ),
('(lambda (+ <HOLE> 2))', (0,) ),
('(lambda (+ 2 <HOLE>))', (0,) ),
('(lambda (+ 1 <HOLE>))', (1,) ),
('(lambda (+ <HOLE> 1))', (1,) ),
('(lambda (+ <HOLE> 2))', (1,) ),
('(lambda (+ 2 <HOLE>))', (1,) ),
('(lambda (+ 1 <HOLE>))', (2,) ),
('(lambda (+ <HOLE> 1))', (2,) ),
('(lambda (+ <HOLE> 2))', (2,) ),
('(lambda (+ 2 <HOLE>))', (2,) ),
('(lambda (+ 1 <HOLE>))', (3,) ),
('(lambda (+ <HOLE> 1))', (3,) ),
('(lambda (+ <HOLE> 2))', (3,) ),
('(lambda (+ 2 <HOLE>))', (3,) ),

('(lambda (map (lambda <HOLE>) <HOLE>))', ([0, 0, 0, 0],) ),
('(lambda (map (lambda <HOLE>) <HOLE>))', ([1, 1, 0, 0],) ),
('(lambda (map (lambda <HOLE>) <HOLE>))', ([1, 1, 1, 0],) ),
('(lambda (map (lambda <HOLE>) <HOLE>))', ([1, 1, 1, 1],) ),
('(lambda (map (lambda <HOLE>) <HOLE>))', ([0, 0, 0, 1],) ),
('(lambda (map (lambda <HOLE>) <HOLE>))', ([0, 0, 1, 1],) ),
('(lambda (map (lambda <HOLE>) <HOLE>))', ([0, 1, 1, 1],) ),

('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([0, 0, 0, 0],) ),
('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([1, 1, 0, 0],) ),
('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([1, 1, 1, 0],) ),
('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([1, 1, 1, 1],) ),
('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([0, 0, 0, 1],) ),
('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([0, 0, 1, 1],) ),
('(lambda (map (lambda (+ 1 $0)) <HOLE>))', ([0, 1, 1, 1],) ),

('(lambda (map (lambda <HOLE>) $0))', ([0, 0, 0, 0],) ),
('(lambda (map (lambda <HOLE>) $0))', ([1, 1, 0, 0],) ),
('(lambda (map (lambda <HOLE>) $0))', ([1, 1, 1, 0],) ),
('(lambda (map (lambda <HOLE>) $0))', ([1, 1, 1, 1],) ),
('(lambda (map (lambda <HOLE>) $0))', ([0, 0, 0, 1],) ),
('(lambda (map (lambda <HOLE>) $0))', ([0, 0, 1, 1],) ),
('(lambda (map (lambda <HOLE>) $0))', ([0, 1, 1, 1],) ),

('(lambda (car <HOLE>))', ([0, 1, 1, 1],) ),
('(lambda (car <HOLE>))', ([0, 1, 0, 1],) ),
('(lambda (car <HOLE>))', ([1, 1, 1, 1],) ),
('(lambda (car <HOLE>))', ([1, 1, 1, 0],) ),

('(lambda (car $0))', ([0, 1, 1, 1],) ),
('(lambda (car $0))', ([0, 1, 0, 1],) ),
('(lambda (car $0))', ([1, 1, 1, 1],) ),
('(lambda (car $0))', ([1, 1, 1, 0],) ),

('(lambda (cdr <HOLE>))', ([0, 1, 1, 1],) ),
('(lambda (cdr <HOLE>))', ([0, 1, 0, 1],) ),
('(lambda (cdr <HOLE>))', ([1, 1, 1, 1],) ),
('(lambda (cdr <HOLE>))', ([1, 1, 1, 0],) ),

('(lambda (cdr $0))', ([0, 1, 1, 1],) ),
('(lambda (cdr $0))', ([0, 1, 0, 1],) ),
('(lambda (cdr $0))', ([1, 1, 1, 1],) ),
('(lambda (cdr $0))', ([1, 1, 1, 0],) ),

]

p = Program.parse('(lambda (map (lambda <HOLE>) <HOLE>))')
e = p.evaluate([])
print(e)

valueHead = result.recognitionModel.valueHead
embeddings = [valueHead._computeSketchRepresentation(Program.parse(sketchStr), xs).cpu().detach().numpy() for sketchStr, xs in s_xs ]
print("num embeddings", len(embeddings))
e = np.array(embeddings)
names = [str(Program.parse(sketchStr)) + "inputs: " + str(xs) for sketchStr, xs in s_xs]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
plt.rcParams['figure.dpi'] = 800

for i in range(10):
    e_tsne = TSNE(n_components=2).fit_transform(e)
    x = [x[0] for x in e_tsne]
    y = [x[1] for x in e_tsne]

    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)

    l_x, l_y = maxx-minx, maxy-miny

    for xx,yy,name in zip(x,y,names):
        xx = xx - minx
        yy = yy - miny
        plt.text(xx/l_x, yy/l_y, name, fontsize=4)

    plt.savefig(f'emb_images/emb_{i}.png')
    plt.clf()