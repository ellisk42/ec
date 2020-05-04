#towerVisualizaton.py
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.tower.towerPrimitives import ttower, arrow
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *
import dill
import numpy as np

def getSketchesFromRollout(i, rS, nSamples=1, excludeConcrete=True):
    tasks = rS.getTestingTasks()
    task = tasks[i]
    gS = rS.recognitionModel.grammarOfTask(task)
    from dreamcoder.zipper import sampleSingleStep,baseHoleOfType,findHoles
    tp = arrow(ttower, ttower)  
    h = baseHoleOfType(tp)
    zippers = findHoles(h, tp)

    sketches = set()
    for _ in range(nSamples):
        newOb = h
        newZippers = zippers
        while newZippers:
            newOb, newZippers = sampleSingleStep(gS, newOb, tp, holeZippers=newZippers, maximumDepth=8)
            if newZippers: sketches.add(newOb)
    return list(sketches)

#load tasks and models
sys.setrecursionlimit(5000)
n = 3
ID = 'towers' + str(n)

nameSalt = "towers"

paths = [(f'experimentOutputs/{ID}Sample_SRE=True_graph=True.pickle', 'Sample'),
    (f'experimentOutputs/{ID}RNN_SRE=True_graph=True.pickle', 'RNN value'),
    (f'experimentOutputs/{ID}REPL_SRE=True_graph=True.pickle', 'REPL modular value')]

paths = [(f'experimentOutputs/{ID}Sample_SRE=True.pickle', 'Sample'),
    (f'experimentOutputs/{ID}RNN_SRE=True.pickle', 'RNN value'),
    (f'experimentOutputs/{ID}REPL_SRE=True.pickle', 'REPL modular value')]
paths, names = zip(*paths)
with open(paths[0], 'rb') as h:
    rS = dill.load(h)
with open(paths[1], 'rb') as h:
    rRNN = dill.load(h)
with open(paths[2], 'rb') as h:
    rR = dill.load(h)
tasks = rS.getTestingTasks()

ALGOS = ['tsne', 'pca']
MODES = ['text', 'color', 'image', 'valueText', 'imageValue', 'symbolicOrientation', 'maxH', 'symbolicHand', 'absState']

mode = 'absState'
TASK_IDS = list(range(len(tasks)))
nImages = 4
nRollouts = 1


print(f"tasks:")
for i in TASK_IDS:
    print(i, tasks[i].name)

sketches, ids = zip(*[( s, task_id ) \
    for task_id in TASK_IDS for s in getSketchesFromRollout(task_id, rS, nSamples=nRollouts)])
   
print("num sketches:", len(sketches))
embeddings = [ rR.recognitionModel.valueHead._computeSketchRepresentation(
                s.betaNormalForm()).cpu().detach().numpy()
                    for s in sketches]
names = [ str(s.betaNormalForm()) for s in sketches] #could do betaNormalForm here
values = [ rR.recognitionModel.valueHead.computeValue(s, tasks[i]) for s, i in zip(sketches, ids) ]
values = [ math.exp(-v) for v in values]
from dreamcoder.symbolicAbstractTowers import executeAbstractSketch, ConvertSketchToAbstract, renderAbsTowerHist

symbolicVals = [ executeAbstractSketch( ConvertSketchToAbstract().execute(s) ) for s in sketches ]

if mode == 'absState': renderedAbsVals = [renderAbsTowerHist(v.history) for v in symbolicVals]


e = np.array(embeddings)

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
plt.rcParams['figure.dpi'] = 800

for algo in ALGOS:
    for j in range(nImages):
        if algo == 'tsne':
            e_red = TSNE(n_components=2).fit_transform(e)
        else:
            e_red = PCA(n_components=2).fit_transform(e)
        x = [x[0] for x in e_red]
        y = [x[1] for x in e_red]
        minx, maxx = min(x), max(x)
        miny, maxy = min(y), max(y)
        l_x, l_y = maxx-minx, maxy-miny

        if mode == 'text':
            for xx,yy,name in zip(x,y,names):
                xx = xx - minx
                yy = yy - miny
                plt.text(xx/l_x, yy/l_y, name, fontsize=4)

        elif mode == 'color':
            x = x - minx
            y = y - miny
            plt.title("score given by value (higher is better)")
            plt.scatter(x, y, c=values)
            plt.colorbar()

        elif mode == 'image':
            for xx,yy,name,i in zip(x,y,names, ids):
                xx = xx - minx
                yy = yy - miny
                img = tasks[i].getImage()
                #assert False, f"type img: {img.shape}"
                plt.figimage(img, xx/l_x*5120*.95, yy/l_y*3840*.95)

        elif mode == 'imageValue':
            x = x - minx
            y = y - miny
            fig, ax = plt.subplots()

            plt.title("score given by value (higher is better)")
            ax.scatter(x, y, c=values)
            #plt.colorbar()
            for xx,yy,name,i in zip(x,y,names, ids):
                xx = xx - minx
                yy = yy - miny
                img = tasks[i].getImage()
                #plt.figimage(img, xx/l_x*5120*.95, yy/l_y*3840*.95)
                ab = AnnotationBbox(img, (xx, yy), frameon=False)
                ax.add_artist(ab)

        elif mode == 'valueText':
            for xx,yy,v in zip(x,y,values):
                xx = xx - minx
                yy = yy - miny
                plt.text(xx/l_x, yy/l_y, "%.4f" %(v), fontsize=4)

        elif mode == 'symbolicHand':
            for xx,yy,sv in zip(x,y,symbolicVals):
                xx = xx - minx
                yy = yy - miny
                plt.text(xx/l_x, yy/l_y, f"{sv.hand}", fontsize=4)

        elif mode == 'symbolicOrientation':
            for xx,yy,sv in zip(x,y,symbolicVals):
                xx = xx - minx
                yy = yy - miny
                plt.text(xx/l_x, yy/l_y, f"{sv.orientation}", fontsize=4)

        elif mode == 'maxH':
            for xx,yy,sv in zip(x,y,symbolicVals):
                maxH = max(sv.history, key=lambda x: x[1])
                if maxH != (-257, 0):
                    xx = xx - minx
                    yy = yy - miny
                    plt.text(xx/l_x, yy/l_y, f"{maxH}", fontsize=4)

        elif mode == 'absState':
            for xx,yy,name,img, v in zip(x,y,names, renderedAbsVals, symbolicVals):
                if v.history != [(-257, 0)]:
                    #print(v.history)
                    xx = xx - minx
                    yy = yy - miny
                    #assert False, f"type img: {img.shape}"
                    plt.figimage(img, xx/l_x*5120*.95, yy/l_y*3840*.95)


        plt.savefig(f'embed_tower/TASKS={TASK_IDS}_{algo}_{mode}_{j  }.png')
        plt.clf()