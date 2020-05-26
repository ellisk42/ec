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
models = ['repl', 'rnn']


mode = 'absStateNum' #'absState'
skipValueComp = True
excludeZeroHist = True
model = 'rnn'

TASK_IDS = list(range(len(tasks)))
nImages = 5
nRollouts = 2


print(f"tasks:")
for i in TASK_IDS:
    print(i, tasks[i].name)

sketches, ids = zip(*[( s, task_id ) \
    for task_id in TASK_IDS for s in getSketchesFromRollout(task_id, rS, nSamples=nRollouts)])

allSketches = set()
newSketches = []
newIds = []
for sk, i in zip(sketches, ids):
    if sk in allSketches: continue
    allSketches.add(sk)
    newSketches.append(sk)
    newIds.append(i)

sketches = newSketches
ids = newIds

from dreamcoder.symbolicAbstractTowers import executeAbstractSketch, ConvertSketchToAbstract, renderAbsTowerHist
symbolicVals = [ executeAbstractSketch( ConvertSketchToAbstract().execute(s) ) for s in sketches ]  

if excludeZeroHist:
    sketches, ids, symbolicVals = zip(* [(s, i, v) for (s, i, v) in zip(sketches, ids, symbolicVals)  if v.history != [(-257, 0)] ] )


print("num sketches:", len(sketches))
if model == 'repl':
    embeddings = [ rR.recognitionModel.valueHead._computeSketchRepresentation(
                    s.betaNormalForm()).cpu().detach().numpy()
                        for s in sketches]
elif model == 'rnn':
    embeddings = [ rRNN.recognitionModel.valueHead._encodeSketches(
                   [s]).squeeze(0).cpu().detach().numpy() for s in sketches ]
#import pdb; pdb.set_trace()

names = [ str(s.betaNormalForm()) for s in sketches] #could do betaNormalForm here

if skipValueComp:
    print("WARNING, not doing neural value comp")
else:
    values = [ rR.recognitionModel.valueHead.computeValue(s, tasks[i]) for s, i in zip(sketches, ids) ]
    values = [ math.exp(-v) for v in values]


if mode in ['absState', 'absStateNum']: renderedAbsVals = [renderAbsTowerHist(v, renderHand=True) for v in symbolicVals]


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
                ab = AnnotationBbox(OffsetImage(img), (xx, yy), frameon=False)
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
            for i, (xx,yy,name,img, v) in enumerate(zip(x,y,names, renderedAbsVals, symbolicVals)):
                if True: #v.history != [(-257, 0)]:
                    #print(v.history)
                    xx = xx - minx
                    yy = yy - miny
                    cropY = 32
                    cropX = 64
                    shape = img.shape
                    shape[1]//2
                    img = img[shape[0]-cropY:, shape[1]//2 - cropX//2:shape[1]//2 + cropX//2, :]
                    img = img.repeat(2, axis=0).repeat(2, axis=1)
                    #assert False, f"type img: {img.shape}"
                    plt.figimage(img, xx/l_x*5120*.95, yy/l_y*3840*.95)
                    #plt.text(xx/l_x*5120*.95, yy/l_y*3840*.95, str(i), fontsize=4)

        elif mode == 'absStateNum':
            fig, ax = plt.subplots()
            for i, (xx,yy,name,img, v) in enumerate(zip(x,y,names, renderedAbsVals, symbolicVals)):
                if True: #v.history != [(-257, 0)]:
                    #print(v.history)
                    xx = xx - minx
                    yy = yy - miny
                    cropY = 32
                    cropX = 64
                    shape = img.shape
                    shape[1]//2
                    img = img[shape[0]-cropY:, shape[1]//2 - cropX//2:shape[1]//2 + cropX//2, :]
                    img = img.repeat(2, axis=0).repeat(2, axis=1)

                    #plt.text(xx/l_x*5120*.95, yy/l_y*3840*.95, str(i), fontsize=4)
                    #b = AnnotationBbox(OffsetImage(img), (xx/l_x*5120*.95, yy/l_y*3840*.95), frameon=False)

                    ab = AnnotationBbox(OffsetImage(img, zoom=.05), (xx/l_x, yy/l_y), frameon=False)
                    ax.add_artist(ab)
                    
                    plt.text(xx/l_x, yy/l_y, str(i), fontsize=4)

        plt.savefig(f'embed_tower/model={model}_TASKS=ALL_WNum_{algo}_{mode}_{j  }.png')
        plt.clf()


    d = {}
    for i, (xx,yy,name,img, v) in enumerate(zip(x,y,names, renderedAbsVals, symbolicVals)):
        if v in d: 
            baseNum = d[v][0][0]
            d[v].append( (i, np.linalg.norm(embeddings[baseNum]-embeddings[i]) ) )
        else: d[v] = [ (i, 0.0) ]

    # distances = {}
    # for val, lst in d.items():
    #     i = lst[0]
    #     dists = [ np.linalg.norm(embeddings[i]-embeddings[j]) for j in lst ]
    #     distances[i] = dists

    for lst in d.values(): print(lst)

    #distances[i]
    #for idx in distances[i]: print()

