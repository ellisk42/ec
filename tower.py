from ec import *

from towerPrimitives import primitives, executeTower
from makeTowerTasks import *
from listPrimitives import bootstrapTarget

import os
import random
import time
import datetime

from recognition import variable
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class TowerCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=64):
        super(TowerCNN, self).__init__()

        self.outputDimensionality = H
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.inputImageDimension = 256
        self.resizedDimension = 64
        assert self.inputImageDimension % self.resizedDimension == 0

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(3, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 1024

    def forward(self, v):
        v = np.transpose(v,(2,0,1))
        assert v.shape == (3,self.inputImageDimension,self.inputImageDimension)
        v = variable(v).float()
        # insert batch
        v = torch.unsqueeze(v, 0)
        window = int(self.inputImageDimension/self.resizedDimension)
        v = F.avg_pool2d(v, (window,window))
        v = self.encoder(v)
        return v.view(-1)

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.getImage())

    def taskOfProgram(self, p, t):
        #from telecom_and import TowerWorld
        pl = executeTower(p,0.05)
        if pl is None or len(pl) == 0: return None
        
        t = SupervisedTower("tower dream", p.evaluate([]))
        return t
        

class TowerFeatureExtractor(ImageFeatureExtractor):
    def _featuresOfProgram(self, p, _):
        # [perturbation, mass, height, length, area]
        p = executeTower(p)
        mass = sum(w * h for _, w, h in p)

        masses = {
            t.maximumMass for t in TowerTask.tasks if mass <= t.maximumMass}
        if len(masses) == 0:
            return None
        mass = random.choice(masses)

        heights = {t.minimumHeight for t in TowerTask.tasks}
        lengths = {t.minimumLength for t in TowerTask.tasks}
        areas = {t.minimumArea for t in TowerTask.tasks}
        staircases = {t.maximumStaircase for t in TowerTask.tasks}

        # Find the largest perturbation that this power can withstand
        perturbations = sorted(
            {t.perturbation for t in TowerTask.tasks}, reverse=True)
        for perturbation in perturbations:
            result = TowerTask.evaluateTower(p, perturbation)

            possibleHeightThresholds = {
                h for h in heights if result.height >= h}
            possibleLengthThresholds = {
                l for l in lengths if result.length >= l}
            possibleAreaThresholds = {a for a in areas if result.area >= a}
            possibleStaircases = {
                s for s in staircases if result.staircase <= s}

            if len(possibleHeightThresholds) > 0 and \
               len(possibleLengthThresholds) > 0 and \
               len(possibleStaircases) > 0 and \
               len(possibleAreaThresholds) > 0:
                if result.stability > TowerTask.STABILITYTHRESHOLD:
                    return [perturbation,
                            mass,
                            random.choice(possibleHeightThresholds),
                            random.choice(possibleLengthThresholds),
                            random.choice(possibleAreaThresholds),
                            random.choice(possibleStaircases)]
            else:
                return None

        return None



def tower_options(parser):
    parser.add_argument("--tasks",
                        choices=["supervised","everything","distant"],
                        default="supervised")
    parser.add_argument("--visualize",
                        default=None, type=str)

def dreamOfTowers(grammar, prefix, N=250):
    request = arrow(ttower,ttower)
    randomTowers = [tuple(centerTower(t))
                    for _ in range(N)
                    for program in [grammar.sample(request,
                                                   maximumDepth=12,
                                                   maxAttempts=100)]
                    if program is not None
                    for t in [executeTower(program, timeout=0.5) or []]
                    if len(t) >= 1 and len(t) < 65 and towerLength(t) < 25.]
    for ti,randomTower in enumerate(randomTowers):
        fn = '%s_%d.png'%(prefix,ti)
        try:
            exportTowers([randomTower], fn)
            eprint("Exported random tower to %s\n"%fn)
        except ImportError:
            eprint("Could not import required libraries for dreaming.")
            break
        except: pass

def visualizePrimitives(primitives):
    from itertools import product
    from tower_common import fastRendererPlan,montageMatrix
    from pylab import imshow,show

    from towerPrimitives import epsilon,TowerContinuation,xOffset,_left,_right,_loop,_embed
    w,h = 2,1
    _21 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    w,h = 1,2
    _12 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    w,h = 1,3
    _13 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    w,h = 3,1
    _31 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    r = lambda n,k: _right(2*n)(k)
    l = lambda n,k: _left(2*n)(k)
    _e = _embed
    _lp = lambda n,b,k: _loop(n)(b)(k)
    _arch = lambda k: l(1,_13(r(2,_13(l(1,_31(k))))))
    _tallArch = lambda h,z,k: _lp(h, lambda _: _13(r(2,_13(l(2,z)))),
                                  r(1,_31(k)))

    matrix = []
    for p in primitives:
        if not p.isInvented: continue
        eprint(p,":",p.tp)
        t = p.tp
        if t.returns() != ttower: continue

        def argumentChoices(t):
            if t == ttower:
                return [lambda h: (h,[])]
            elif t == tint:
                return list(range(5))
            elif t == arrow(ttower,ttower):
                return [_arch,_13,_31]
            else:
                assert False

        ts = []
        for arguments in product(*[argumentChoices(t) for t in t.functionArguments() ]):
            t = p.evaluate([])
            for a in arguments: t = t(a)
            t = t(0.)[1]
            ts.append(t)

        if ts == []: continue
        
        matrix.append([fastRendererPlan(p,pretty=True)
                       for p in ts])

    matrix = montageMatrix(matrix)
    imshow(matrix)
    
    import scipy.misc
    scipy.misc.imsave('/tmp/tower_primitives.png', matrix)
    show()
    
        
            

if __name__ == "__main__":
    from tower_common import exportTowers

    g0 = Grammar.uniform(primitives)

    arguments = commandlineArguments(
        featureExtractor=TowerCNN,
        CPUs=numberOfCPUs(),
        helmholtzRatio=0.5,
        iterations=5,
        a=3,
        structurePenalty=1,
        pseudoCounts=10,
        topK=10,
        maximumFrontier=10,
        extras=tower_options)

    checkpoint = arguments.pop("visualize")
    if checkpoint is not None:
        with open(checkpoint,'rb') as handle:
            primitives = pickle.load(handle).grammars[-1].primitives
        visualizePrimitives(primitives)
        sys.exit(0)
        
    
    tasks = arguments.pop("tasks")
    if tasks == "supervised":
        tasks = makeSupervisedTasks()
    elif tasks == "distant":
        tasks = makeTasks()
    elif tasks == "everything":
        tasks = makeTasks() + makeSupervisedTasks()
    else: assert False
        
    test, train = testTrainSplit(tasks, 1.)
    eprint("Split %d/%d test/train" % (len(test), len(train)))

    evaluationTimeout = 0.005
    generator = ecIterator(g0, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/tower",
                           evaluationTimeout=evaluationTimeout,
                           solver="ocaml",
                           compressor="pypy",
                           **arguments)
    os.system("python tower_server.py KILL")
    time.sleep(1)
    os.system("python tower_server.py &")
    time.sleep(1)

    perturbations = {t.perturbation for t in train if isinstance(t,TowerTask)}

    timestamp = datetime.datetime.now().isoformat()
    os.system("mkdir -p experimentOutputs/towers/%s"%timestamp)
    dreamOfTowers(g0, "experimentOutputs/towers/%s/random_0"%timestamp)
    
    for result in generator:
        iteration = len(result.learningCurve)
        newTowers = [tuple(centerTower(executeTower(frontier.sample().program)))
                     for frontier in result.taskSolutions.values() if not frontier.empty]
        try:
            fn = 'experimentOutputs/towers/%s/solutions_%d.png'%(timestamp,iteration)
            exportTowers(newTowers, fn)
            eprint("Exported solutions to %s\n"%fn)
            dreamOfTowers(result.grammars[-1],
                          'experimentOutputs/towers/%s/random_%d'%(timestamp,iteration))
        except ImportError:
            eprint("Could not import required libraries for exporting towers.")
