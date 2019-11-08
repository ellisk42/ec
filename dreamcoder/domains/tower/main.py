from dreamcoder.dreamcoder import *

from dreamcoder.domains.tower.towerPrimitives import primitives, new_primitives, animateTower
from dreamcoder.domains.tower.makeTowerTasks import *
from dreamcoder.domains.tower.tower_common import renderPlan, towerLength, centerTower
from dreamcoder.utilities import *

import os
import datetime

from dreamcoder.recognition import variable
import torch.nn as nn
import torch.nn.functional as F



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class TowerCNN(nn.Module):
    special = 'tower'
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(TowerCNN, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True

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
            conv_block(6, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 1024

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

    def forward(self, v, v2=None):
        """v: tower to build. v2: image of tower we have built so far"""
        # insert batch if it is not already there
        if len(v.shape) == 3:
            v = np.expand_dims(v, 0)
            inserted_batch = True
            if v2 is not None:
                assert len(v2.shape) == 3
                v2 = np.expand_dims(v2, 0)
        elif len(v.shape) == 4:
            inserted_batch = False
            pass
        else:
            assert False, "v has the shape %s"%(str(v.shape))
        
        if v2 is None: v2 = np.zeros(v.shape)
        
        v = np.concatenate((v,v2), axis=3)
        v = np.transpose(v,(0,3,1,2))
        assert v.shape == (v.shape[0], 6,self.inputImageDimension,self.inputImageDimension)
        v = variable(v, cuda=self.CUDA).float()
        window = int(self.inputImageDimension/self.resizedDimension)
        v = F.avg_pool2d(v, (window,window))
        #showArrayAsImage(np.transpose(v.data.numpy()[0,:3,:,:],[1,2,0]))
        v = self.encoder(v)
        if inserted_batch:
            return v.view(-1)
        else:
            return v

    def featuresOfTask(self, t, t2=None):  # Take a task and returns [features]
        return self(t.getImage(),
                    None if t2 is None else t2.getImage(drawHand=True))
    
    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        if t2 is None:
            pass
        elif isinstance(t2, Task):
            assert False
            #t2 = np.array([t2.getImage(drawHand=True)]*len(ts))
        elif isinstance(t2, list):
            t2 = np.array([t.getImage(drawHand=True) if t else np.zeros((self.inputImageDimension,
                                                                         self.inputImageDimension,
                                                                         3))
                           for t in t2])
        else:
            assert False
            
        return self(np.array([t.getImage() for t in ts]),
                    t2)

    def taskOfProgram(self, p, t,
                      lenient=False):
        try:
            pl = executeTower(p,0.05)
            if pl is None or (not lenient and len(pl) == 0): return None
            if len(pl) > 100 or towerLength(pl) > 360: return None

            t = SupervisedTower("tower dream", p)
            return t
        except Exception as e:
            return None




def tower_options(parser):
    parser.add_argument("--tasks",
                        choices=["old","new"],
                        default="old")
    parser.add_argument("--visualize",
                        default=None, type=str)
    parser.add_argument("--solutions",
                        default=None, type=str)
    parser.add_argument("--split",
                        default=1., type=float)
    parser.add_argument("--dream",
                        default=None, type=str)
    parser.add_argument("--primitives",
                        default="old", type=str,
                        choices=["new", "old"])


def dreamOfTowers(grammar, prefix, N=250, make_montage=True):
    request = arrow(ttower,ttower)
    randomTowers = [tuple(centerTower(t))
                    for _ in range(N)
                    for program in [grammar.sample(request,
                                                   maximumDepth=12,
                                                   maxAttempts=100)]
                    if program is not None
                    for t in [executeTower(program, timeout=0.5) or []]
                    if len(t) >= 1 and len(t) < 100 and towerLength(t) <= 360.]
    matrix = [renderPlan(p,Lego=True,pretty=True)
              for p in randomTowers]

    # Only visualize if it has something to visualize.
    if len(matrix) > 0:
        import scipy.misc
        if make_montage:
            matrix = montage(matrix)
            scipy.misc.imsave('%s.png'%prefix, matrix)
        else:
            for n,i in enumerate(matrix):
                scipy.misc.imsave(f'{prefix}/{n}.png', i)
    else:
        eprint("Tried to visualize dreams, but none to visualize.")

    
def visualizePrimitives(primitives, fn=None):
    from itertools import product
    #from pylab import imshow,show

    from dreamcoder.domains.tower.towerPrimitives import _left,_right,_loop,_embed,_empty_tower,TowerState
    _13 = Program.parse("1x3").value
    _31 = Program.parse("3x1").value

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
                return [_empty_tower]
            elif t == tint:
                return list(range(5))
            elif t == arrow(ttower,ttower):
                return [_arch,_13,_31]
            else:
                return []

        ts = []
        for arguments in product(*[argumentChoices(t) for t in t.functionArguments() ]):
            t = p.evaluate([])
            for a in arguments: t = t(a)
            t = t(TowerState())[1]
            ts.append(t)

        if ts == []: continue
        
        matrix.append([renderPlan(p,pretty=True)
                       for p in ts])

    # Only visualize if it has something to visualize.
    if len(matrix) > 0:
        matrix = montageMatrix(matrix)
        # imshow(matrix)
        
        import scipy.misc
        scipy.misc.imsave(fn, matrix)
        #    show()
    else:
        eprint("Tried to visualize primitives, but none to visualize.")

def animateSolutions(checkpoint):
    with open(checkpoint,"rb") as handle: result = dill.load(handle)
    for n,f in enumerate(result.taskSolutions.values()):
        animateTower(f"/tmp/tower_animation_{n}",f.bestPosterior.program)
    
def visualizeSolutions(solutions, export, tasks=None):

    if tasks is None:
        tasks = list(solutions.keys())
        tasks.sort(key=lambda t: len(t.plan))

    matrix = []
    for t in tasks:
        i = renderPlan(centerTower(t.plan),pretty=True,Lego=True)
        if solutions[t].empty: i = i/3.
        matrix.append(i)

    # Only visualize if it has something to visualize.
    if len(matrix) > 0:
        matrix = montage(matrix)
        import scipy.misc
        scipy.misc.imsave(export, matrix)
    else:
        eprint("Tried to visualize solutions, but none to visualize.")


def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of tower-building tasks.
    """

    # The below global statement is required since primitives is modified within main().
    # TODO(lcary): use a function call to retrieve and declare primitives instead.
    global primitives

    import scipy.misc

    g0 = Grammar.uniform({"new": new_primitives,
                          "old": primitives}[arguments.pop("primitives")],
                         continuationType=ttower)

    checkpoint = arguments.pop("visualize")
    if checkpoint is not None:
        with open(checkpoint,'rb') as handle:
            primitives = pickle.load(handle).grammars[-1].primitives
        visualizePrimitives(primitives)
        sys.exit(0)
    checkpoint = arguments.pop("solutions")
    if checkpoint is not None:
        with open(checkpoint,'rb') as handle:
            solutions = pickle.load(handle).taskSolutions
        visualizeSolutions(solutions,
                           checkpoint + ".solutions.png")
        animateSolutions(checkpoint)
        sys.exit(0)
    checkpoint = arguments.pop("dream")
    if checkpoint is not None:
        with open(checkpoint,'rb') as handle:
            g = pickle.load(handle).grammars[-1]
        os.system("mkdir  -p data/tower_dreams")
        dreamOfTowers(g, "data/tower_dreams", make_montage=False)
        sys.exit(0)
        
    
    tasks = arguments.pop("tasks")
    if tasks == "new":
        tasks = makeSupervisedTasks()
    elif tasks == "old":
        tasks = makeOldSupervisedTasks()
    else: assert False
        
    test, train = testTrainSplit(tasks, arguments.pop("split"))
    eprint("Split %d/%d test/train" % (len(test), len(train)))

    # Make a montage for the paper
    shuffledTrain = list(train)
    shuffledTest = list(test)
    random.shuffle(shuffledTrain)
    shuffledTrain = shuffledTrain + [None]*(60 - len(shuffledTrain))
    random.shuffle(shuffledTest)
    shuffledTest = shuffledTest + [None]*(60 - len(shuffledTest))
    SupervisedTower.exportMany("/tmp/every_tower.png",shuffledTrain + shuffledTest, shuffle=False, columns=10)
    for j,task in enumerate(tasks):
        task.exportImage(f"/tmp/tower_task_{j}.png")
    for k,v in dSLDemo().items():
        scipy.misc.imsave(f"/tmp/tower_dsl_{k}.png", v)
        os.system(f"convert /tmp/tower_dsl_{k}.png -channel RGB -negate /tmp/tower_dsl_{k}.png")
        

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/towers/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    os.system("mkdir  -p data/tower_dreams_initial")
    dreamOfTowers(g0, "data/tower_dreams_initial", make_montage=False)

    evaluationTimeout = 0.005
    generator = ecIterator(g0, train,
                           testingTasks=test,
                           outputPrefix="%s/tower"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **arguments)

    dreamOfTowers(g0, "%s/random_0"%outputDirectory)
    
    for result in generator:
        continue
        iteration = len(result.learningCurve)
        newTowers = [tuple(centerTower(executeTower(frontier.sample().program)))
                     for frontier in result.taskSolutions.values() if not frontier.empty]
        try:
            fn = '%s/solutions_%d.png'%(outputDirectory,iteration)
            visualizeSolutions(result.taskSolutions, fn,
                               train)
            eprint("Exported solutions to %s\n"%fn)
            dreamOfTowers(result.grammars[-1],
                          '%s/random_%d'%(outputDirectory,iteration))           
        except ImportError:
            eprint("Could not import required libraries for exporting towers.")
        primitiveFilename = '%s/primitives_%d.png'%(outputDirectory, iteration)
        visualizePrimitives(result.grammars[-1].primitives,
                            primitiveFilename)
        eprint("Exported primitives to",primitiveFilename)
