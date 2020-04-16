from collections import OrderedDict
import datetime
import json
import os
import pickle
import random as random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.domains.logo.makeLogoTasks import makeTasks, montageTasks, drawLogo
from dreamcoder.domains.logo.logoPrimitives import primitives, turtle, tangle, tlength
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.utilities import eprint, testTrainSplit, loadPickle


def animateSolutions(allFrontiers):
    programs = []
    filenames = []
    for n,(t,f) in enumerate(allFrontiers.items()):
        if f.empty: continue

        programs.append(f.bestPosterior.program)
        filenames.append(f"/tmp/logo_animation_{n}")
        
    drawLogo(*programs, pretty=True, smoothPretty=True, resolution=128, animate=True,
             filenames=filenames)
        
        
    
def dreamFromGrammar(g, directory, N=100):
    if isinstance(g,Grammar):
        programs = [ p
                     for _ in range(N)
                     for p in [g.sample(arrow(turtle,turtle),
                                        maximumDepth=20)]
                     if p is not None]
    else:
        programs = g
    drawLogo(*programs,
             pretty=False, smoothPretty=False,
             resolution=512,
             filenames=[f"{directory}/{n}.png" for n in range(len(programs)) ],
             timeout=1)
    drawLogo(*programs,
             pretty=True, smoothPretty=False,
             resolution=512,
             filenames=[f"{directory}/{n}_pretty.png" for n in range(len(programs)) ],
             timeout=1)
    drawLogo(*programs,
             pretty=False, smoothPretty=True,
             resolution=512,
             filenames=[f"{directory}/{n}_smooth_pretty.png" for n in range(len(programs)) ],
             timeout=1)
    for n,p in enumerate(programs):
        with open(f"{directory}/{n}.dream","w") as handle:
            handle.write(str(p))        
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class LogoFeatureCNN(nn.Module):
    special = "LOGO"
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(LogoFeatureCNN, self).__init__()

        self.sub = prefix_dreams + str(int(time.time()))

        self.recomputeTasks = False

        def conv_block(in_channels, out_channels, p=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.Conv2d(out_channels, out_channels, 3, padding=1),
                # nn.ReLU(),
                nn.MaxPool2d(2))

        self.inputImageDimension = 128
        self.resizedDimension = 128
        assert self.inputImageDimension % self.resizedDimension == 0

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 256

        


    def forward(self, v):
        assert len(v) == self.inputImageDimension*self.inputImageDimension
        floatOnlyTask = list(map(float, v))
        reshaped = [floatOnlyTask[i:i + self.inputImageDimension]
                    for i in range(0, len(floatOnlyTask), self.inputImageDimension)]
        v = variable(reshaped).float()
        # insert channel and batch
        v = torch.unsqueeze(v, 0)
        v = torch.unsqueeze(v, 0)
        v = maybe_cuda(v, next(self.parameters()).is_cuda)/256.
        window = int(self.inputImageDimension/self.resizedDimension)
        v = F.avg_pool2d(v, (window,window))
        v = self.encoder(v)
        return v.view(-1)

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.highresolution)

    def tasksOfPrograms(self, ps, types):
        images = drawLogo(*ps, resolution=128)
        if len(ps) == 1: images = [images]
        tasks = []
        for i in images:
            if isinstance(i, str): tasks.append(None)
            else:
                t = Task("Helm", arrow(turtle,turtle), [])
                t.highresolution = i
                tasks.append(t)
        return tasks        

    def taskOfProgram(self, p, t):
        return self.tasksOfPrograms([p], None)[0]

def list_options(parser):
    parser.add_argument("--proto",
                        default=False,
                        action="store_true",
                        help="Should we use prototypical networks?")
    parser.add_argument("--target", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--reduce", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--save", type=str,
                        default=None,
                        help="Filepath output the grammar if this is a child")
    parser.add_argument("--prefix", type=str,
                        default="experimentOutputs/",
                        help="Filepath output the grammar if this is a child")
    parser.add_argument("--dreamCheckpoint", type=str,
                        default=None,
                        help="File to load in order to get dreams")
    parser.add_argument("--dreamDirectory", type=str,
                        default=None,
                        help="Directory in which to dream from --dreamCheckpoint")
    parser.add_argument("--visualize",
                        default=None, type=str)
    parser.add_argument("--cost", default=False, action='store_true',
                        help="Impose a smooth cost on using ink")
    parser.add_argument("--split",
                        default=1., type=float)
    parser.add_argument("--animate",
                        default=None, type=str)



def outputDreams(checkpoint, directory):
    from dreamcoder.utilities import loadPickle
    result = loadPickle(checkpoint)
    eprint(" [+] Loaded checkpoint",checkpoint)
    g = result.grammars[-1]
    if directory is None:
        randomStr = ''.join(random.choice('0123456789') for _ in range(10))
        directory = "/tmp/" + randomStr
    eprint(" Dreaming into",directory)
    os.system("mkdir  -p %s"%directory)
    dreamFromGrammar(g, directory)

def enumerateDreams(checkpoint, directory):
    from dreamcoder.dreaming import backgroundHelmholtzEnumeration
    from dreamcoder.utilities import loadPickle
    result = loadPickle(checkpoint)
    eprint(" [+] Loaded checkpoint",checkpoint)
    g = result.grammars[-1]
    if directory is None: assert False, "please specify a directory"
    eprint(" Dreaming into",directory)
    os.system("mkdir  -p %s"%directory)
    frontiers = backgroundHelmholtzEnumeration(makeTasks(None,None), g, 100,
                                               evaluationTimeout=0.01,
                                               special=LogoFeatureCNN.special)()
    print(f"{len(frontiers)} total frontiers.")
    MDL = 0
    def L(f):
        return -list(f.entries)[0].logPrior
    frontiers.sort(key=lambda f: -L(f))
    while len(frontiers) > 0:
        # get frontiers whose MDL is between [MDL,MDL + 1)
        fs = []
        while len(frontiers) > 0 and L(frontiers[-1]) < MDL + 1:
            fs.append(frontiers.pop(len(frontiers) - 1))
        if fs:
            random.shuffle(fs)
            print(f"{len(fs)} programs with MDL between [{MDL}, {MDL + 1})")

            fs = fs[:500]
            os.system(f"mkdir {directory}/{MDL}")
            dreamFromGrammar([list(f.entries)[0].program for f in fs],
                             f"{directory}/{MDL}")
        MDL += 1

def visualizePrimitives(primitives, export='/tmp/logo_primitives.png'):
    from itertools import product
    from dreamcoder.program import Index,Abstraction,Application
    from dreamcoder.utilities import montageMatrix,makeNiceArray
    from dreamcoder.type import tint
    import scipy.misc
    from dreamcoder.domains.logo.makeLogoTasks import parseLogo

    angles = [Program.parse(a)
              for a in ["logo_ZA",
                        "logo_epsA",
                        "(logo_MULA logo_epsA 2)",
                        "(logo_DIVA logo_UA 4)",
                        "(logo_DIVA logo_UA 5)",
                        "(logo_DIVA logo_UA 7)",
                        "(logo_DIVA logo_UA 9)",
                        ] ]
    specialAngles = {"#(lambda (lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT (logo_MULL logo_UL 3) (logo_MULA $2 4) $0))) $1)))":
                     [Program.parse("(logo_MULA logo_epsA 4)")]+[Program.parse("(logo_DIVA logo_UA %d)"%n) for n in [7,9] ]}
    numbers = [Program.parse(n)
               for n in ["1","2","5","7","logo_IFTY"] ]
    specialNumbers = {"#(lambda (#(lambda (lambda (lambda (lambda (logo_forLoop $2 (lambda (lambda (logo_FWRT $5 (logo_DIVA logo_UA $3) $0))) $0))))) (logo_MULL logo_UL $0) 4 4))":
                      [Program.parse(str(n)) for n in [1,2,3] ]}
    distances = [Program.parse(l)
                 for l in ["logo_ZL",
                           "logo_epsL",
                           "(logo_MULL logo_epsL 2)",
                           "(logo_DIVL logo_UL 2)",
                           "logo_UL"] ]
    subprograms = [parseLogo(sp)
                   for sp in ["(move 1d 0a)",
                              "(loop i infinity (move (*l epsilonLength 4) (*a epsilonAngle 2)))",
                              "(loop i infinity (move (*l epsilonLength 5) (/a epsilonAngle 2)))",
                              "(loop i 4 (move 1d (/a 1a 4)))"]]

    entireArguments = {"#(lambda (lambda (#(#(lambda (lambda (lambda (logo_forLoop $2 (lambda (lambda (logo_FWRT $2 $3 $0))))))) logo_IFTY) (logo_MULA (#(logo_DIVA logo_UA) $1) $0) (#(logo_MULL logo_UL) 3))))":
                       [[Program.parse(str(x)) for x in xs ]
                        for xs in [("3", "1", "$0"),
                                   ("4", "1", "$0"),
                                   ("5", "1", "$0"),
                                   ("5", "3", "$0"),
                                   ("7", "3", "$0")]]}
    specialDistances = {"#(lambda (lambda (logo_forLoop 7 (lambda (lambda (#(lambda (lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $2 (lambda (lambda (logo_FWRT $2 $3 $0))))))) 7 $1 $2 $0)))) $3 logo_epsA $0))) $0)))":
                        [Program.parse("(logo_MULL logo_epsL %d)"%n) for n in range(5)]}
    
    matrix = []
    for p in primitives:
        if not p.isInvented: continue
        t = p.tp
        eprint(p,":",p.tp)
        if t.returns() != turtle:
            eprint("\t(does not return a turtle)")
            continue

        def argumentChoices(t):
            if t == turtle:
                return [Index(0)]
            elif t == arrow(turtle,turtle):
                return subprograms
            elif t == tint:
                return specialNumbers.get(str(p),numbers)
            elif t == tangle:
                return specialAngles.get(str(p),angles)
            elif t == tlength:
                return specialDistances.get(str(p),distances)
            else: return []

        ts = []
        for arguments in entireArguments.get(str(p),product(*[argumentChoices(t) for t in t.functionArguments() ])):
            eprint(arguments)
            pp = p
            for a in arguments: pp = Application(pp,a)
            pp = Abstraction(pp)
            i = np.reshape(np.array(drawLogo(pp, resolution=128)), (128,128))
            if i is not None:
                ts.append(i)
            

        if ts == []: continue

        matrix.append(ts)
        if len(ts) < 6: ts = [ts]
        else: ts = makeNiceArray(ts)
        r = montageMatrix(ts)
        fn = "/tmp/logo_primitive_%d.png"%len(matrix)
        eprint("\tExported to",fn)
        scipy.misc.imsave(fn, r)
        
    matrix = montageMatrix(matrix)
    scipy.misc.imsave(export, matrix)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on LOGO tasks.
    """

    # The below legacy global statement is required since prefix_dreams is used by LogoFeatureCNN.
    # TODO(lcary): use argument passing instead of global variables.
    global prefix_dreams

    # The below global statement is required since primitives is modified within main().
    # TODO(lcary): use a function call to retrieve and declare primitives instead.
    global primitives

    visualizeCheckpoint = args.pop("visualize")
    if visualizeCheckpoint is not None:
        with open(visualizeCheckpoint,'rb') as handle:
            primitives = pickle.load(handle).grammars[-1].primitives
        visualizePrimitives(primitives)
        sys.exit(0)

    dreamCheckpoint = args.pop("dreamCheckpoint")
    dreamDirectory = args.pop("dreamDirectory")

    proto = args.pop("proto")

    if dreamCheckpoint is not None:
        #outputDreams(dreamCheckpoint, dreamDirectory)
        enumerateDreams(dreamCheckpoint, dreamDirectory)
        sys.exit(0)

    animateCheckpoint = args.pop("animate")
    if animateCheckpoint is not None:
        animateSolutions(loadPickle(animateCheckpoint).allFrontiers)
        sys.exit(0)
        
    target = args.pop("target")
    red = args.pop("reduce")
    save = args.pop("save")
    prefix = args.pop("prefix")
    prefix_dreams = prefix + "/dreams/" + ('_'.join(target)) + "/"
    prefix_pickles = prefix + "/logo." + ('.'.join(target))
    if not os.path.exists(prefix_dreams):
        os.makedirs(prefix_dreams)
    tasks = makeTasks(target, proto)
    eprint("Generated", len(tasks), "tasks")

    costMatters = args.pop("cost")
    for t in tasks:
        t.specialTask[1]["costMatters"] = costMatters
        # disgusting hack - include whether cost matters in the dummy input
        if costMatters: t.examples = [(([1]), t.examples[0][1])]

    os.chdir("prototypical-networks")
    subprocess.Popen(["python","./protonet_server.py"])
    time.sleep(3)
    os.chdir("..")


    test, train = testTrainSplit(tasks, args.pop("split"))
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))
    if test: montageTasks(test,"test_")    
    montageTasks(train,"train_")

    if red is not []:
        for reducing in red:
            try:
                with open(reducing, 'r') as f:
                    prods = json.load(f)
                    for e in prods:
                        e = Program.parse(e)
                        if e.isInvented:
                            primitives.append(e)
            except EOFError:
                eprint("Couldn't grab frontier from " + reducing)
            except IOError:
                eprint("Couldn't grab frontier from " + reducing)
            except json.decoder.JSONDecodeError:
                eprint("Couldn't grab frontier from " + reducing)

    primitives = list(OrderedDict((x, True) for x in primitives).keys())
    baseGrammar = Grammar.uniform(primitives, continuationType=turtle)

    eprint(baseGrammar)

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/logo/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)


    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/logo"%outputDirectory,
                           evaluationTimeout=0.01,
                           **args)

    r = None
    for result in generator:
        iteration = len(result.learningCurve)
        dreamDirectory = "%s/dreams_%d"%(outputDirectory, iteration)
        os.system("mkdir  -p %s"%dreamDirectory)
        eprint("Dreaming into directory",dreamDirectory)
        dreamFromGrammar(result.grammars[-1],
                         dreamDirectory)
        r = result

    needsExport = [str(z)
                   for _, _, z
                   in r.grammars[-1].productions
                   if z.isInvented]
    if save is not None:
        with open(save, 'w') as f:
            json.dump(needsExport, f)
