from ec import ecIterator, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs, parallelMap
from makeLogoTasks import makeTasks, montageTasks
from logoPrimitives import *
from collections import OrderedDict
from program import Program
from task import Task
from type import arrow

import datetime
import random as random
import json
import torch
import torchvision
import time
import subprocess
import os
import torch.nn as nn
import torch.nn.functional as F
from sys import exit
import pickle

from recognition import variable

global prefix_dreams

def renderLogoProgram(program,R=128):
    import numpy as np
    
    hr = subprocess.check_output(['./logoDrawString',
                                  '0',
                                  "none",
                                  str(R),
                                  str(program)],
                                 timeout=1).decode("utf8")
    try:
        hr = list(map(float, hr.split(',')))
        return np.reshape(np.array(hr),(R,R))
    except: return None

def dreamFromGrammar(g, directory, N=500):
    parallelMap(numberOfCPUs(), lambda x: saveDream(g.sample(arrow(turtle,turtle),
                                                             maximumDepth=20),
                                                    x,
                                                    directory),
                range(N))

def saveDream(program, index, directory):
    with open("%s/%d.dream"%(directory, index), "w") as handle:
        handle.write(str(program))

    for suffix in [[],["pretty"],["smooth_pretty"]]:
        try:
            subprocess.check_output(['./logoDrawString',
                                     '512',
                                     "%s/%d%s"%(directory, index,
                                                suffix[0] if suffix else ""),
                                     '0',
                                     str(program)] + suffix,
                                    timeout=1).decode("utf8")
        except: continue
        
    

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

        self.outputDimensionality = H
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.inputImageDimension = 128
        self.resizedDimension = 64
        assert self.inputImageDimension % self.resizedDimension == 0

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 1024

        


    def forward(self, v):
        assert len(v) == self.inputImageDimension*self.inputImageDimension
        floatOnlyTask = list(map(float, v))
        reshaped = [floatOnlyTask[i:i + self.inputImageDimension]
                    for i in range(0, len(floatOnlyTask), self.inputImageDimension)]
        v = variable(reshaped).float()
        # insert channel and batch
        v = torch.unsqueeze(v, 0)
        v = torch.unsqueeze(v, 0)
        window = int(self.inputImageDimension/self.resizedDimension)
        v = F.avg_pool2d(v, (window,window))
        v = self.encoder(v)
        return v.view(-1)

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.highresolution)

    def taskOfProgram(self, p, t):
        try:
            [output, highresolution] = \
                    [subprocess.check_output(['./logoDrawString',
                                              '0',
                                              "none",
                                              str(resolution),
                                              str(p)],
                                             timeout=0.05).decode("utf8")
                     for resolution in [28,self.inputImageDimension]]
            shape = list(map(float, output.split(',')))
            highresolution = list(map(float, highresolution.split(',')))
            t = Task("Helm", t, [(([0]), shape)])
            t.highresolution = highresolution
            return t
        except subprocess.TimeoutExpired:
            return None
        except subprocess.CalledProcessError:
            return None
        except ValueError:
            return None
        except OSError as exc:
            raise exc

    def renderProgram(self, p, t, index=None):
        if not os.path.exists(self.sub):
            os.system("mkdir -p %s"%self.sub)
        try:
            if index is None:
                randomStr = ''.join(random.choice('0123456789') for _ in range(10))
            else:
                randomStr = str(index)
            fname = self.sub + "/" + randomStr
            for suffix in [[],["pretty"],["smooth_pretty"]]:
                subprocess.check_output(['./logoDrawString',
                                         '512',
                                         fname + ("" if len(suffix) == 0 else suffix[0]),
                                         '0',
                                         str(p)] + suffix,
                                        timeout=1).decode("utf8")
            if os.path.isfile(fname + ".png"):
                with open(fname + ".dream", "w") as f:
                    f.write(str(p))
            return None
        except subprocess.TimeoutExpired:
            return None
        except subprocess.CalledProcessError:
            return None
        except ValueError:
            return None
        except OSError as exc:
            raise exc


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
    parser.add_argument("--split",
                        default=0., type=float)



def outputDreams(checkpoint, directory):
    from utilities import loadPickle
    result = loadPickle(checkpoint)
    eprint(" [+] Loaded checkpoint",checkpoint)
    g = result.grammars[-1]
    if directory is None:
        randomStr = ''.join(random.choice('0123456789') for _ in range(10))
        directory = "/tmp/" + randomStr
    eprint(" Dreaming into",directory)
    os.system("mkdir  -p %s"%directory)
    for n in range(500):
        try:
            p = g.sample(arrow(turtle, turtle),
                         maximumDepth=8)
            fname = directory + "/" + str(n)
            for suffix in [[],["pretty"],["smooth_pretty"]]:
                subprocess.check_output(['./logoDrawString',
                                         '512',
                                         fname + ("" if len(suffix) == 0 else suffix[0]),
                                         '0',
                                         str(p)] + suffix,
                                        timeout=1).decode("utf8")
            if os.path.isfile(fname + ".png"):
                with open(fname + ".dream", "w") as f:
                    f.write(str(p))
        except: continue
        
def visualizePrimitives(primitives, export='/tmp/logo_primitives.png'):
    from itertools import product
    from pylab import imshow,show
    from program import Index,Abstraction,Application,Primitive
    from utilities import montageMatrix,makeNiceArray
    from type import tint
    import scipy.misc
    from makeLogoTasks import parseLogo

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
            i = renderLogoProgram(pp)
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

        
if __name__ == "__main__":
    args = commandlineArguments(
        structurePenalty=1.5,
        recognitionTimeout=3600,
        a=3,
        topK=2,
        iterations=10,
        useRecognitionModel=True,
        helmholtzRatio=0.5,
        helmholtzBatch=500,
        featureExtractor=LogoFeatureCNN,
        maximumFrontier=5,
        CPUs=numberOfCPUs(),
        pseudoCounts=30.0,
        activation="tanh",
        extras=list_options)
    visualizeCheckpoint = args.pop("visualize")
    if visualizeCheckpoint is not None:
        with open(visualizeCheckpoint,'rb') as handle:
            primitives = pickle.load(handle).grammars[-1].primitives
        visualizePrimitives(primitives)
        import sys                            
        sys.exit(0)

    dreamCheckpoint = args.pop("dreamCheckpoint")
    dreamDirectory = args.pop("dreamDirectory")

    proto = args.pop("proto")

    if dreamCheckpoint is not None:
        outputDreams(dreamCheckpoint, dreamDirectory)
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

    os.chdir("prototypical-networks")
    subprocess.Popen(["python","./protonet_server.py"])
    time.sleep(3)
    os.chdir("..")


    test, train = testTrainSplit(tasks, args.pop("split"))
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))
    montageTasks(test,"test_")
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
