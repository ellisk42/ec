from ec import ecIterator, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs, parallelMap
from makeLogoTasks import makeTasks
from logoPrimitives import primitives, turtle
from collections import OrderedDict
from program import Program
from task import Task
from type import arrow

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


from recognition import variable

global prefix_dreams

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class LogoFeatureCNN(nn.Module):

    def __init__(self, tasks, cuda=False, H=64):
        super(LogoFeatureCNN, self).__init__()

        self.sub = prefix_dreams + str(int(time.time()))

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
                                             timeout=1).decode("utf8")
                     for resolution in [28,128]] 
            shape = list(map(float, output.split(',')))
            t = Task("Helm", t, [(([0]), shape)])
            t.highresolution = highresolution
            return t
        except subprocess.TimeoutExpired:
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
        

        
if __name__ == "__main__":
    args = commandlineArguments(
        structurePenalty=1.5,
        steps=2500,
        a=3,
        topK=5,
        iterations=10,
        useRecognitionModel=True,
        helmholtzRatio=0.5,
        helmholtzBatch=500,
        featureExtractor=LogoFeatureCNN,
        maximumFrontier=30,
        CPUs=numberOfCPUs(),
        pseudoCounts=10.0,
        activation="tanh",
        extras=list_options)
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


    test, train = testTrainSplit(tasks, 1.)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

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
    baseGrammar = Grammar.uniform(primitives)

    eprint(baseGrammar)

    fe = LogoFeatureCNN(tasks)
    # for x in range(0, 50):
        # program = baseGrammar.sample(arrow(turtle, turtle), maximumDepth=20)
        # features = fe.renderProgram(program, arrow(turtle, turtle), index=x)

    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix=prefix_pickles,
                           compressor="pypy",
                           evaluationTimeout=0.01,
                           **args)

    r = None
    for result in generator:
        fe = LogoFeatureCNN(tasks)
        parallelMap(numberOfCPUs(),
                    lambda x: fe.renderProgram(result.grammars[-1].sample(arrow(turtle, turtle),
                                                                          maximumDepth=20),
                                               arrow(turtle, turtle), index=x),
                    list(range(0, 500)))
        iteration = len(result.learningCurve)
        r = result

    needsExport = [str(z)
                   for _, _, z
                   in r.grammars[-1].productions
                   if z.isInvented]
    if save is not None:
        with open(save, 'w') as f:
            json.dump(needsExport, f)
