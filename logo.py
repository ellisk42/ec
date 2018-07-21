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
import time
import subprocess
import os
import torch.nn as nn
import torch.nn.functional as F

from recognition import variable

global prefix_dreams


class LogoFeatureCNN(nn.Module):

    def __init__(self, tasks, cuda=False, H=10):
        super(LogoFeatureCNN, self).__init__()

        self.sub = prefix_dreams + str(int(time.time()))

        self.outputDimensionality = H

        self.pad = nn.ConstantPad2d(2, 0)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, H)

    def forward(self, v):
        x = 28
        y = 28
        floatOnlyTask = list(map(float, v))
        reshaped = [floatOnlyTask[i:i + x]
                    for i in range(0, len(floatOnlyTask), y)]
        v = variable(reshaped).float()
        v = self.pad(v)
        v = torch.unsqueeze(v, 0)
        v = torch.unsqueeze(v, 0)
        out = F.relu(self.conv1(v))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = torch.squeeze(out)
        return out

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.examples[0][1])

    def taskOfProgram(self, p, t):
        try:
            output = subprocess.check_output(['./logoDrawString',
                                              '0',
                                              "none",
                                              '28',
                                              str(p)],
                                             timeout=1).decode("utf8")
            shape = list(map(float, output.split(',')))
            return Task("Helm", t, [(([0]), shape)])
        except subprocess.TimeoutExpired:
            return None
        except ValueError:
            return None
        except OSError as exc:
            raise exc

    def renderProgram(self, p, t, index=None):
        if not os.path.exists(self.sub):
            os.system("mkdir -p "%self.sub)
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


if __name__ == "__main__":
    args = commandlineArguments(
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
    target = args.pop("target")
    red = args.pop("reduce")
    save = args.pop("save")
    prefix = args.pop("prefix")
    prefix_dreams = prefix + "/dreams/" + ('_'.join(target)) + "/"
    prefix_pickles = prefix + "/logo." + ('.'.join(target))
    if not os.path.exists(prefix_dreams):
        os.makedirs(prefix_dreams)
    tasks = makeTasks(target)
    eprint("Generated", len(tasks), "tasks")

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
    for x in range(0, 100):
        program = baseGrammar.sample(arrow(turtle, turtle), maximumDepth=20)
        features = fe.renderProgram(program, arrow(turtle, turtle), index=x)

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
