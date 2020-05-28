from dreamcoder.dreamcoder import ecIterator
# from dreamcoder.domains.text.makeTextTasks import makeTasks, loadPBETasks
# from dreamcoder.domains.text.textPrimitives import primitives
from dreamcoder.domains.list.listPrimitives import bootstrapTarget
from dreamcoder.recognition import *
from dreamcoder.enumeration import *
from dreamcoder.domains.rb.rbPrimitives import robustFillPrimitives, texpression

from dreamcoder.utilities import count_parameters

import os
import datetime
import random
from functools import reduce
import dill

import dreamcoder.ROB as ROB
from dreamcoder.ROB import BUTT
import string

"""
TODO:
- [X] import robut and use its sampling
- [X] can use robut execution I suppose, or can use own execution

- [X] implement dsl
- [X] helmholtz stuff - no motif gen here

- [ ] value eval code (seperate possibly?)
- [X] neural net via featureExtractor
- [ ] dreamcoder and main cleanup
- [ ] neural net & featurextractor

- [X] make RBTask
- [X] make test tasks
"""

class RBTask():
    def __init__(self, name, request, examples, supervision=None):
        self.name = name
        self.request = request
        self.examples = examples
        self.supervision = supervision

    def __eq__(self, o): return self.name == o.name

    def __ne__(self, o): return not (self == o)

    def __hash__(self): return hash(self.name)

    def __str__(self):
        if self.supervision is None:
            return self.name
        else:
            return self.name + " (%s)"%self.supervision

    def __repr__(self):
        return "RBTask(name={self.name}, request={self.request}, examples={self.examples}"\
            .format(self=self)

    def check(self, e, timeout=None):
        #TODO errors
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        try:
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

            exprs = e.evaluate([])
            newP = ROB.P( exprs ([]) ) 
            for i, o in self.examples:
                out = ROB.executeProg(newP, i)
                if not o == out: return False
            return True

        except EvaluationTimeout:
            eprint("Timed out while evaluating", e)
            return False
        finally:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)

    def logLikelihood(self, e, timeout=None):
        if self.check(e, timeout):
            return 0.0
        else:
            return NEGATIVEINFINITY

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        #self.activation 
    def forward(self, x):
        return self.linear(x).relu()

class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, input_size, output_size):
        super(DenseBlock, self).__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_layers - 1):
            modules.append(DenseLayer(growth_rate * i + input_size, growth_rate))
        modules.append(DenseLayer(growth_rate * (num_layers - 1) + input_size, output_size))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(torch.cat(inputs, dim=-1))
            inputs.append(output)
        return inputs[-1]


class RBFeatureExtractor(nn.Module):

    def __init__(self, _=None, tasks=None,  testingTasks=[], cuda=False, lexicon=None, H=512):
        super(RBFeatureExtractor, self).__init__()
        #self.outputDimensionality 
        self.H = H
        self.nChars = len(string.printable[:-4]) + 1
        self.strLen = 36
        
        self.embedding = nn.Embedding(self.nChars, 20)                                    
        self.stringEncoding = nn.Sequential(nn.Linear(self.strLen*20 , H), nn.ReLU())
        
        self.exRepresentation = nn.Sequential(nn.Linear(2*H, H), nn.ReLU())
        self.taskRepresentation = nn.Sequential(nn.Linear(H, H), nn.ReLU())

        print("num of params in featureExtractor", count_parameters(self))

        #dense = DenseBlock(10, 128, 2*512, 2*512)
        #print("num of params in dense", count_parameters(dense))
        #print("num of params in relu", count_parameters(nn.Linear(2*H, 2*H)))

        self.use_cuda=cuda
        if cuda:
            self.cuda()


    def sampleHelmholtzTask(self, request, motifs=[]):
        assert request == arrow(texpression, texpression)
        p, I, O = ROB.generate_FIO(4)
        program = p.ecProg()
        if motifs:
            for _, expr in program.walkUncurried():
                if any( f(expr) for f in motifs):
                    return None, None

        examples = list(zip(I, O))
        task = RBTask("rb dream", request, examples, supervision=program)
        return program, task 

    @property
    def outputDimensionality(self): return self.H

    def forward(self, examples):
        I, O = zip(*examples)
        state = BUTT.RobState.new( list(I) , list(O) ) #TODO import
        inputArrays = state.str_to_np(state.inputs)
        outputArrays = state.str_to_np(state.outputs) 

        inputTensors = torch.tensor(inputArrays).long()
        if self.use_cuda: inputTensors = inputTensors.cuda()
        outputTensors = torch.tensor(outputArrays).long()
        if self.use_cuda: outputTensors = outputTensors.cuda()

        inputEmb = self.embedding(inputTensors)
        inputEnc = self.stringEncoding(inputEmb.view(inputEmb.size(0), -1) ) #TODO
 
        outputEmb = self.embedding(outputTensors)
        outputEnc = self.stringEncoding(outputEmb.view(outputEmb.size(0), -1) ) #TODO


        x = self.exRepresentation(torch.cat((inputEnc, outputEnc), dim=1) )
        x = self.taskRepresentation(x.max(0)[0])
        return x 

    def featuresOfTask(self, t):
        if hasattr(self, 'useFeatures'):
            f = self(t.features)
        else:
            # Featurize the examples directly.
            f = self(t.examples)
        return f

    def taskOfProgram(self, p, tp):
        assert False, "Shouldn't be using task of program here"

    def encodeString(self, lst):
        state = BUTT.RobState.new( lst, ["" for _ in lst])

        inputArrays = state.str_to_np(state.inputs)
        inputTensors = torch.tensor(inputArrays).long()
        if self.use_cuda: inputTensors = inputTensors.cuda()
        inputEmb = self.embedding(inputTensors)
        inputEnc = self.stringEncoding(inputEmb.view(inputEmb.size(0), -1) ) #TODO
        return inputEnc        


    def inputsRepresentation(self, task):
        I, O = zip(*task.examples)
        state = BUTT.RobState.new( list(I) , list(O) ) #TODO import
        inputArrays = state.str_to_np(state.inputs)
        inputTensors = torch.tensor(inputArrays).long()
        if self.use_cuda: inputTensors = inputTensors.cuda()

        inputEmb = self.embedding(inputTensors)
        inputEnc = self.stringEncoding(inputEmb.view(inputEmb.size(0), -1) ) #TODO
        return inputEnc

    def outputsRepresentation(self, task):
        I, O = zip(*task.examples)
        state = BUTT.RobState.new( list(I) , list(O) ) #TODO import
        #inputArrays = state.str_to_np(state.inputs)
        outputArrays = state.str_to_np(state.outputs) 
        # inputTensors = torch.tensor(inputArrays).long()
        # if self.use_cuda: inputTensors = inputTensors.cuda()
        outputTensors = torch.tensor(outputArrays).long()
        if self.use_cuda: outputTensors = outputTensors.cuda()

        outputEmb = self.embedding(outputTensors)
        outputEnc = self.stringEncoding(outputEmb.view(outputEmb.size(0), -1) ) #TODO
        return outputEnc


def makeTasks():

    import dill
    with open("full_dataset.p", 'rb') as h:
        exs = dill.load(h)

    tasks = []
    for ins, outs in exs:
        examples = list(zip(ins, outs))
        name = str(examples[0])
        tasks.append(RBTask(name, arrow(texpression, texpression), examples))
    return tasks

def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of text.
    """

    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    for t in tasks:
        t.mustTrain = False

    test, train = testTrainSplit(tasks, 0.)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))


    baseGrammar = Grammar.uniform( robustFillPrimitives(), continuationType=texpression)
    #challengeGrammar = baseGrammar  # Grammar.uniform(targetTextPrimitives)

    evaluationTimeout = 0.01

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/text/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/text"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **arguments)
    for result in generator:
        pass
