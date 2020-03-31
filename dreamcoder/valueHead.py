#test apis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from dreamcoder.program import Hole
from dreamcoder.grammar import *
from dreamcoder.zipper import *
import random
import time

class computeValueError(Exception):
    pass

def binary_cross_entropy(y,t, epsilon=10**-10, average=True):
    """y: tensor of size B, elements <= 0. each element is a log probability.
    t: tensor of size B, elements in [0,1]. intended target.
    returns: 1/B * - \sum_b t*y + (1 - t)*(log(1 - e^y + epsilon))"""

    B = y.size(0)
    assert len(y.size()) == 1
    log_yes_probability = y
    log_no_probability = torch.log(1 - y.exp() + epsilon)
    assert torch.ByteTensor.all(log_yes_probability <= 0.)
    assert torch.ByteTensor.all(log_no_probability <= 0.)
    correctYes = t
    correctNo = 1 - t
    ce = -(correctYes*log_yes_probability + correctNo*log_no_probability).sum()
    if average: ce = ce/B
    return ce


def sketchesFromProgram(e, tp, g):
    singleHoleSks = list( sk for sk, _ in g.enumerateHoles(tp, e, k=10))
    """really bad code for getting two holes"""
    sketches = []
    for expr in singleHoleSks:
        for sk, _ in g.enumerateHoles(tp, expr, k=10):
            sketches.append(sk)
    return sketches 

def negSketchFromProgram(e, tp, g):
    # TODO
    assert 0
    for mut in g.enumerateNearby(request, expr, distance=3.0): #Need to rewrite
        g.enumerateHoles(self, request, expr, k=10, return_obj=Hole)
        pass

def stringify(line):
    lst = []
    string = ""       
    for char in line+" ": 
        if char == " ":  
            if string != "":      
                lst.append(string)     
            string = ""              
        elif char in '()':            
            if string != "":        
                lst.append(string)     
            string = ""                
            lst.append(char)        
        else:                       
            string += char       
    return lst


class BaseValueHead(nn.Module):
    def __init__(self):
        super(BaseValueHead, self).__init__()
    def computeValue(self, sketch, task):
        assert False, "not implemented"
    def valueLossFromFrontier(frontier, g):
        assert False, "not implemented"

class SampleDummyValueHead(BaseValueHead):
    def __init__(self):
        super(BaseValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    def computeValue(self, sketch, task):
        return 0.
    def valueLossFromFrontier(self, frontier, g):
        if self.use_cuda:
            return torch.tensor([0.]).cuda()
        else: 
            return torch.tensor([0.])

class SimpleRNNValueHead(BaseValueHead):

    def __init__(self, g, featureExtractor, H=512):
        #specEncoder can be None, meaning you dont use the spec at all to encode objects
        super(SimpleRNNValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available() #FIX THIS

        extras = ['(', ')', 'lambda', '<HOLE>', '#'] + ['$'+str(i) for i in range(15)] 

        self.lexicon = [str(p) for p in g.primitives] + extras
        self.embedding = nn.Embedding(len(self.lexicon), H)

        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }

        self.model = nn.GRU(H,H,1)
        self.H = H
        self.outputDimensionality = H

        self._distance = nn.Sequential(
                nn.Linear(featureExtractor.outputDimensionality + H, H),
                nn.ReLU(),
                nn.Linear(H, 1),
                nn.Softplus())

        self.featureExtractor = featureExtractor

        if self.use_cuda:
            self.cuda()

    def _encodeSketches(self, sketches):
        #don't use spec, just there for the API
        assert type(sketches) == list
        #idk if obj is a list of objs... presuably it ususaly is 
        tokens_list = [ stringify(str(sketch)) for sketch in sketches]
        symbolSequence_list = [[self.wordToIndex[t] for t in tokens] for tokens in tokens_list]
        inputSequences = [torch.tensor(ss) for ss in symbolSequence_list] #this is impossible
        if self.use_cuda: #TODO
            inputSequences = [s.cuda() for s in inputSequences]
        inputSequences = [self.embedding(ss) for ss in inputSequences]
        # import pdb; pdb.set_trace()
        idxs, inputSequence = zip(*sorted(enumerate(inputSequences), key=lambda x: -len(x[1])  ) )
        try:
            packed_inputSequence = torch.nn.utils.rnn.pack_sequence(inputSequence)
        except ValueError:
            print("padding issues, not in correct order")
            import pdb; pdb.set_trace()

        _,h = self.model(packed_inputSequence) #dims
        unperm_idx, _ = zip(*sorted(enumerate(idxs), key = lambda x: x[1]))
        h = h[:, unperm_idx, :]
        h = h.squeeze(0)
        #o = o.squeeze(1)
        objectEncodings = h
        return objectEncodings

    def computeValue(self, sketch, task):
        taskFeatures = self.featureExtractor.featuresOfTask(task).unsqueeze(0) #memoize this plz
        sketchEncoding = self._encodeSketches([sketch])
        
        return self._distance(torch.cat([sketchEncoding, taskFeatures], dim=1)).squeeze(1).data.item()

    def valueLossFromFrontier(self, frontier, g):
        """
        given a frontier, should sample a postive trace and a negative trace
        and then train value head on those
        """
        features = self.featureExtractor.featuresOfTask(frontier.task)
        if features is None: return None, None
        features = features.unsqueeze(0)

        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        tp = frontier.task.request
        fullProg = entry.program._fullProg
        posTrace, negTrace =  getTracesFromProg(fullProg, frontier.task.request, g)

        #discard negative sketches which overlap with positive
        negTrace = [sk for sk in negTrace if (sk not in posTrace) ]

        nPos = len(posTrace)
        nNeg = len(negTrace)
        nTot = nPos + nNeg

        sketchEncodings = self._encodeSketches(posTrace + negTrace)

        # copy features a bunch
        distance = self._distance(torch.cat([sketchEncodings, features.expand(nTot, -1)], dim=1)).squeeze(1)

        targets = [1.0]*nPos + [0.0]*nNeg
        targets = torch.tensor(targets)
        if self.use_cuda:
            targets = targets.cuda()
        #import pdb; pdb.set_trace()

        loss = binary_cross_entropy(-distance, targets, average=False) #average?
        return loss


class NMN(nn.Module):
    def __init__(self, prim, H=128):
        super(NMN, self).__init__()
        self.operator = prim
    
        nArgs = len(prim.tp.functionArguments()) #is this right??
        self.nArgs = nArgs
        if nArgs > 0: #TODO
            #can just do a stack I think ...
            self.params = nn.Sequential(nn.Linear(nArgs*H, H), nn.ReLU())
        else:
            self.params = nn.Parameter(torch.randn(H))
        
    def forward(self, *args):
        #print(self.operator.name)
        if self.nArgs == 0:
            return self.params
        else:
            inp = torch.cat(args, 0) #TODO i think this is correct dim
            out = self.params(inp)
            return out


class AbstractREPLValueHead(BaseValueHead):

    def __init__(self, g, featureExtractor, H=512): #should be 512 or something
                #specEncoder can be None, meaning you dont use the spec at all to encode objects
        super(AbstractREPLValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available() #FIX THIS

        self.H = H

        assert featureExtractor.outputDimensionality == H, f"{featureExtractor.outputDimensionality} vs {H}"

        self._distance = nn.Sequential(
                nn.Linear(featureExtractor.outputDimensionality + H, H),
                nn.ReLU(),
                nn.Linear(H, 1),
                nn.Softplus())
        self.featureExtractor = featureExtractor

        self.fn_modules = nn.ModuleDict()
        self.holeParam = nn.Sequential(nn.Linear(1*H, H), nn.ReLU())
        for _, _, prim in g.productions:
            if not prim.isPrimitive: continue #This should be totally fine i think...
            self.fn_modules[prim.name] = NMN(prim, H)

        self.RNNHead = SimpleRNNValueHead(g, featureExtractor, H=self.H)

        if self.use_cuda:
            self.cuda()
        #call the thing which converts 

    def cuda(self, device=None):
        self.RNNHead.use_cuda = True
        super(AbstractREPLValueHead, self).cuda(device=device)

    def cpu(self):
        self.RNNHead.use_cuda = False
        super(AbstractREPLValueHead, self).cpu()

    def _computeOutputVectors(self, task):
        #get output representation
        outVectors = []
        for xs, y in task.examples: #TODO 
            outVectors.append( self.convertToVector(y) )
        outVectors = torch.stack(outVectors, dim=0)
        return outVectors

    def _computeSketchRepresentation(self, sketch, xs, p=None):
        if p is None:
            p = self._getInitialSketchRep(sketch)
        try:
            res = p
            for x in xs:
                res = res(x) 
        except (ValueError, IndexError, ZeroDivisionError, computeValueError, RuntimeError) as e:
            print("caught exception")
            print("sketch", sketch)
            print(e)
            raise computeValueError
        except Exception:
            print("caught exception")
            print("sketch", sketch)
            print("IO", xs)
            assert 0
        res = self.convertToVector(res) #TODO
        return res

    def _getInitialSketchRep(self, sketch):
        try:
            return sketch.abstractEval(self, [])
        except (ValueError, IndexError, ZeroDivisionError, computeValueError, RuntimeError):
            print("caught exception")
            print("sketch", sketch)
            print(e)
            raise computeValueError

    def _computeValue(self, sketch, task, outVectors=None):
        """TODO: [x] hash things like outputVector representation
        """
        if outVectors is None:
            outVectors = self._computeOutputVectors(task)
            
        p = self._getInitialSketchRep(sketch)

        evalVectors = []
        for xs, y in task.examples: #TODO 
            evalVectors.append( self._computeSketchRepresentation(sketch, xs, p=p) )

        #compare outVectors to evalVectors
        evalVectors = torch.stack(evalVectors, dim=0)
        distance = self._distance(torch.cat([evalVectors, outVectors], dim=1)).mean(0) #TODO
        return distance #Or something ...

    def computeValue(self, sketch, task):
        try:
            return self._computeValue(sketch, task).data.item()
        except (computeValueError, RuntimeError):
            return float(10**10)

    def convertToVector(self, value):
        """This is the only thing which is domain dependent. Using value for list domain currently
        could use featureExtractor
        incorporate holes by feeding them into featureExtractor as output"""
        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, Program):
            return self.RNNHead._encodeSketches([value]).squeeze(0)

        #if value in self.g.productions.values: then use module???
        y = value
        pseudoExamples = [ (() , y) ]
        vec = self.featureExtractor(pseudoExamples)
        if vec is None:
            print("ERROR: vec is none... possibly too long?")
            raise computeValueError
        return vec

    def applyModule(self, primitive, abstractArgs):
        return self.fn_modules[primitive.name](*abstractArgs)

    def encodeHole(self, hole, env):
        """todo: 
        - [ ] encode type"""
        assert hole.isHole        
        stackOfEnvVectors = [self.convertToVector(val) for val in env]
        environmentVector = self._encodeStack(stackOfEnvVectors)
        return self.holeParam(environmentVector)

    def _encodeStack(self, stackOfVectors):
        """TODO: maybe use something more sophisticated"""
        return self.convertToVector(stackOfVectors)

    def _valueLossFromFrontier(self, frontier, g):
        """
        given a frontier, should sample a postive trace and a negative trace
        and then train value head on those
        TODO: [ ] only a single call to the distance fn
        """
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        task = frontier.task
        tp = frontier.task.request
        fullProg = entry.program._fullProg
        posTrace, negTrace =  getTracesFromProg(fullProg, frontier.task.request, g)

        #discard negative sketches which overlap with positive
        negTrace = [sk for sk in negTrace if (sk not in posTrace) ]

        # compute outVectors: 
        outVectors = self._computeOutputVectors(task)

        distances = []
        targets = []
        for sk in posTrace:
            try:
                distance = self._computeValue(sk, task, outVectors=outVectors)
            except (computeValueError, RuntimeError):
                continue #TODO
            distances.append(distance)
            targets.append(1.0)

        for sk in negTrace:
            try:
                distance = self._computeValue(sk, task, outVectors=outVectors)
            except (computeValueError, RuntimeError):
                continue #TODO
            distances.append(distance)
            targets.append(0.0)

        targets = torch.tensor(targets)
        if self.use_cuda:
            targets = targets.cuda()

        distance = torch.stack( distances, dim=0 ).squeeze(1)
        loss = binary_cross_entropy(-distance, targets, average=False) #average?
        return loss

    def valueLossFromFrontier(self, frontier, g):
        try:
            return self._valueLossFromFrontier(frontier, g)
        except RuntimeError:
            return torch.tensor([0.]).cuda()

if __name__ == '__main__':
    try:
        import binutil  # required to import from dreamcoder modules
    except ModuleNotFoundError:
        import bin.binutil  # alt import if called as module

    from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
    g = Grammar.uniform([k0,k1,addition, subtraction])
    g = g.randomWeights(lambda *a: random.random())
    #p = Program.parse("(lambda (+ 1 $0))")

    m = RNNSketchEncoder(g)

    request = arrow(tint,tint)
    for ll,_,p in g.enumeration(Context.EMPTY,[],request,
                               12.):
        ll_ = g.logLikelihood(request,p)
        print(ll,p,ll_)
        d = abs(ll - ll_)
        assert d < 0.0001

        encoding = m([p])

        assert 0



