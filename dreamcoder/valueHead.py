#test apis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamcoder.program import Hole,Program
from dreamcoder.grammar import *
from dreamcoder.zipper import *
from dreamcoder.utilities import RunWithTimeout
import random
import mlb
from dreamcoder.domains.tower.towerPrimitives import TowerState, _empty_tower
from dreamcoder.domains.tower.tower_common import renderPlan

from dreamcoder.program import Index, Program
import types 
from dreamcoder.domains.rb.rbPrimitives import *
from dreamcoder.ROBUT import ButtonSeqError, CommitPrefixError, NoChangeError
from dreamcoder.domains.misc.deepcoderPrimitives import int_to_int, int_to_bool, int_to_int_to_int
from dreamcoder.domains.list.makeDeepcoderData import *
from dreamcoder.pnode import PNode,PTask,FoundSolution

from dreamcoder.matt.sing import sing


class computeValueError(Exception):
    pass

def binary_cross_entropy(y,t, epsilon=10**-10, average=True):
    """y: tensor of size B, elements <= 0. each element is a log probability.
    t: tensor of size B, elements in [0,1]. intended target.
    returns: 1/B * - \sum_b t*y + (1 - t)*(log(1 - e^y + epsilon))"""

    B = y.size(0)
    assert len(y.size()) == 1, len(y.size())
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


class SemiOracleValueHead(nn.Module):
    def __init__(self, tasks, g):
        super(SemiOracleValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        ID = 'towers' + str(20)
        path = f'experimentOutputs/{ID}Sample_SRE=True.pickle'

        import dill
        with open(path, 'rb') as h:
            rS = dill.load(h)

        self.taskToSolutions = {}#todo
        for task in tasks:
            self.taskToSolutions[task] = []

            if task in rS.recognitionTaskMetrics:
                if 'frontier' in rS.recognitionTaskMetrics[task]:
                    for entry in rS.recognitionTaskMetrics[task]['frontier']:
                        self.taskToSolutions[task].append(entry.program) 

            if "Max" in task.name:
                self.taskToSolutions[task].append(task.original)
        
        self.g = g    

        #import pdb; pdb.set_trace()        

    def computeValue(self, sketch, task):
        sols = self.taskToSolutions[task]
        if not sols: return 10**10
        lls = []
        for sol in sols:
            try: 
                val, _ = self.g.sketchLogLikelihood(task.request, sol, sketch)
                lls.append(val.item())
            except AssertionError:
                raise
                continue
        ll = max(lls + [-50])
        return -ll #TODO

    def valueLossFromFrontier(self, frontier, g):
        if self.use_cuda:
            return torch.tensor([0.]).cuda()
        else: 
            return torch.tensor([0.])


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
    def value(self, root):
        assert False, "not implemented"
    def train_loss(self, p, task):
        raise NotImplementedError
    def values(self, hole, prods):
        raise NotImplementedError

class UniformValueHead(ValueHead):
    def __init__(self):
        super().__init__()
    def computeValue(self, sketch, task):
        return 0.
    def valueLossFromFrontier(self, frontier, g):
        return torch.tensor([0.], device=sing.device)

class RNNValueHead(ValueHead):

    def __init__(self):
        super().__init__()
        sing.model.em.include_encoder()
        sing.model.em.include_program_rnn()
        self.H = H = sing.cfg.model.H


        self.output = nn.Sequential(
                nn.Linear(H*2, H),
                nn.ReLU(),
                nn.Linear(H, 1),
                nn.Softplus())


    def computeValue(self, sketch, task):
        taskFeatures = self.featureExtractor.featuresOfTask(task).unsqueeze(0) #memoize this plz
        sketchEncoding = sing.em.program_rnn.encode_sketches([sketch])
        
        return self.output(torch.cat([sketchEncoding, taskFeatures], dim=1)).squeeze(1).data.item()

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

        sketchEncodings = self._encodeSketches(posTrace + negTrace) # [10,128]

        # copy features a bunch
        distance = self._distance(torch.cat([sketchEncodings, features.expand(nTot, -1)], dim=1)).squeeze(1)
        # features :: [1,128] -> expand to [10,128]
        # result of cat is [10,256]
        # result of _distance is [10,1] -> squeeze to [10]
        

        targets = [1.0]*nPos + [0.0]*nNeg
        targets = torch.tensor(targets) # :: [10]
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


class AbstractREPLValueHead(ValueHead):

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
        from dreamcoder.program import Primitive
        from dreamcoder.type import tint
        zero = Primitive('0', tint, 0)
        for _, _, prim in g.productions:
            self.fn_modules['0'] = NMN(zero, H)
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
        except RunWithTimeout:
            print("timeout on sketch:")
            print(sketch)
            raise RunWithTimeout()
        except InferenceTimeout:
            raise InferenceTimeout()
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
        except (ValueError, IndexError, ZeroDivisionError, computeValueError, RuntimeError) as e:
            print("caught exception")
            print("sketch", sketch)
            print(e)
            raise computeValueError

    def _computeValue(self, sketch, task, outVectors=None):
        """TODO: [x] hash things like outputVector representation
        """
        if outVectors is None:
            outVectors = self._computeOutputVectors(task)
            
        sketch = sketch.betaNormalForm()
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

        ###for bugs:
        # if frontier in self.mem:
        #     posTrace, negTrace = self.mem[frontier]

        # else:
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

            if distance != distance: #checks for nan
                print("got nan distance value")
                print("sketch:", sk)
                print("task:", task)
                continue
            distances.append(distance)
            targets.append(0.0)

        targets = torch.tensor(targets)
        if self.use_cuda:
            targets = targets.cuda()

        distance = torch.stack( distances, dim=0 ).squeeze(1)
        try:        
            loss = binary_cross_entropy(-distance, targets, average=False) #average?
        except AssertionError:
            raise
            print("assertion ERROR")
            print("distance:")
            print(distance)
            print("targets:")
            print(targets)
            assert 0

        return loss

    def valueLossFromFrontier(self, frontier, g):
        try:
            return self._valueLossFromFrontier(frontier, g)
        except RuntimeError as e:
            print("runtime Error")
            assert 0
            return torch.tensor([0.]).cuda()


class TowerREPLValueHead(AbstractREPLValueHead):
    def __init__(self, g, featureExtractor, H=512): #should be 512 or something
                #specEncoder can be None, meaning you dont use the spec at all to encode objects
        super(TowerREPLValueHead, self).__init__(g, featureExtractor, H=H)

        self.envEncoder = nn.GRU(H,H,1)

        self.empty_towerVec = nn.Parameter(torch.randn(H))

        self.blank_fn_vector = nn.Parameter(torch.randn(H))

        self.towerHole = nn.Sequential(nn.Linear(2*H, H), nn.ReLU())

        if self.use_cuda:
            self.cuda()

        self.mem = {}

    def convertToVector(self, value):
        if isinstance(value, torch.Tensor):
            return value

        elif isinstance(value, Program):
            print('using RNNHEAD!!!!!!!!!')
            return self.RNNHead._encodeSketches([value]).squeeze(0)

        elif isinstance(value, int):
            return self.fn_modules[str(value)]() #this should work?

        elif isinstance(value, TowerState): #is this right?
            state = value
            plan = [tup for tup in value.history if isinstance(tup, tuple)] #filters out states, leaving only actions
            hand = state.hand
            # print(hand)
            # print(state)
            # print(plan)
            image = renderPlan(plan, drawHand=hand, pretty=False, drawHandOrientation=state.orientation)
            return self.featureExtractor(image) #also encode orientation info !!!

        elif isinstance(value, tuple) and isinstance(value[0], TowerState):
            return self.convertToVector(value[0])

        elif value is _empty_tower:
            return self.empty_towerVec

        elif isinstance(value, int):
            return self.fn_modules[str(value)]()

        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], torch.Tensor):
            return value[0]

        else:
            #return value
            print(f"uncaught object {value} of type {type(value)}")
            assert False
            raise computeValueError
####
    def encodeHole(self, hole, env):
        assert hole.isHole
        #env = [e for e in env if not ( callable(e) and e is _empty_tower) ]
        #if not env == []:
            #print(f"WARNING: env for this non-tower hole: {env}")
            #raise computeValueError
        #if len(env) > 1: assert False
        #stackOfEnvVectors = [self.convertToVector(val) for val in env]
        #environmentVector = self._encodeStack(stackOfEnvVectors)
        #return self.holeParam(environmentVector)
        envStack = []
        for val in env:
            if isinstance(val, types.FunctionType): #TODO
                envStack.append(self.blank_fn_vector)
            else: envStack.append(self.convertToVector(val))

        #envStack = [self.convertToVector(e) for e in env]
        envEncoding = self._encodeStack(envStack)
        return self.holeParam(envEncoding)

    def _encodeStack(self, stackOfVectors):
        #return self.convertToVector(stackOfVectors)
        #TODO  self.envEncoder

        #inputSequences = [self.embedding(ss) for ss in inputSequences]

        inputSequences = [ torch.stack(stackOfVectors) ]
        idxs, inputSequence = zip(*sorted(enumerate(inputSequences), key=lambda x: -len(x[1])  ) )
        try:
            packed_inputSequence = torch.nn.utils.rnn.pack_sequence(inputSequence)
        except ValueError:
            print("padding issues, not in correct order")
            import pdb; pdb.set_trace()

        _,h = self.envEncoder(packed_inputSequence) #dims
        unperm_idx, _ = zip(*sorted(enumerate(idxs), key = lambda x: x[1]))
        h = h[:, unperm_idx, :]
        h = h.squeeze(0)
        #o = o.squeeze(1)
        objectEncodings = h
        return objectEncodings.squeeze(0)

    def encodeTowerHole(self, hole, env, state):
        stateVec = self.convertToVector(state)
        #assert env == [_empty_tower], f"env for this tower hole: {env}"
        #env = [e for e in env if not ( callable(e) and e is _empty_tower) ]
        #print("WARNING: env of tower hole not encoded. ENV:", env)
        env = [self.convertToVector(e) for e in env]
        envEncoding = self._encodeStack(env)
        #if len(env) > 1: assert False
        return self.towerHole( torch.cat([stateVec, envEncoding], 0))
        #return self.holeParam(stateVec) #TODO may need to change all of this up

    def _computeOutputVectors(self, task):
        outVectors = [self.featureExtractor.featuresOfTask(task)]
        outVectors = torch.stack(outVectors, dim=0)
        return outVectors
####

    def _computeSketchRepresentation(self, sketch, p=None, oldSketch=None):
        #print(sketch)
        #assert "$1" not in str(sketch), f"{sketch}"
        if p is None:
            p = self._getInitialSketchRep(sketch)
        try:
            res = p(_empty_tower)          
            res = res(TowerState(history=[]))

        except (RecursionError) as e:
            print("caught exception")
            print("sketch", sketch)
            print("oldSketch", oldSketch)
            print(e)
            #import pdb; pdb.set_trace()
            raise computeValueError
        except (ValueError, IndexError, ZeroDivisionError, computeValueError, RuntimeError) as e:
            print("caught exception")
            print("sketch", sketch)
            print(e)
            assert False
            raise computeValueError
        except RunWithTimeout:
            print("timeout on sketch:")
            print(sketch)
            raise RunWithTimeout()
        except InferenceTimeout:
            raise InferenceTimeout()
        except Exception:
            print("caught exception")
            print("sketch", sketch)
            print("oldSketch", oldSketch)
            #print("IO", xs)
            assert 0

        if isinstance(res, tuple):
            assert len(res) == 2, f"bad res type, res is: {res}"
            #assert res[1] == [], f"bad res type, res is: {res}"
            res = res[0]

        res = self.convertToVector(res) #TODO
        return res

    def _getInitialSketchRep(self, sketch):
        try:
            if not sketch.hasHoles:
                return sketch.evaluate([])
            return sketch.abstractEval(self, [])
        except (ValueError, IndexError, ZeroDivisionError, computeValueError, RuntimeError) as e:
            print("caught exception")
            print("sketch", sketch)
            print(e)
            raise computeValueError

    def _computeValue(self, sketch, task, outVectors=None):
        """TODO: [x] hash things like outputVector representation
        """
        if outVectors is None:
            outVectors = self._computeOutputVectors(task)

        oldSketch = sketch
        sketch = oldSketch.betaNormalForm()
        p = self._getInitialSketchRep(sketch)

        evalVectors = [self._computeSketchRepresentation(sketch, p=p, oldSketch=oldSketch)]
        evalVectors = torch.stack(evalVectors, dim=0)

        distance = self._distance(torch.cat([evalVectors, outVectors], dim=1)).mean(0) #TODO
        return distance #Or something ...


class InvalidIntermediatesValueHead(ValueHead):
    def __init__(self):
        super().__init__()
    def train_loss(self, p, task):
        return torch.tensor([0.], device=sing.device)
    def value(self, root):
        assert root.ntype.output, "value fn shd be fed the root of the tree, bro please try to keep up"
        assert root.has_holes
        try:
            root.propagate_upward(concrete_only=True)
        except InvalidSketchError as e:
            print(f"vhead.value() caught an invalid sketch {e}")
            return np.inf
        return 0
    def values(self, hole:PNode, prods):
        """
        Try out all the prods in the hole (preferably without much cloning) and return the list of value()s for each
        
        I think this method can really go in ValueHead in general? Other than some cache_mode question i guess. Just need a self.cache_mode for vhead ig?
        """
        root = hole.root()
        vcosts = []
        for prod in prods:
            hole.expand_to(prod) # as long as our vhead propagates upwards all cache clears will happen naturally
            if hole.check_solve():
                raise FoundSolution(hole)
            vcost = self.value(root) if root.has_holes else np.inf # infinite cost if concrete and not soln
            vcosts.append(vcost)
            hole.into_hole(cache_mode='parents') # upwards-only style cache mode

        return vcosts




class RBPrefixValueHead(ValueHead):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    def valueLossFromFrontier(self, frontier, g):
        if self.use_cuda:
            return torch.tensor([0.]).cuda()
        else: 
            return torch.tensor([0.])
    def computeValue(self, sketch, task):

        #print(sketch)
        #print(task)


        I, O = zip(*task.examples)
        prefix = self._getPrefix(sketch, task)
        exprs = prefix.evaluate([])
        try:
            newP = ROB.P( exprs ([]) ) 

            previousWords = [ ROB.executeProgWithOutputs(newP, i, o) for i, o in zip(I,O)] #TODO change this in a few ways
        except (IndexError, ButtonSeqError, CommitPrefixError, NoChangeError) as e:
            #print("fail")
            return 100000000

        #print('okay')
        return 0.


    def _getPrefix(self, sk, task):
        prev, scratch = self._seperatePrevAndScratch(sk, task.request)
        return prev

    def _seperatePrevAndScratch(self, sk, request):
        """
        prev should be full prog
        scratch is not
        """
        zippers = findHoles(sk, request)
        if len(zippers) == 1:
            assert zippers[0].tp == texpression
            scratch = Hole(tp=texpression)
            prev = NewExprPlacer().execute(sk, zippers[0].path, Index(0))

        else: 
            commonPath = []
            for group in zip(*[zipp.path for zipp in zippers]):
                if all(move == group[0] for move in group ):
                    commonPath.append(group[0])
                else: break
            prev, scratch = NewExprPlacer(allowReplaceApp=True, returnInnerObj=True ).execute(sk, commonPath, Index(0)) 
        return prev, scratch

class NM(nn.Module):
    def __init__(self, nArgs, H=512):
        super().__init__()
        self.nArgs = nArgs
        if nArgs > 0:
            self.params = nn.Sequential(nn.Linear(nArgs*H, H), nn.ReLU())
        else:
            self.params = nn.Parameter(torch.randn(1, H))
        
    def forward(self, *args):
        if self.nArgs == 0:
            assert len(args) == 0
            return self.params

        args = torch.cat(args,dim=1) # cat along example dimension. Harmless if only one thing in args anyways
        return self.params(args)

class LambdaCtx:
    def __init__(self,fn,args,vhead):
        self.fn = fn
        self.args = args
        self.vhead = vhead
        self._ctx = None
        self._idxs = {}
        self._holes = {}
    def _encode_ctx(self):
        assert len(self.args) > 0
        num_exs = self.args[0].shape[0]
        lex_embed = self.vhead.featureExtractor.lexicon_embedder
        fn = lex_embed(lex_embed.idx_of_tok[self.fn.name]).expand(1,num_exs,-1)
        start = lex_embed(lex_embed.ctx_start).expand(1,num_exs,-1)
        end = lex_embed(lex_embed.ctx_end).expand(1,num_exs,-1)
        ctx = torch.cat([fn,start] + [x.unsqueeze(0) for x in self.args] + [end])
        _, res = self.vhead.featureExtractor.ctx_encoder(ctx)
        res = res.sum(0) # sum bidir
        self._ctx = res
    def encode_idx(self, i):
        if self._ctx is None:
            self._encode_ctx()
        if i not in self._idxs:
            self._idxs[i] = self.vhead.lambdaIndexModules[i](self._ctx)
        return self._idxs[i]
    def encode_hole(self, tp):
        if self._ctx is None:
            self._encode_ctx()
        tp = tp.show(True)
        if tp not in self._holes:
            self._holes[tp] = self.vhead.lambdaHoleModules[tp](self._ctx)
        return self._holes[tp]
class ListREPLValueHead(ValueHead):

    def __init__(self):
        sing.model.em.include_encoder()
        sing.model.em.include_nms()
        super().__init__()
        self.ordering = sing.cfg.model.ordering
        self.H = H = sing.cfg.model.H
        self.compareModule = NM(2, H)
        self.output = nn.Sequential(
                nn.Linear(H*2, H),
                nn.ReLU(),
                nn.Linear(H, 1),
                nn.Softplus())


    def old_rep(self,sk,task,ctxs=None, in_lambda=False, lambda_ctx=None):
        """
        ctxs :: a list of tuples. The outer list iterates over examples, and the
            inner tuple is a context where the 0th thing is the value of $0 etc.
            Pass in None initially to initialize it. Note that this isn't a "default"
            argument for ctxs because that would make it very easy to forget to
            pass in the existing one when recursing.
        returns :: Tensor[num_exs,H] or a list of concrete values (one per example)
            in which case the type is [int] or [[int]] or [bool] where the outermost list
            is always iterating over examples.
        """
        mlb.log(f'rep() being called on sk={sk}')
        if in_lambda:
            assert lambda_ctx is not None

        # first, if this was called at the top level (ctx=0),
        # we clear out as many abstractions as there are top level inputs
        if ctxs is None: # pull initial context out of the task inputs
            assert not in_lambda
            ctxs = ctxs_of_examples(task.examples)
            sk,num_lambdas = strip_lambdas(sk)
            assert len(ctxs[0]) == num_lambdas, "Mismatch between num args passed in and num lambda abstractions"


        # if a node is a hole, encode it
        if sk.isHole:
            if in_lambda:
                if self.cfg.model.ctxful_lambdas:
                    res = lambda_ctx.encode_hole(sk.tp)
                    return res
                else:
                    holeModule = self.lambdaHoleModules[sk.tp.show(True)]
                    res = holeModule().expand(len(task.examples),-1)
                    return res
            # non-lambda
            holeModule = self.holeModules[sk.tp.show(True)]
            if holeModule.nArgs == 1:
                # hole could have a program input as a subtree so lets pass in extractor(taskinputs)
                # we use ignore_output=True to delete the outputs of each example
                arg = self.featureExtractor.inputFeatures(task)
                if self.cfg.debug.zero_input_feats:
                    arg = torch.zeros_like(arg)
                res = holeModule(arg)
                return res
            assert holeModule.nArgs == 0
            return holeModule().expand(len(task.examples),-1)

        # if a node has no holes, evalute it concretely
        if not sk.hasHoles and self.allow_concrete_eval:
            if not (in_lambda and has_index(sk,None)):
                # we dont run this if we're inside a lambda and we contain an index
                # since those can't be concrete evaluated in a lambda
                try:
                    res = evaluate_ctxs(sk,ctxs,self.cfg.data.train.V)
                except InvalidSketchError as e:
                    mlb.red(f'Valuehead validator should have already caught this: {e}')
                    mlb.red(f'\t {sk} {in_lambda}')
                    try:
                        concrete_rep(sk,task,ctxs,in_lambda,self.cfg.data.train.V)
                    except InvalidSketchError as e:
                        mlb.purple("WOULD HAVE WORKED")
                        mlb.purple("common error: your cfg.data.test for the test-time Astar is NOT actually the same as this")
                        #breakpoint()
                        print(self.cfg.data.train.V)
                        print(self.cfg.data.test.V) # but also check the ASTARs pecific test
                    raise

                if sk.size() > 1:
                    #print(f"ran concrete eval on sk of size {sk.size()}: {sk}")
                    if self._curr_task_concrete_count == task:
                        self.concrete_count += sk.size()
                return res
        
        # primitive like HALF
        if sk.isPrimitive:
            # only happens when concrete eval is turned off
            # in which case constants (eg functions like _half) can show up here
            # and in lambdas, constants can show up
            assert not self.allow_concrete_eval or in_lambda
            return [sk.value for _ in range(len(task.examples))]

        # index like $0
        if sk.isIndex:
            if in_lambda:
                assert sk.i != 2
                if self.cfg.model.ctxful_lambdas:
                    res = lambda_ctx.encode_idx(sk.i)
                    return res
                else:
                    res = self.lambdaIndexModules[sk.i]().expand(len(task.examples),-1)
                    return res
            # not in lambda
            assert not self.allow_concrete_eval
            assert sk.i == 0 # just bc im not being careful of other cases and I wanna know when they show up
            mode = self.cfg.model.encode_index_as
            if mode == 'inputs':
                return self.featureExtractor.inputFeatures(task)
            elif mode == 'constant':
                assert sk.i == 0
                return self.indexModule().expand(len(task.examples),-1)
            else:
                raise ValueError

        if sk.isAbstraction:
            # deepcoder++ only
            assert self.cfg.data.expressive_lambdas
            assert not in_lambda, "nested lambda should never happen"
            sk,i = strip_lambdas(sk)
            assert i <= 2
            return self.rep(sk,task,ctxs,in_lambda=True, lambda_ctx=lambda_ctx)
            

        # sk is an Application
        fn, args = sk.applicationParse()
        assert len(args) > 0
        reps = []
        # recurse on children
        for arg in reversed(args): # eval in reverse order so we see any arg that is a lambda last
            if not in_lambda: # if already in a lambda just propagate down the above context, dont overwrite it
                lambda_ctx = LambdaCtx(fn, list(reversed(reps)), self)
            rep = self.rep(arg,task,ctxs,in_lambda,lambda_ctx=lambda_ctx)
            # rep :: [num_exs,H]
            if not torch.is_tensor(rep):
                # encode concrete values
                rep = self.featureExtractor.encodeValue(rep)
                if self.cfg.debug.zero_concrete_eval:
                    rep = torch.zeros_like(rep)
            reps.append(rep)
        
        reps = list(reversed(reps))

        if fn.isAbstraction:
            assert False
            ## note this wont even show up with deepcocder++ lambdas bc theyre always simply arguments to higher order functions like MAP
            # never the case in vanilla deepcoder
            # note that we only ever approach abstractions from this higher level of
            # the applicationParse so that we already know what args it takes
            assert len(args) == 1 # i think abstractions can only take one argument?
            ctxs = [(arg,)+ctx for ctx,arg in zip(ctxs,args[0])]
            raise NotImplementedError # TODO I'm gonna hold off on this until we actually need it in case details when we actually need it affect things
            return self.rep(fn.body)() # unfinished

        return self.fnModules[fn.name](*reps)

    def computeValue(self, sketch, task):
        compared = sing.model.em.sk_task_pnode_compare([sketch], task, reduce='max')
        value = self.output(compared).squeeze(1)
        return value
    

    def valueLossFromFrontier(self, frontier, g):
        """
        given a frontier, should sample a postive trace and a negative trace
        and then train value head on those
        """

        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        tp = frontier.task.request
        fullProg = entry.program._fullProg
        posTrace, negTrace =  getTracesFromProg(fullProg, frontier.task.request, g)

        #discard negative sketches which overlap with positive
        negTrace = [sk for sk in negTrace if (sk not in posTrace) ]

        # TODO I added this bc otherwise we dont get tensor outputs from rep(). Idk if it makes sense tho
        posTrace = [sk for sk in posTrace if sk.hasHoles]
        negTrace = [sk for sk in negTrace if sk.hasHoles]
        #negTrace = [] # TODO TEMP

        nPos = len(posTrace)
        nNeg = len(negTrace)
        nTot = nPos + nNeg

        compared = sing.em.model.compare(posTrace+negTrace, frontier.task, reduce='max')
        distance = self._distance(compared).squeeze(1)

        targets = torch.tensor([1.0]*nPos + [0.0]*nNeg)
        if self.use_cuda:
            targets = targets.cuda()

        loss = binary_cross_entropy(-distance, targets, average=False) #average?
        return loss




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



