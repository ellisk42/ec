#policyHead

"""
if useValue is on, then require a policyHead as well, and specify which type
PolicyHead

solver needs to know about policyHead

policy needs to know how to get g ?

policy is called exclusively at (train time) or by solver, so solver can shove itself in there for base policy ... 

recognition.py:
- [X] init: should take policyHead to construct
- [X] train: if useValue: policyHead.computePolicyLoss always
- [X] inference: .cpu and .cuda policyHead

dreamcoder.py:
- [X] constructor for policyHead
- [X] singleValround thing

Astar.py
- [X] rewrite so it calls the policyHead

SMC
- [X] rewrite so it calls the policyHead

- [X] supply dist
    - [X] zipper
    - [X] grammar

- [X] specialHole ...
Build simplest policyHead

- [X] deal with token specific hole
- [X] canonical ordering

What to do about grammar we infer? leave it in ...

REPL HEAD:
- [X] build tower repl policy

- [X] for RNN, do encodeTarget option
- [X] do canonicalOrderingOption

- [ ] make easy way to copy weights
- [ ] fuss with featureExtractor

- [ ] do light policy training

- [ ] test on frontiers w/out doing search


"""
import torch
from torch import nn

import mlb
from dreamcoder.zipper import sampleSingleStep, enumSingleStep
from dreamcoder.valueHead import RNNValueHead, binary_cross_entropy, TowerREPLValueHead, ListREPLValueHead
from dreamcoder.program import Index, Program
from dreamcoder.zipper import *
from dreamcoder.utilities import count_parameters
from dreamcoder.domains.rb.rbPrimitives import *
from dreamcoder.ROBUT import ButtonSeqError, CommitPrefixError, NoChangeError
from dreamcoder.domains.list.makeDeepcoderData import *
from dreamcoder.grammar import NoCandidates
from dreamcoder.domains.misc.deepcoderPrimitives import get_lambdas
from dreamcoder.pnode import PNode,PTask
from dreamcoder.matt.sing import sing


class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()

    def sampleSingleStep(self, task, g, sk,
                        request, holeZippers=None,
                        maximumDepth=4):

        if self.ordering == 'first':
            zipper = holeZippers[0]
        elif self.ordering == 'last':
            zipper = holeZippers[-1]
        elif self.ordering == 'random':
            zipper = random.choice(holeZippers)
        else:
            raise ValueError
        try:
            dist = self._computeDist([sk], [zipper], task, g) #TODO
        except InvalidSketchError as e:
            mlb.red(f"sampleSingleStep Valuehead should have caught this: {e}")
            raise NoCandidates
        dist = dist.squeeze(0)
        supplyDist = { expr: dist[i].data.item() for i, expr in self.indexToProduction.items()}

        # for k, v in supplyDist.items():
        #     print(v, k)
        # print()

        newSk, newZippers = sampleOneStepFromHole(zipper, sk, request, g, maximumDepth, supplyDist=supplyDist)
        return newSk, newZippers

    def enumSingleStep(self, task, g, sk, request, 
                        holeZipper=None,
                        maximumDepth=4):

        try:
            dist = self.distribution([sk], [holeZipper], task, g)
        except InvalidSketchError as e:
            mlb.red(f"enumSingleStep Valuehead should have caught this: {e}")
            return # pretend there are no expansions off of it
        dist = dist.squeeze(0)
        supplyDist = { expr: dist[i].data.item() for i, expr in self.indexToProduction.items()}
        try:
            yield from enumSingleStep(g, sk, request, holeZipper=holeZipper, maximumDepth=maximumDepth, supplyDist=supplyDist)
        except NoCandidates:
            return

    def policyLossFromFrontier(self, frontier, g):
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        tp = frontier.task.request
        mlb.log('entering policyLossFromFrontier')
        
        if isinstance(entry.program, Program):
            fullProg = entry.program
        else:
            fullProg = entry.program._fullProg

        if not isinstance(self,RNNPolicyHead) and sing.cfg.model.pnode.allow_concrete_eval:
            p = PNode(fullProg,parent=None,ctx=[],from_task=frontier.task)
            # make sure concrete part of propagate() works
            assert p.upward_only_embedding().concrete == p.task.outputs.concrete
            # make sure execute_single() works
            #assert p.execute_single([]) == p.task.outputs.concrete
            assert p.execute_single([])(p.task.inputs[0].concrete[0]) == p.task.outputs.concrete[0]


        mlb.log(f'program: {fullProg}')
        #print(f'program: {fullProg}')
        mlb.log(f'request: {tp}')
        posTraces, _, targetNodes, holesToExpand = getTracesFromProg(fullProg, frontier.task.request, g, 
                                                        onlyPos=True, returnNextNode=True,
                                                        ordering=self.ordering)
        for zipper in holesToExpand:
            assert sing.cfg.solver.max_depth > len([ t for t in zipper.path if t != 'body' ]), "Astar wont be able to search this deep"
        mlb.log('pos traces:')
        #print("traces:")
        for trace,hole,target in zip(posTraces,holesToExpand,targetNodes):
            mlb.log(f'\t{trace}')
            mlb.log(f'\t\thole={hole}')
            mlb.log(f'\t\ttarget={target}')
            #print(f'\t{trace}')
            #print(f'\t\thole={hole}')
            #print(f'\t\ttarget={target}')

        maskedDist = self.distribution(posTraces, holesToExpand, frontier.task, g)
        
        # maskedDist :: [5,49]
        targets = [self._sketchNodeToIndex(node) for node in targetNodes]
        targets = torch.tensor(targets, device=sing.device)# :: [5]
        loss = self.lossFn(maskedDist, targets)
        if loss.item() == np.inf:
            mlb.red("ISSUE FOUND, you seem to be masking out the right answer")
            idx = (nn.NLLLoss(reduction='none')(maskedDist,targets) == np.inf).nonzero()
            target = targets[idx]
            node = targetNodes[idx]
            zipper = holesToExpand[idx]
            trace = posTraces[idx]
            print(f"""
            {target=}
            {node=}
            {zipper=}
            {trace=}
            {idx=}
            """)
            mask = self._buildMask([trace],[zipper],frontier.task,g)
            md1 = self._computeDist([trace], [zipper], frontier.task, g)
            print("ayy")
            maskedDist = self._computeDist(posTraces, holesToExpand, frontier.task, g)
        # if loss.item() > 300:
        #     print("MASSIVE LOSS, breakpointing")
        #     breakpoint()
        return loss

    def _computeDist(): raise NotImplementedError

    def _designateTargetHole(self, zipper, sk):
        tmpHole = Hole() # loll silly workaround im sorry i dont know how to just visit a node
        newSk,hole = NewExprPlacer(returnInnerObj=True).execute(sk, zipper.path, tmpHole) #TODO
        specialHole = Hole(target=True,tp=hole.tp)
        newSk = NewExprPlacer().execute(sk, zipper.path, specialHole) #TODO
        return newSk

    def _sketchNodeToIndex(self, node):
        if node in self.productionToIndex:
            return self.productionToIndex[node]
        if node.isAbstraction:
            return self._sketchNodeToIndex(node.body)
        if node.isApplication:
            f, xs = node.applicationParse()
            return self._sketchNodeToIndex(f)            
        else: assert False, f"invalid node {node}"

    def _buildMask(self, sketches, zippers, task, g):
        masks = []
        for zipper, sk in zip(zippers, sketches):

            # if this is a zipper into a lambda then use lambdas grammar
            if len(zipper.env) > 1:
                assert zipper.path[0] == 'body'
                assert zipper.path[1] != 'body'
                if g.g_lambdas is None: # backwards compatability. Careful it doesnt carry the max depth thru tho
                    g.g_lambdas = Grammar.uniform(get_lambdas())
                g_use = g.g_lambdas
            else:
                g_use = g

            mask = [0. for _ in range(len(self.productionToIndex))]
            candidates = returnCandidates(zipper, sk, task.request, g_use)
            for c in candidates:
                mask[self._sketchNodeToIndex(c)] = 1. 
            masks.append(mask)
        mask = torch.tensor(masks)
        mask = torch.log(mask)
        mask = mask.to(sing.device)
        return mask
class UniformPolicyHead(PolicyHead):
    def __init__(self):
        super().__init__()

    def sampleSingleStep(self, task, g, sk, request, holeZippers, maximumDepth):
        return sampleSingleStep(g, sk, request, holeZippers=holeZippers, maximumDepth=maximumDepth)

    def policyLossFromFrontier(self, frontier, g):
        return torch.tensor([0.],device=sing.device)

    def enumSingleStep(self, task, g, sk, request, holeZipper, maximumDepth):
        try:
            yield from enumSingleStep(g, sk, request, holeZipper=holeZipper, maximumDepth=maximumDepth)
        except NoCandidates:
            return

class DeepcoderListPolicyHead(PolicyHead):
    def __init__(self, g, em, cfg):
        super().__init__()
        extractor = em.encoder
        self.em = em
        self.featureExtractor = extractor
        self.cfg = cfg
        from dreamcoder.recognition import RecognitionModel
        from dreamcoder.valueHead import SampleDummyValueHead
        self.rec_model = RecognitionModel(
            featureExtractor=extractor,
            grammar=g,
            activation='relu',
            hidden=[256,256,256],
            contextual=False, # unigram
            cuda=True,
            # these might not come into play:
            useValue=True,
            valueHead=SampleDummyValueHead(),
            policyHead=BasePolicyHead(cfg),
            searchType=cfg.data.test.solver,
        )
    def policyLossFromFrontier(self, frontier, g):
        entry = frontier.sample()
        tp = frontier.task.request
        
        if isinstance(entry.program, Program):
            fullProg = entry.program
        else:
            fullProg = entry.program._fullProg

        # frontierKL just calls extractor.featuresOfTask and rec_model._MLP
        # then uses hte result as prod rule probabilities and does program.logLikelihood(grammar) with that
        # and returns the negation of the result
        neg_ll, _ = self.rec_model.frontierKL(frontier,auxiliary=False,vectorized=False)
        return neg_ll
    def sampleSingleStep(self, *args, **kwargs):
        assert False, "please initialize Astar with BasePolicyHead and the grammar returned by DeepcoderListPolicyHead if you want to do deepcoder search"
    def enumSingleStep(self, *args, **kwargs):
        assert False, "please initialize Astar with BasePolicyHead and the grammar returned by DeepcoderListPolicyHead if you want to do deepcoder search"




class RNNPolicyHead(PolicyHead):
    def __init__(self):
        super().__init__() #should have featureExtractor?
        maxVar = 15

        self.H = H = sing.cfg.model.H
        self.ordering = sing.cfg.model.ordering

        self.indexToProduction = {}
        self.productionToIndex = {}
        i = 0
        for _, _, expr in sing.g.productions:
            self.indexToProduction[i] = expr
            self.productionToIndex[expr] = i
            i += 1

        for v in range(maxVar):
            self.indexToProduction[i] = Index(v)
            self.productionToIndex[Index(v)] = i
            i += 1

        inshape = H*3
        self.output = nn.Sequential(
                nn.Linear(inshape, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.productionToIndex) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')

        print("num of params in rnn policy model", count_parameters(self))

    def distribution(self, sketches, zippers, task, g):
        ptask = PTask(task)
        #need raw dist, and then which are valid and which is correct ... 
        sketchEncodings = sing.model.program_rnn.encode_sketches(sketches) # [5,64]
        if sing.cfg.model.abstraction_fn.digitwise:
            # input feats
            #in_feats = self.featureExtractor.inputFeatures(task)
            in_feats = ptask.input_features()
            # other = sing.em.encoder.old_inputFeatures(task)
            # assert in_feats.isclose(other).all()
            in_feats = in_feats.mean(0) # mean over examples
            if sing.cfg.debug.zero_input_feats:
                in_feats = torch.zeros_like(in_feats)

            # output feats
            #out_feats = self.featureExtractor.outputFeatures(task)
            out_feats = ptask.output_features()
            # other = sing.em.encoder.old_outputFeatures(task)
            # assert out_feats.isclose(other).all()
            out_feats = out_feats.mean(0) # mean over examples
            if sing.cfg.debug.zero_output_feats:
                out_feats = torch.zeros_like(out_feats)
            
            # combine them
            features = torch.cat((in_feats,out_feats),dim=0) # shape [H*2]
        else:
            assert False
            features = self.featureExtractor.featuresOfTask(task) 
        x = features.expand(len(sketches), -1)
        features = torch.cat([sketchEncodings, x ], dim=1)
        dist = self.output(features)
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist

class ListREPLPolicyHead(PolicyHead):
    def __init__(self):
        super().__init__()
        maxVar = 10

        self.H = H = sing.cfg.model.H
        self.ordering = sing.cfg.model.ordering
        
        self.indexToProduction = {}
        self.productionToIndex = {}
        i = 0
        for _, _, expr in sing.g.productions:
            self.indexToProduction[i] = expr
            self.productionToIndex[expr] = i
            i += 1

        for v in range(maxVar):
            self.indexToProduction[i] = Index(v)
            self.productionToIndex[Index(v)] = i
            i += 1

        self.output = nn.Sequential(
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.productionToIndex) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')

        print(f"num of params in {self.__class__.__name__} policy model", count_parameters(self))

    def distribution(self, sks, zippers, task, g):
        compared = sing.model.abstract_comparer(sks,task)

        output_pnodes = [PNode(p=sk,from_task=task,parent=None,ctx=[]) for sk in sks]
        sk_reps = torch.stack([pnode.upward_only_embedding().abstract for pnode in output_pnodes]) # [num_sketches,num_exs,H]

        compared = compared.max(1).values
        dist = self.output(compared)
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist # [num_sks,49]

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


class SimpleNM(nn.Module):
    def __init__(self, nArgs, H=128):
        super(SimpleNM, self).__init__()
        self.nArgs = nArgs
        if nArgs > 0: #TODO
            #can just do a stack I think ...
            self.params = nn.Sequential(nn.Linear(nArgs*H, H), nn.ReLU())
        else:
            self.params = nn.Parameter(torch.randn(1, H))
        
    def forward(self, *args):
        #print(self.operator.name)

        if self.nArgs == 0:
            return self.params
        else:
            #print(type(args))
            inp = args[0]
            return self.params(inp)

class RBREPLPolicyHead(PolicyHead):
    """
    does not specify the target hole at all here
    """
    def __init__(self, g, featureExtractor, H, maxVar=15, encodeTargetHole=False, ordering='first'):
        super(RBREPLPolicyHead, self).__init__() #should have featureExtractor?
        assert not encodeTargetHole
        self.ordering = ordering
        self.use_cuda = torch.cuda.is_available()
        self.featureExtractor = featureExtractor
        self.H = H
        #self.REPLHead = RBREPLValueHead(g, featureExtractor, H=self.H) #hack #TODO

        self.indexToProduction = {}
        self.productionToIndex = {}
        i = 0
        for _, _, expr in g.productions:
            self.indexToProduction[i] = expr
            self.productionToIndex[expr] = i
            i += 1

        for v in range(maxVar):
            self.indexToProduction[i] = Index(v)
            self.productionToIndex[Index(v)] = i
            i += 1

        self.output = nn.Sequential(
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.productionToIndex) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')
        #print("num of params in repl policyhead (includes unused rnn weights)", count_parameters(self))


        #modules:
        self.appendModule = SimpleNM(2, H)
        self.compareModule = SimpleNM(2, H)


        self.encodeSimpleHole = nn.ModuleDict()
        for tp in [tposition, tregex, tindex, tboundary, tdelimiter, ttype]:
            self.encodeSimpleHole['t' + tp.show(True)] = SimpleNM(0, H)

        self.fnModules = nn.ModuleDict()
        for _, _, p in g.productions:
            if not p.isPrimitive: continue
            if p.tp == arrow(texpression, texpression): continue
            nArgs = len(p.tp.functionArguments())
            if p.tp.functionArguments() and p.tp.functionArguments()[0].isArrow():
               nArgs -= 1 
            
            self.fnModules[p.name] = SimpleNM(nArgs, H)
    

        self.encodeExprHole = SimpleNM(1, H)
        self.toFinishMarker = SimpleNM(1, H)

        print("num of params in repl policy model", count_parameters(self))
        if self.use_cuda: self.cuda()

    def cuda(self, device=None):
        self.use_cuda = True
        #self.REPLHead.use_cuda = True
        self.featureExtractor.use_cuda = True
        self.featureExtractor.CUDA = True
        super(RBREPLPolicyHead, self).cuda(device=device)

    def cpu(self):
        self.use_cuda = False
        #self.REPLHead.use_cuda = False
        self.featureExtractor.use_cuda = False
        self.featureExtractor.CUDA = False
        super(RBREPLPolicyHead, self).cpu()

    def _computeDist(self, sketches, zippers, task, g):
        #need raw dist, and then which are valid and which is correct ... 
        
        #features = self.featureExtractor.featuresOfTask(task)
            
        #should be the same weights here
        #inputRep = self.featureExtractor.inputsRepresentation(task)
        

        outputRep = self.featureExtractor.outputsRepresentation(task).unsqueeze(0)
        outputRep = outputRep.expand(len(sketches), -1, -1) # :: 14,4,512
        #print("outrep shape", outputRep.shape)

        currentState = [self._buildCurrentState(sk, zp, task) for sk, zp in zip(sketches, zippers)] 
        #print("currentstate0", currentState[0].shape)
        currentState = torch.stack(currentState, dim=0)
        #print("currentstate", currentState.shape)

        features = self.compareModule( torch.cat( [currentState, outputRep], dim=-1 ) ) #TODO batch


        dist = self.output(features.max(1)[0])
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist


    def _buildCurrentState(self, sketch, zipper, task):

        prevSk, scratchSk = self._seperatePrevAndScratch(sketch, task.request)
        prevRep = self.getPrevEncoding(prevSk, task)
        scratchRep = self.encodeScratch(scratchSk, task)

        currentState = self.appendModule(torch.cat( [prevRep, scratchRep], dim=-1 )) #TODO batch
        #append and compare modules do a lot of work
        return currentState


    def encodeScratch(self, sk, task, is_inner=False):
        """
        asssume that sk is an e -> e sketch with stuff unfinished

        if it's not an n prim, need to insert a first arg, which is input
        if it is an nPrim, 
        """
        if sk.isHole:
            inputRep = self.featureExtractor.inputsRepresentation(task) #TODO
            return self.encodeExprHole(inputRep)

        f, xs = sk.applicationParse()
        isNPrim = (f.tp.functionArguments() and f.tp.functionArguments()[0] == arrow(texpression, texpression))

        #might be okay if is_inner triggers nothing

        args = xs[:-1]
        neuralArgs = []

        #deal with tricky first arg 
        if not isNPrim:
            inputRep = self.featureExtractor.inputsRepresentation(task) #TODO
            neuralArgs.append(inputRep)
        else:
            inputRep = self.encodeInnerN(args[0], findHoles(args[0], task.request), task) #TODO #args[0] could be a full hole ig..
            neuralArgs.append(inputRep)
            args = args[1:]

        #deal with rest of args
        for arg in args:
            if arg.isHole:
                neuralArgs.append( self.encodeSimpleHole['t' + arg.tp.show(True)]().expand(4, -1) ) #TODO
            else:
                neuralArgs.append( self.fnModules[arg.name]().expand(4, -1) ) #TODO

        fn_input = torch.cat(neuralArgs, dim=-1)
        return self.fnModules[f.name](fn_input) #TODO


    def encodeInnerN(self, sk, zippers, task):
        assert sk.isAbstraction
        inputRep = self.featureExtractor.inputsRepresentation(task)

        #if unchosen:
        if sk.body.isHole:
            return self.encodeExprHole(inputRep)
        if not zippers:
            return self.getPrevEncoding(sk, task)

        if len(zippers)==1: 
            assert zippers[0].tp == texpression
            completedSk = NewExprPlacer().execute(sk, zippers[0].path, Index(0))
            return self.toFinishMarker(self.getPrevEncoding(completedSk, task)) #TODO

        return self.encodeScratch(sk.body, task, is_inner=True)

        #big q: how do i encode that we need to chose to end the inner fn???
        #can just include an unfinishedParam, which we need to put in ... oy.

    def getPrevEncoding(self, sk, task):
        I, O = zip(*task.examples)
        exprs = sk.evaluate([])
        try:
            newP = ROB.P( exprs ([]) ) 
            previousWords = [ ROB.executeProg(newP, i) for i in I]
            return self.featureExtractor.encodeString(previousWords) #TODO

        # except  as e:
        #     previousWords = [i for i in I ]
        #     return self.featureExtractor.encodeString(previousWords)

        except (IndexError, NoChangeError) as e:
            # print(e)
            # previousWords = [i for i in I ] #Terrible hack ... you are right.
            # print(sk)
            # print(task)
            assert False, "this should have been taken care of by valueHead"
            return self.featureExtractor.encodeString(previousWords) #TODO

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
