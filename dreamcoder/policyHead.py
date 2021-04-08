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

from einops import reduce,rearrange

class PolicyHead(nn.Module):
    def __init__(self, cache_mode):
        super().__init__()
        self.cache_mode = cache_mode

        self.H = H = sing.cfg.model.H
        #self.ordering = sing.cfg.model.ordering
        #self.tiebreaking = sing.cfg.model.tiebreaking

        self.index_to_prod = {}
        self.prod_to_index = {}
        i = 0
        for p in sing.g.primitives:
            self.index_to_prod[i] = p
            self.prod_to_index[p] = i
            i += 1
        for v in range(sing.cfg.model.max_var):
            self.index_to_prod[i] = Index(v)
            self.prod_to_index[Index(v)] = i
            i += 1

    def sample_action(self,
                      root,
                      max_depth):
        """
        Returns a production which you could then apply with hole.expand_to(prod, clear_cache=self.cache_mode) (eg SMC)
        """
        assert root.has_holes
        verify_str = root.root_str()
        hole = root.get_hole(sing.cfg.model.ordering,sing.cfg.model.tiebreaking)
        try:
            prods,lls = self.action_distribution(hole,max_depth)
        except InvalidSketchError as e:
            red(f"sampleSingleStep Valuehead should have caught this: {e}")
            raise NoCandidates
        idx = sample_log_dist(lls)
        prod = prods[idx]
        assert root.root_str() == verify_str, "root was mutated"
        return hole, prod

    def enumerate_actions(self,
                      root,
                      max_depth):
        """
        Returns a (prods,lls) tuple, that is a list of productions and their corresponding lls. 
        hole.expand_to(prod, clear_cache=self.cache_mode) could be used on each with cloning in between (eg Astar)
        """
        assert root.has_holes
        verify_str = root.root_str()
        hole = root.get_hole(sing.cfg.model.ordering,sing.cfg.model.tiebreaking)
        try:
            prods,lls = self.action_distribution(hole,max_depth)
        except InvalidSketchError as e:
            red(f"enumSingleStep Valuehead should have caught this: {e}")
            raise NoCandidates

        assert root.root_str() == verify_str, "root was mutated"
        return hole, prods,lls

    def train_loss(self, ps, tasks):
        """

        """

        assert all(not p.hasHoles for p in ps)

        if sing.cfg.debug.pnode_concrete_check:
            if not isinstance(self,RNNPolicyHead) and sing.cfg.model.pnode.allow_concrete_eval:
                _p = PNode.from_dreamcoder(ps[0],tasks[0])
                """
                Test all parts of our concrete eval system including
                    beval()
                    beval_single_concrete()
                    {Examplewise,EWClosure,EWContext}.split()
                """

                #assert _p.propagate_upward().concrete == _p.task.outputs.concrete
                assert (tmp:=_p.beval(ctx=None).concrete) == _p.task.outputs.concrete
                del tmp
                _p_closure = _p.beval_single_concrete(ctx=None)
                for _inputs,_outputs in zip(_p.task.ctx.split(), _p.task.outputs.concrete):
                    assert _p_closure(*_inputs) == _outputs

                del _p,_p_closure,_inputs,_outputs
        
        ps = [PNode.from_dreamcoder(p,task) for p,task in zip(ps,tasks)]
        tasks = [p.task for p in ps]

        processed_holes, masks, targets, strings = [],[],[],[]
        in_feats, out_feats = [],[]

        for p,task in zip(ps,tasks):
            _processed_holes, _masks, _targets, _strings = self.trace_and_process_holes(p)
            processed_holes += _processed_holes
            masks += _masks
            targets += _targets
            strings += _strings
            in_feats.append(task.input_features().expand(len(_processed_holes),-1,-1)) # input_features() -> [num_exs,H] so we expand to [num_sks, num_exs,H]
            out_feats.append(task.output_features().expand(len(_processed_holes),-1,-1))

        mlb.freezer('trace')
        masks = rearrange(masks, 'num_sks Q -> num_sks Q') # [num_sks,Q]
        targets = rearrange(targets, 'num_sks -> num_sks') # [num_sks] the non-onehot version
        in_feats = torch.cat(in_feats) # [num_sks, num_exs,H]
        out_feats = torch.cat(out_feats) # [num_sks, num_exs,H]

        dists = self.unmasked_distributions(processed_holes,in_feats,out_feats) # [num_sks,Q]
        assert dists.shape == masks.shape

        masked_dists = dists + masks

        loss = self.lossFn(masked_dists, targets)
        
        if loss.item() == np.inf:
            mlb.red("You seem to be masking out the right answer")
            assert False
            # idx = (nn.NLLLoss(reduction='none')(maskedDist,targets) == np.inf).nonzero()
        return loss
    
    def masked_distribution(self, hole, max_depth):
        """
        """
        mask = self.build_mask(hole, max_depth)
        processed_hole = self.process_hole(hole)
        in_feats = hole.task.input_features()[None,:,:]
        out_feats = hole.task.output_features()[None,:,:]
        dist = self.unmasked_distributions([processed_hole],in_feats,out_feats).squeeze(0)
        return mask + dist

    def action_distribution(self, hole, max_depth):
        """
        """
        dist = self.masked_distribution(hole, max_depth)
        prod_ll = [(prod, dist[i].item()) for i, prod in self.index_to_prod.items() if dist[i].item() != -np.inf]
        prods,lls = list(zip(*prod_ll))
        lls = normalize_log_dist(lls)
        return prods,lls


    def build_target(self, hole):
        """
        """
        target = self.prod_to_index[hole.get_prod()]
        return torch.tensor(target, device=sing.device) # not a onehot

    def build_mask(self, hole, max_depth):
        """
        """
        g_use = sing.g.g_lambdas if hole.in_HOF_lambda else sing.g

        # tp.returns() is `self` if its a base type or the return type if its an arrow type
        indices = [self.prod_to_index[p] for p in g_use.primitives if p.tp.returns() == hole.tp]
        # we gotta include variables too
        indices += [self.prod_to_index[Index(i)] for i,tp in enumerate(hole.ctx_tps) if tp == hole.tp]

        if hole.depth() >= max_depth:
            # force leaf (ie an index or a non-arrow primitive)
            indices = [i for i in indices if self.index_to_prod[i].isIndex or not self.index_to_prod[i].tp.isArrow()]

        mask = torch.zeros(len(self.prod_to_index),device=sing.device)
        mask[indices] = 1.
        mask = mask.log()
        return mask

    def trace_and_process_holes(self, root):
        """
        """
        assert root.ntype.output
        root.hide(recursive=True)

        processed_holes = []
        masks = []
        targets = []
        strings = []

        while True:
            hole = root.get_hole(sing.cfg.model.ordering,sing.cfg.model.tiebreaking)
            if hole is None:
                return processed_holes, masks, targets, strings
            masks.append(self.build_mask(hole,sing.cfg.data.max_depth))
            targets.append(self.build_target(hole))
            strings.append(str(root))
            processed_holes.append(self.process_hole(hole))
            hole.unhide()
        assert False
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
        cache_mode = None
        super().__init__(cache_mode)
        H = self.H

        inshape = H*3
        self.output = nn.Sequential(
                nn.Linear(inshape, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.prod_to_index) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')

        print("num of params in rnn policy model", count_parameters(self))

    def process_hole(self,hole):
        return str(hole.root())

    def unmasked_distributions(self, processed_holes, in_feats, out_feats):
        num_sks = len(processed_holes)
        if num_sks > 1:
            assert processed_holes[0] != processed_holes[1] # sanity check that we arent messing with the inplace operation and making everything the same by accident
        # assert all(h.task is processed_holes[0].task for h in processed_holes), "we assume everyone has the same task"

        # sk features
        sk_feats = sing.model.program_rnn.encode_sketches(processed_holes)

        # input feats
        in_feats = reduce(in_feats, 'sks exs H -> sks H', 'mean')
        if sing.cfg.debug.zero_input_feats:
            in_feats = torch.zeros_like(in_feats)

        # output feats
        out_feats = reduce(out_feats, 'sks exs H -> sks H', 'mean')
        if sing.cfg.debug.zero_output_feats:
            out_feats = torch.zeros_like(out_feats)
        
        task_feats = torch.cat((in_feats,out_feats)) # [H*2]
        task_feats = task_feats.expand(num_sks, -1)

        assert task_feats.dim() == sk_feats.dim() == 2

        input = torch.cat((sk_feats,task_feats),dim=1) # [num_sks,H] `cat dim=1` [num_sks,H*2] -> [num_sks,H*3]
        res = self.output(input)
        return res

    


ReplProcessedHole = namedtuple('ReplProcessedHole','sk_rep ctx_rep')
class ListREPLPolicyHead(PolicyHead):
    def __init__(self):
        cache_mode = None if sing.cfg.model.multidir else 'parents'
        super().__init__(cache_mode)
        H = self.H

        self.output = nn.Sequential(
                nn.Linear(H*2, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.prod_to_index) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')

        print(f"num of params in {self.__class__.__name__} policy model", count_parameters(self))

    def process_hole(self,hole):
        """
        Process hole always returns something that will not be affected by changes to
        the tree that it came from

        Take a hole (pnode) and return an embedding of it along with an embedding of its context
        Note we call .get_abstract() because we dont want to return a closure that may depend on
        the tree still
        """

        try:
            hole.needs_ctx = None # mark that we need the ctx
            if sing.cfg.model.multidir:
                # MBAS
                sk_rep = hole.embed_from_above().get_abstract()
            else:
                # BAS
                sk_rep =  hole.root().beval(None).get_abstract()
            
            # could make less hacky lol
            ctx = hole.needs_ctx
            if ctx is None:
                ctx = hole.pnode_cache.ctx
                assert ctx is not None
            ctx_rep = ctx.encode()
            return ReplProcessedHole(sk_rep,ctx_rep) 

        finally:
            del hole.needs_ctx
        #return hole.root().propagate_upward()

    def unmasked_distributions(self, processed_holes, in_feats, out_feats):
        """
        """
        # assert all(h.task is processed_holes[0].task for h in processed_holes), "we assume everyone has the same task"
        num_sks = len(processed_holes)
        # stack and possibly zero out sketches
        sk_reps = torch.stack([p.sk_rep for p in processed_holes]) # [num_sks,num_exs,H]
        ctx_reps = torch.stack([p.ctx_rep for p in processed_holes]) # [num_sks,num_exs,H]

        if sing.cfg.debug.zero_sk:
            sk_reps = torch.zeros_like(sk_reps)

        if sing.cfg.model.multidir:
            # MBAS
            input = torch.cat([sk_reps,ctx_reps],dim=-1)
            input = input.max(1).values # max over examples to yield [num_sks,H]
            return self.output(input)
        else:
            # BAS
            if sing.cfg.debug.zero_output_feats:
                out_feats = torch.zeros_like(out_feats)
            
            compared = sing.model.abstract_comparer(sk_reps,out_feats) # [num_sketches,num_exs,H]
            # input = torch.cat([compared,ctx_reps],dim=-1)
            input = rearrange([compared,ctx_reps], 'list sks exs H -> sks exs (list H)') # cat along hidden dim
            input = reduce(input, 'sks exs H2 -> sks H2', 'max')
            # input = input.max(1).values # max over examples to yield [num_sks,H]
            return self.output(input)



