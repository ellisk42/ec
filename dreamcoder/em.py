
from dreamcoder.domains.list.makeDeepcoderData import check_in_range
from dreamcoder.program import Program
from dreamcoder.domains.misc.deepcoderPrimitives import int_to_int, int_to_bool, int_to_int_to_int
from dreamcoder.task import Task
from dreamcoder.type import tlist,tbool,tint
from torch import nn
from typing import Union
import functools
import torch
import mlb

import dreamcoder.matt.sing as sing


# class Asn:
#     def __init__(self) -> None:
#         super().__init__()
#         self.val = None
#     @property
#     def is_set(self):
#         return self.val is None


class PTask:
    """
    My version of the Task.
        .task :: Task = the original task object
        .request = the request
        .argc :: number of toplevel input arguments
        .outputs :: Examplewise
        .inputs :: [Examplewise] where len(.inputs) == .argc
    CAREFUL:
        You should only have one PTask instances per program, dont same the same one among multiple programs.
        This is because the ._neural fields of the inputs and outputs shouldnt be shared
        Or well wait actually would it be okay or even good to share them lol
    """
    def __init__(self, task) -> None:
        super().__init__()
        self.task = task
        self.request = task.request
        self.num_exs = len(task.examples)
        self.outputs = Examplewise([o for i,o in task.examples])
        _argcs = [len(i) for i,o in task.examples]
        self.argc = _argcs[0]
        assert all(argc == self.argc for argc in _argcs), "not all io examples have the same argc"
        # self.inputs takes a little more effort since each input is a tuple of values and we need to list(zip(*)) it
        _exwise = [i for i,o in task.examples] # outer list = examples, inner list = args
        _argwise = list(zip(*_exwise)) # outer list = args, inner list = examples
        assert len(_argwise) == self.argc
        self.inputs = [Examplewise(arg) for arg in _argwise]
    def reset_encodings(self):
        self.outputs._neural = None
        for input in self.inputs:
            input._neural = None

class Val:
    def __init__(self,val) -> None:
        super().__init__()
        self.val = val

# TODO I duplicated this from valuehead.py so delete that other copy
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

class ExecutionModule(nn.Module):
    def __init__(self, cfg, g, encoder, max_index) -> None:
        """
        g: Grammar
        encoder: feature extractor

        """
        super().__init__()
        self.cfg = cfg
        self.num_exs = cfg.data.train.N
        # encoder
        self.encoder = encoder # very important so the params are in the optimizer

        # hole NMs
        possible_hole_tps = [tint,tbool,tlist(tint)]
        if not cfg.data.train.expressive_lambdas:
            possible_hole_tps += [int_to_int, int_to_bool, int_to_int_to_int]
        self.hole_nms = nn.ModuleDict()
        for tp in possible_hole_tps:
            self.hole_nms[tp.show(True)] = NM(1, cfg.model.H)


        # populate fnModules
        self.fn_nms = nn.ModuleDict()
        for p in g.primitives:
            argc = len(p.tp.functionArguments())
            self.fn_nms[p.name] = nn.ModuleList([NM(argc, cfg.model.H) for _ in range(argc)+1])

        self.index_nm = NM(0, cfg.model.H)
        # TODO in the future to allow for >1 toplevel arg the above could be replaced with:
        # self.index_nms = [NM(0,cfg.model.H) for _ in range(max_index+1)]

        # TODO this is kept the same as the BAS paper however is totally worth experimenting with in the future
        # (we'd like to improve how lambdas get encoded)
        nargs = 1 if self.cfg.model.ctxful_lambdas else 0
        self.lambda_index_nms = nn.ModuleList([NM(nargs,cfg.model.H) for _ in range(2)])
        self.lambda_hole_nms = nn.ModuleDict()
        for tp in [tint,tbool]:
            self.lambda_hole_nms[tp.show(True)] = NM(nargs, cfg.model.H)



    def encode_val(self,val):
        """
        This gets called by Examplewise.encode() on its .data field
        list[len=num_exs] -> Tensor[num_exs,H]
        uses self.encoder
        """
        return self.encoder.encodeValue(val)
    def encode_ctx(self,ctx):
        """
        Called to generate the input that a hole network will take.
            (maybe an index network too if you make it take the ctx)

        [Asn] where Asn.val is Examplewise -> Tensor[num_exs,H]
            notice the num_exs bit bc the ctx should look diff for each example
        uses self.encoder
        """
        raise NotImplementedError
        for asn in ctx:
            if asn is None:
                raise NotImplementedError
        vectors = [asn.val.encode() for asn in ctx]

        lex = self.encoder.lexicon_embedder

        start = lex(lex.ctx_start).expand(1,self.num_exs,-1)
        end = lex(lex.ctx_end).expand(1,self.num_exs,-1)
        ctx = torch.cat([start] + [vec.unsqueeze(0) for vec in vectors] + [end])
        _, res = self.encoder.ctx_encoder(ctx)
        res = res.sum(0) # sum bidir
        return res



class Examplewise:
    """
    The basic object that gets passed around to represent a value (neural or concrete)
        .data : Tensor[num_examples,H] | [values] where len([values]) == num_examples
            basically this is the concrete value if it exists and the neural value otherwise
        .encode() : Tensor[num_examples.H]
            returns .data, encoding it with Examplewise._encoder to a Tensor if it's not already
            one. Note this doesnt change .data inplace into a Tensor so the concrete value will
            still be there. This function is smart about caching so if you've called .encode()
            before itll just return you the same Tensor result as before, so no worries about
            overusing it
    """
    def __init__(self,data) -> None:
        super().__init__()
        self.concrete = self._abstract = None

        if torch.is_tensor(data):
            assert data.dim() == 2
            self._abstract = data
            self.num_exs = data.shape[0]
        else:
            assert isinstance(data,(list,tuple))
            self.concrete = data
            self.num_exs = len(data)
            check_in_range(self.data,sing.em.cfg.data.test.V) # may raise InvalidSketchError

    @property
    def abstract(self):
        if self._abstract is None:
            self._abstract = sing.em.encode_val(self.concrete)
        return self._abstract

    def as_concrete_fn(self,*args):
        """
        assume self.data is just num_exs copies of the same function.
        and args are Examplewise instances with concrete values
        then just apply the function to the values
        """
        assert self.concrete is not None and callable(self.concrete[0]) and [x==self.concrete[0] for x in self.concrete]
        assert all(isinstance(arg,Examplewise) and arg.concrete is not None for arg in args)

        fn = self.concrete[0]

        results = []
        for ex in range(self.num_exs):
            results.append(fn(*[arg.concrete[ex] for arg in args])) # makes sense bc fn() takes len(args) arguments
        return Examplewise(results)



class PNode:
    """

    PNodes basically look like the Tree structure you'd expect instead
    of the wildness of dreamcoder.program.Program where you have to do
    applicationParse() etc.

    pnode.p gives you the Progrma used to generate this PNode which is
    useful e.g. if you wanna concretely execute the whole subtree

    lambda abstractions are still used bc many things operating over
    PNodes will want to be aware that the Context is changing

    pnode.ctx is a stack representing the context, it gets updated
    on entering an abstraction's body. 


    create a new one of these from a program like so:
        node = PNode(p=some_program, parent=PNode(PTask(task)), ctx=[])
    then run it with
        node.evaluate(output_node=None)

    """
    def __init__(self, p: Program, parent:PNode, ctx:list, from_task=None) -> None:
        super().__init__()

        if from_task is not None:
            # create a full program from the top level task and program
            # meaning this will be our isOutput node
            assert parent is None
            assert p is not None
            assert isinstance(from_task,Task)
            self.task = PTask(from_task)
        else:
            # continue building new nodes off an existing task/program
            assert parent is not None
            self.task = parent.task

        self.parent = parent
        self.p = p
        self.ctx = ctx
        self.isPrimitive = self.isIndex = self.isHole = self.isAbstraction = self.isApplication = self.isOutput = False
        self.hasHoles = p.hasHoles

        if from_task is not None:
            # OUTPUT
            self.tree = PNode(p, parent=self)
            self.isOutput = True
        elif p.isPrimitive:
            # PRIM
            self.name = p.name
            self.value = p.value
            self.isPrimitive = True
        elif p.isIndex:
            # IDX
            self.i = p.i
            assert self.i < len(self.ctx)
            self.isIndex = True
        elif p.isHole:
            # HOLE
            self.tp = p.tp
            self.isHole = True
        elif p.isAbstraction:
            # LAMBDA
            which_toplevel_arg = len(ctx) # ie if we're just entering the first level of lambda this will be 0

            # now if this arg is one of our inputs we grab that input, otherwise we just return None
            # (so for non toplevel lambdas this is always None)
            exwise = self.task.inputs[which_toplevel_arg] if len(self.task.inputs) >= which_toplevel_arg else None

            # push it onto the ctx stack
            self.body = PNode(p.body, parent=self, ctx=[exwise]+ctx)

            # do some sanity checks
            if self.parent.isOutput:
                inner_node,num_args = self.unwrap_abstractions()
                assert self.task.argc == num_args, "task argc doesnt match number of toplevel abstraction"
                assert all(exwise is not None for exwise in inner_node.ctx), "toplevel ctx was somehow not populated"

            self.isAbstraction = True
        elif p.isApplication:
            # FN APPLICATION
            f,xs = p.applicationParse()
            self.xs = [PNode(x,parent=self, ctx=ctx) for x in xs]
            self.f = PNode(f, parent=self, ctx=ctx) # an abstraction or a primitive
            if self.f.isAbstraction:
                # this never happens, but if we wanted to cover $0 $1 etc properly for
                # it we could do so easily: just modify the (mutable) 
                # self.f.ctx[0].val = xs[0]
                # self.f.ctx[1].val = xs[1]
                # etc.
                assert False 
            self.isApplication = True
        else:
            assert False
    
    def propagate(self,towards:PNode):
        """
        returns :: Examplewise

        Evaluates this subtree relative to the PNode towards.
        set towards=None and call this on the isOutput node if you want to eval the whole tree with BAS semantics

        Note the towards field can be safely ignored by all leaves of the tree since theyre always propagated
        towards their parent

        Prim -> do concrete eval
        Index -> 


        """
        if self.isOutput:
            if towards is not None:
                # propagate downward
                return self.task.outputs
            else:
                # propagate upwards: evaluate the whole tree
                return self.tree.propagate(self)

        # if towards is self.parent and not self.hasHoles and self.allow_concrete_eval and not self.has_unset_index():
        #     # if parent wants our output
        #     # and we have no holes
        #     # and concrete eval is turned on
        #     # and all variables in the context are set
        #     # then just evaluate it concretely
        #     raise NotImplementedError
        elif self.isPrimitive:
            return Examplewise([self.value for _ in range(self.task.num_exs)])

        elif self.isIndex:
            # if the index is bound to something return that
            # otherwise return self.index_nm[self.i]()
            exwise = self.ctx[self.i]
            if exwise is not None:
                # this branch is taken for all toplevel args
                return exwise
            # this branch is taken for all lambdas that arent actually applied (ie theyre HOF inputs)
            return Examplewise(sing.em.lambda_index_nms[self.i]().expand(self.task.num_exs,-1))

        elif self.isHole:
            ctx = sing.em.encode_ctx(self.ctx) # TODO. Note we want a [num_exs,H] vector out
            return  Examplewise(sing.em.hole_nms[self.tp.show(True)](ctx))

        elif self.isAbstraction:
            # in terms of evaluation, abstractions do nothing but pass along data
            if towards is self.parent: # evaluate self.body
                return self.body.propagate(self)
            elif towards is self.body: # evaluate self.parent
                return self.parent.propagate(self)
            else:
                raise ValueError

        elif self.isApplication:
            possible_args = [self.parent, *self.xs]
            mask =  [x is towards for x in possible_args]
            assert sum(mask) == 1
            propagation_direction = mask.index(True)
            # propagation_direction == 0 is how computation normally happens (output is upward)
            # propagation_direction == 1 is when the output is arg0, ==2 is when output is arg1, etc

            assert self.fn.isPrimitive, "Limitation, was there in old abstract repl too, can improve upon when it matters"
            evaluated_args = [node.propagate(self) for node in possible_args if node is not towards]
            if towards is self.parent and all(arg.concrete is not None for arg in evaluated_args) and self.allow_concrete_eval:
                ## CONCRETE EVAL
                # calls evaluate() on self.fn which should return a concrete callable primitive
                # wrapped in an Examplewise then we just use Examplewise.as_concrete_function
                # to apply it to other examplewise arguments
                return self.fn.evaluate().as_concrete_fn(*evaluated_args)
            ## ABSTRACT EVAL
            nm = sing.em.fn_nms[self.fn.name][propagation_direction]
            return nm(*[arg.abstract for arg in evaluated_args])

        else:
            raise TypeError
    def traverse(self,fn):
        """
        simple helper that just calls fn() on every PNode in the tree.
        Doesn't return anything. If you want to return something have
        your fn maintain some state (eg a class with __call__ or 
        just a `nonlocal` variable closured in)
        """
        fn(self)
        if self.isAbstraction:
            fn(self.body)
        elif self.isApplication:
            fn(self.f)
            for x in self.xs:
                fn(x)
    def has_unset_index(self):
        found = False
        def finder(node):
            nonlocal found
            if node.isIndex and node.ctx[node.i].is_unset:
                found = True
        self.traverse(finder)
        return found
    def unwrap_abstractions(self):
        """
        traverse down .body attributes until you get to the first non-abstraction PNode
        """
        node = self
        i = 0
        while node.isAbstraction:
            node = node.body
            i += 1
        return node,i



def range_check(fn):
    """
    decorator that checks the output of a function and raises an InvalidSketchError
    if the function ever outputs a value outside of the expected range
    """
    @functools.wraps(fn)
    def wrapped(self,*args, **kwargs):
        assert isinstance(self,PNode)
        res = fn(self,*args, **kwargs)
        return res
    return wrapped
