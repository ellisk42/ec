
from dreamcoder.program import Program
from dreamcoder.task import Task
from torch import nn
import torch
import numpy as np
from dreamcoder.domains.list.makeDeepcoderData import InvalidSketchError

from dreamcoder.matt.sing import sing
from dreamcoder.matt.util import *

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
        assert len(task.examples) == sing.num_exs
        self.outputs = Examplewise([o for i,o in task.examples])
        _argcs = [len(i) for i,o in task.examples]
        self.argc = _argcs[0]
        assert all(argc == self.argc for argc in _argcs), "not all io examples have the same argc"
        # self.inputs takes a little more effort since each input is a tuple of values and we need to list(zip(*)) it
        _exwise = [i for i,o in task.examples] # outer list = examples, inner list = args
        _argwise = list(zip(*_exwise)) # outer list = args, inner list = examples
        assert len(_argwise) == self.argc
        self.inputs = [Examplewise(arg) for arg in _argwise]
    def output_features(self):
        return self.outputs.abstract
    def input_features(self):
        return sing.model.abstraction_fn.encode_known_ctx(self.inputs)

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
            assert data.shape[0] == sing.num_exs
        else:
            assert isinstance(data,(list,tuple))
            self.concrete = data
            assert len(data) == sing.num_exs
            self._check_in_range()

    @property
    def abstract(self):
        if self._abstract is None:
            self._abstract = sing.model.abstraction_fn.encode_exwise(self)
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
        for ex in range(sing.num_exs):
            results.append(uncurry(fn,[arg.concrete[ex] for arg in args])) # makes sense bc fn() takes len(args) arguments
        return Examplewise(results)

    def _check_in_range(self):
        """
        check if an int, [int], or [[int]] has all values within the range [-V,V]
        Raises an InvalidSketchError if this is not the case
        """
        if self.concrete is None:
            return
        V = sing.cfg.data.V

        for val in self.concrete:
            if isinstance(val,(int,np.integer)):
                if val > V or val < -V:
                    raise InvalidSketchError(f"integer outside a list had value {val}")
                return
            elif isinstance(val,(list,tuple)):
                if len(val) == 0:
                    return
                assert isinstance(val[0],(int,np.integer))
                maxx = max(val)
                minn = min(val)
                if maxx > V or minn < -V:
                    raise InvalidSketchError(f"min={minn} max={maxx}")
                return
            elif callable(val):
                return # callables are always fine
            else:
                raise NotImplementedError(f"Unrecognized value: {val}")
    def __repr__(self):
        if self.concrete is not None:
            return repr(self.concrete)
        assert self._abstract is not None
        return repr(self.abstract) # wont accidentally trigger lazy compute bc we alreayd made sure _abstract was set


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
    def __init__(self, p: Program, parent, ctx:list, from_task=None) -> None:
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
        self.in_HOF_lambda = (None in ctx)

        if from_task is not None:
            # OUTPUT
            self.tree = PNode(p, parent=self, ctx=ctx)
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
            exwise = self.task.inputs[which_toplevel_arg] if len(self.task.inputs) > which_toplevel_arg else None

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
            fn,xs = p.applicationParse()
            self.xs = [PNode(x,parent=self, ctx=ctx) for x in xs]
            self.fn = PNode(fn, parent=self, ctx=ctx) # an abstraction or a primitive
            if self.fn.isAbstraction:
                # this never happens, but if we wanted to cover $0 $1 etc properly for
                # it we could do so easily: just modify the (mutable) 
                # self.f.ctx[0].val = xs[0]
                # self.f.ctx[1].val = xs[1]
                # etc.
                assert False 
            self.isApplication = True
        else:
            assert False
    
    def __repr__(self):
        return repr(self.p)
    def upward_only_embedding(self):
        assert self.isOutput
        return self.propagate(None)
    def propagate(self,towards, concrete_only=False):
        """
        returns :: Examplewise

        Evaluates this subtree relative to the PNode towards.
        set towards=None and call this on the isOutput node if you want to eval the whole tree with BAS semantics

        Note the towards field can be safely ignored by all leaves of the tree since theyre always propagated
        towards their parent

        With concrete_only turned on this doesnt call any neural networks it just returns None
        in those cases. So if your program has holes then evaluating it will definitely
        return None. Why is this useful? It's good for the InvalidIntermediatesValueHead
        where you want to pull out all the concrete portions of trees and evaluate them so
        that you construct Examplewises (which automatically throw InvalidSketch errors at
        invalid values). I implemented that in here instead of as a traversal bc
        it's fairly straightforward and it guarantees that itll follow propagation semantics
        precisely
        


        """
        if self.isOutput:
            if towards is not None:
                # propagate downward
                return self.task.outputs
            else:
                # propagate upwards: evaluate the whole tree
                return self.tree.propagate(self,concrete_only=concrete_only)

        elif self.isPrimitive:
            return Examplewise([self.value for _ in range(sing.num_exs)])

        elif self.isIndex:
            # if the index is bound to something return that
            # otherwise return self.index_nm[self.i]()
            exwise = self.ctx[self.i]
            if exwise is not None:
                # this branch is taken for all toplevel args
                return exwise
            # this branch is taken for all lambdas that arent actually applied (ie theyre HOF inputs)
            assert self.in_HOF_lambda
            if concrete_only: return None
            return Examplewise(sing.model.abstract_transformers.lambda_index_nms[self.i]().expand(sing.num_exs,-1))

        elif self.isHole:
            if concrete_only: return None
            if self.in_HOF_lambda:
                # contextless hole as in BAS
                return Examplewise(sing.model.abstract_transformers.lambda_hole_nms[self.tp.show(True)]().expand(sing.num_exs,-1))
            # not in lambda
            ctx = sing.model.abstraction_fn.encode_known_ctx(self.ctx)
            return  Examplewise(sing.model.abstract_transformers.hole_nms[self.tp.show(True)](ctx))

        elif self.isAbstraction:
            # in terms of evaluation, abstractions do nothing but pass along data
            if towards is self.parent: # evaluate self.body
                # this handles the case where our parent is an HOF who's about to do concrete application
                # and we're the HOF so it needs us to return a python lambda that it can feed to its HOF
                # primitive function. We use execute_single() to get that python lambda version of ourselves
                if self.body.in_HOF_lambda and not self.hasHoles and sing.cfg.model.allow_concrete_eval:
                    assert self.parent.isApplication
                    if not self.parent.hasHoles:
                        fn = self.execute_single([])
                        return Examplewise([fn for _ in range(sing.num_exs)])
                return self.body.propagate(self,concrete_only=concrete_only)
            elif towards is self.body: # evaluate self.parent
                return self.parent.propagate(self,concrete_only=concrete_only)
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
            evaluated_args = [node.propagate(self,concrete_only=concrete_only) for node in possible_args if node is not towards]

            if concrete_only and None in evaluated_args: return None # one of our chilren was abstract

            if towards is self.parent and all(arg.concrete is not None for arg in evaluated_args) and (sing.cfg.model.pnode.allow_concrete_eval or concrete_only):
                ## CONCRETE EVAL
                # calls evaluate() on self.fn which should return a concrete callable primitive
                # wrapped in an Examplewise then we just use Examplewise.as_concrete_function
                # to apply it to other examplewise arguments
                if not concrete_only: sing.stats.fn_called_concretely += 1
                return self.fn.propagate(self, concrete_only=concrete_only).as_concrete_fn(*evaluated_args)
            ## ABSTRACT EVAL
            assert not concrete_only # the earlier check would have caught this
            sing.stats.fn_called_abstractly += 1
            nm = sing.model.abstract_transformers.fn_nms[self.fn.name][propagation_direction]
            return Examplewise(nm(*[arg.abstract for arg in evaluated_args]))

        else:
            raise TypeError
    def execute_single(self,ctx_single):
        """
        if you call this on an abstraction youll get out a callable version of it. If you call
        this on something fully concrete youll get the actual value.
        
        you can call this on isOutput and itll just return the result of calling it on 
        its child, so thats safe.

        It only goes upward, everything must be concrete, and it works over
        individual values instead of Examplewise sets of values.

        Calling code should always set ctx_single to `[]` because it should
        only ever get populated by recursive calls within this function (ie
        when Abstractions are encountered).

        """
        if self.isOutput:
            assert ctx_single == [] # why would you ever pass in anything else at top level
            return self.tree.execute_single(ctx_single)

        elif self.isPrimitive:
            return self.value

        elif self.isIndex:
            return ctx_single[self.i]

        elif self.isHole:
            assert False

        elif self.isAbstraction:
            return lambda x: self.body.execute_single([x] + ctx_single)

        elif self.isApplication:
            # execute self, execute args, apply self to args
            return uncurry(self.fn.execute_single(ctx_single), [x.execute_single(ctx_single) for x in self.xs])

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
        if self.isOutput:
            self.tree.traverse(fn)
        if self.isAbstraction:
            self.body.traverse(fn)
        elif self.isApplication:
            self.fn.traverse(fn)
            for x in self.xs:
                x.traverse(fn)
        elif self.isIndex or self.isHole or self.isPrimitive:
            pass
        else:
            raise TypeError
    def size(self):
        sz = 0
        def _size(node):
            nonlocal sz
            if not node.isAbstraction and not node.isOutput:
                sz += 1
        self.traverse(_size)
        return sz
    # def has_unset_index(self):
    #     found = False
    #     def finder(node):
    #         nonlocal found
    #         if node.isIndex and node.ctx[node.i].is_unset:
    #             found = True
    #     self.traverse(finder)
    #     return found
    # def unwrap_abstractions(self):
    #     """
    #     traverse down .body attributes until you get to the first non-abstraction PNode
    #     """
    #     node = self
    #     i = 0
    #     while node.isAbstraction:
    #         node = node.body
    #         i += 1
    #     return node,i
