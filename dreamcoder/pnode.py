
from collections import namedtuple
import functools
from dreamcoder.program import Application, Hole, Primitive, Program
from dreamcoder.task import Task
from torch import nn
import enum
import random
from dreamcoder.program import Index
import torch
import numpy as np
from dreamcoder.domains.list.makeDeepcoderData import InvalidSketchError

from dreamcoder.matt.sing import sing
from dreamcoder.matt.util import *

Ctx = namedtuple('Ctx','tp exwise')
Cache = namedtuple('Cache','towards exwise string')
class FoundSolution(Exception):
    def __init__(self,p):
        self.p = p

def cached_propagate(propagate):
    @functools.wraps(propagate)
    def _cached_propagate(self,towards,concrete_only=False):

        if self.cache.towards is not towards or sing.cfg.debug.no_cache:
            exwise = propagate(self, towards, concrete_only=concrete_only)
            self.cache = Cache(towards,exwise,self.subtree_str())
            sing.stats.cache_not_used += 1
        else:
            sing.stats.cache_used += 1
            if sing.cfg.debug.validate_cache:
                """
                for cache validation. Note that this will actually recurse
                since when we manually call propagate here then that inner
                call will call self.propagate (which is _cached_propagate) internally so
                it'll end up verifying the entire cache since validate_cache will be true
                for all these cases.
                """
                if towards is self.parent: # if we're passing upwards and trying to use our cache, our str(self) better not have changed bc that indicates something below us changing
                    assert str(self) == self.cache.string
                exwise = propagate(self, towards, concrete_only=concrete_only)
                if not self.cache.exwise.has_abstract:
                    assert not exwise.has_abstract # programs shd be getting strictly more concrete
                if self.cache.exwise.has_abstract:
                    """
                    We're in a weird position. The caller of propagate() is the one who often then calls
                    .abstract(), so we don't actually know if we're about to be abstracted or not. However,
                    we know that if our cache did NOT have abstract then we definitely wont be abstract
                    (programs get strictly more concrete). So here by calling .abstract() below in the case
                    where the cache was abstract, we may be doing something unnecessary but thats why
                    this is a debugging thing. If the cache hasn't been invalidated then its .abstract()
                    better be the same as our .abstract(), I think that's fair enough.
                    """
                    if not (exwise.has_concrete and callable(exwise.concrete[0])): # dont try to encode callables eg with the weird rare cases of ABS
                        assert torch.allclose(self.cache.exwise.abstract(),exwise.abstract())

        return self.cache.exwise
    return _cached_propagate

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
        self._cached_inputs = None
    def output_features(self):
        return self.outputs.abstract()
    def input_features(self):
        old_cache = self._cached_inputs 
        if self._cached_inputs is None:
            self._cached_inputs = sing.model.abstraction_fn.encode_known_ctx([Ctx(None,i) for i in self.inputs]) # we do know the ctx.tp im just lazy
        if sing.cfg.debug.validate_cache and old_cache is not None:
            assert torch.allclose(self._cached_inputs,old_cache)
        return self._cached_inputs

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

    def abstract(self):
        if self._abstract is None:
            self._abstract = sing.model.abstraction_fn.encode_exwise(self)
        return self._abstract

    @property
    def has_abstract(self):
        return self._abstract is not None

    @property
    def has_concrete(self):
        return self.concrete is not None

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
        return repr(self.abstract()) # wont accidentally trigger lazy compute bc we alreayd made sure _abstract was set


class PNode:
    """

    PNodes basically look like the Tree structure you'd expect instead
    of the wildness of dreamcoder.program.Program where you have to do
    applicationParse() etc.

    lambda abstractions are still used bc many things operating over
    PNodes will want to be aware that the Context is changing

    pnode.ctx is a stack representing the context, it gets updated
    on entering an abstraction's body. 


    create a new one of these from a program like so:
        node = PNode(p=some_program, parent=PNode(PTask(task)), ctx=[])
    then run it with
        node.evaluate(output_node=None)

    """
    
    def __init__(self, ntype, tp, parent, ctx:list):
        super().__init__()

        if isinstance(parent,PTask):
            self.task = parent
            self.parent = self # root node. We dont use None as the parent bc we want towards=None to be reserved for our Cache emptiness
        else:
            self.task = parent.task
            self.parent = parent

        self.ntype = ntype
        self.phantom_ntype = None
        self.tp = tp
        self.ctx = ctx

        self.cache = Cache(None, None, None) # can't use `None` as the cleared cache since that's a legit direciton for the output node
    def root_str(self):
        """
        Useful for when you wanna ensure that the whole program hasnt been inplace
        modified somehow over some period
        """
        return str(self.root())
    def subtree_str(self):
        """
        Useful for when you wanna ensure that the subtree below this hasnt been modified
        over some period
        """
        return str(self)
    def marked_str(self,):
        """
        A unique little string that shows the whole program but with `self` marked clearly with double brackets [[]]
        """
        return self.root().marked_repr(self)
    def expand_to(self, prim, cache_mode=None):
        """
        Call this on a hole to transform it inplace into something else.
         - prim :: Primitive | Index
         - If prim.tp.isArrow() this becomes an APP with the proper holes added for cildren
         - You can never specify "make an Abstraction" because those get implicitly created by build_hole()
            whenever a hole is an arrow type. Note that a hole being an arrow type is different from `prim`
            being an arrow type. An arrow prim can fill a hole that has its return type, however if the arrow
            prim futhermore has arguments that are arrows, those are where the abstractions will be created (HOFs).
         - if `self` is an abstraction it'll get unwrapped to reveal the underlying hole as a convenience (simplifies
            a lot of use cases and recursion cases).

        - use 'cache' to specify how the cache should behave. None means we dont clear the cache at all and
            just let it accumulate. Some mode like 'single' 'tree' or 'parents' will do self.clear_cache(mode).
        - the cache is very dangerous, but ofc by running in cache-verify mode where we test both the cache and noncache
            behavior and ensure theyre always the same, we're okay


        """
        if clear_cache is not None:
            self.clear_cache(clear_cache) # ok to do this here bc we wont be editing cache at all during expand_to
        self = self.unwrap_abstractions()
        assert self.ntype.hole
        assert not self.tp.isArrow(), "We should never have an arrow for a hole bc it'll instantly get replaced by abstractions with an inner hole"
        
        if prim.isIndex:
            assert self.ctx[prim.i].tp == self.tp # the Ctx.tp is the same as our tp
            self.i = prim.i
            self.ntype = NType.VAR
        elif prim.isPrimitive:
            assert prim.tp.returns() == self.tp # in order to fill the hole in a valid way
            if not prim.tp.isArrow():
                # PRIM case
                self.ntype = NType.PRIM
                self.prim = prim
                self.name = prim.name
                self.value = prim.value
            else:
                # APP case
                self.ntype = NType.APP
                # make self.fn as a PRIM 
                self.fn = PNode(NType.PRIM, tp=prim.tp, parent=self, ctx=self.ctx)
                self.fn.prim = prim
                self.fn.name = prim.name
                self.fn.value = prim.value
                # make holes for args
                self.xs = [self.build_hole(arg_tp) for arg_tp in prim.tp.functionArguments()]
        else:
            raise TypeError
        
    def build_hole(self, tp):
        """
        Make a new hole with `self` as parent (and `ctx` calculated from `self`)
        This also handles expanding into Abstractions if tp.isArrow()
        """
        if not tp.isArrow():
            return PNode(NType.HOLE, tp, parent=self, ctx=self.ctx)
        # Introduce a lambda

        which_toplevel_arg = len(self.ctx) # ie if we're just entering the first level of lambda this will be 0

        # now if this arg is one of our inputs we grab that input, otherwise we just return None
        # (so for non toplevel lambdas this is always None)
        exwise = self.task.inputs[which_toplevel_arg] if len(self.task.inputs) > which_toplevel_arg else None

        arg_tp = tp.arguments[0] # the input arg to this arrow
        res_tp = tp.arguments[1] # the return arg (which may be an arrow)

        # push it onto the ctx stack
        abs = PNode(NType.ABS, tp, parent=self, ctx=(Ctx(arg_tp,exwise),*self.ctx))
        # now make an inner hole (which may itself be an abstraction ofc)
        inner_hole = abs.build_hole(res_tp)
        abs.body = inner_hole

        return abs # and return our abstraction
    def into_hole(self, cache_mode=None):
        """
        reverts a PNode thats not a hole back into a hole. If you still want to keep around
        a ground truth non-hole version of the node you probably want PNode.hide() instead.
        """
        assert not self.ntype.hole
        assert not self.ntype.abs, "you never want an arrow shaped hole buddy"
        assert not self.ntype.output, "nonono dont be crazy"
        for attr in ('prim','name','value','fn','xs','i','body','tree'):
            if hasattr(self,attr):
                delattr(self,attr) # not super important but eh why not
        self.cache = Cache(None,None,None)
        self.ntype = NType.HOLE
    def check_solve(self):
        """
        check if we're a solution to the task
        """
        if self.has_holes:
            return False
        return self.root().propagate_upward(concrete_only=True).concrete == self.task.outputs.concrete


    @staticmethod
    def from_ptask(ptask: PTask):
        """
        Create a tree shaped like:
                PNode(OUTPUT)
                     |
                     |
                 PNode(ABS)
                     |
                     |
                 PNode(HOLE)
        
        Or however many abstractions make sense for the given ptask
        Returns the root of the tree (the output node)
        """
        root = PNode(NType.OUTPUT, tp=None, parent=ptask, ctx=())
        root.tree = root.build_hole(ptask.request)

        # do some sanity checks
        hole, num_abs = root.tree.unwrap_abstractions(count=True)
        assert ptask.argc == num_abs, "task argc doesnt match number of toplevel abstraction"
        assert all(ctx.exwise is not None for ctx in hole.ctx), "toplevel ctx was somehow not populated"
        return root

    @staticmethod
    def from_dreamcoder(p: Program, task:Task):
        """
        Given a dreamcoder Program and Task, make an equivalent PNode
        and associated PTask. Returns the root (an output node).
        """
        # create a full program from the top level task and program
        # meaning this will be our ntype.output node
        root = PNode.from_ptask(PTask(task))
        root.tree.expand_from_dreamcoder(p)
        assert repr(root.tree) == repr(p), "these must be the same for the sake of the RNN which does str() of the pnode"
        return root

    def expand_from_dreamcoder(self, p: Program):
        """
        Like expand_to() except for replacing the hole with a subtree equivalent
        to the given dreamcoder program. All abstractions in both `self` and `p`
        will be unwrapped to get at the underlying holes so it's okay to pass in
        abstractions (this simplifies recursion cases).
        """
        self = self.unwrap_abstractions()
        assert self.ntype.hole

        # unwrap abstractions
        while p.isAbstraction:
            p = p.body

        if p.isPrimitive or p.isIndex:
            # p is a Primitive or Index
            self.expand_to(p)
        elif p.isHole:
            pass # we already are a hole!
        elif p.isAbstraction:
            assert False # can't happen bc of the `while p.isAbstraction` unwrapping above
        elif p.isApplication:
            # application. We expand each hole we create (the f hole and the xs holes)
            f, xs = p.applicationParse()
            assert f.isPrimitive
            self.expand_to(f) # expand to an APP with proper fn and holes for args
            for x,x_ in zip(self.xs,xs):
                x.expand_from_dreamcoder(x_)
        else:
            raise TypeError
    
    def __repr__(self):
        if self.ntype.abs:
            return f'(lambda {self.body})'
        elif self.ntype.app:
            args = ' '.join(repr(arg) for arg in self.xs)
            return f'({self.fn} {args})'
        elif self.ntype.prim:
            return f'{self.name}'
        elif self.ntype.var:
            return f'${self.i}'
        elif self.ntype.hole:
            return f'<HOLE>'
        elif self.ntype.output:
            return f'{self.tree}'
        else:
            raise TypeError
    def marked_repr(self,marked_node):
        """
        A recursive repr() function like repr() but if it encounters 'marked_node' that node
        will be printed with [[]] brackets around it
        """
        if self.ntype.abs:
            res = f'(lambda {self.body.marked_repr(marked_node)})'
        elif self.ntype.app:
            args = ' '.join(arg.marked_repr(marked_node) for arg in self.xs)
            res = f'({self.fn.marked_repr(marked_node)} {args})'
        elif self.ntype.prim or self.ntype.var or self.ntype.hole:
            res = repr(self)
        elif self.ntype.output:
            res = self.tree.marked_repr(marked_node)
        else:
            raise TypeError
        if self is marked_node:
            return f'[[{res}]]'
        return res
    @property
    def in_HOF_lambda(self):
        return len(self.ctx) > len(self.task.inputs)
    def propagate_upward(self, concrete_only=False):
        """
        BAS-style upward-only propagation
        """
        return self.propagate(self.parent, concrete_only=concrete_only)
    def propagate_to_hole(self):
        """
        MBAS-style propagation to hole
        """
        assert self.ntype.hole
        return self.propagate(self)
    
    @cached_propagate
    def propagate(self, towards, concrete_only=False):
        """
        returns :: Examplewise

        Evaluates this subtree relative to the PNode towards.
        set towards=None and call this on the ntype.output node if you want to eval the whole tree with BAS semantics

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

        if self.ntype.output:
            if towards is self:
                # propagate upwards: evaluate the whole tree
                return self.tree.propagate(self,concrete_only=concrete_only)
            else:
                assert towards is self.tree
                # propagate downward
                return self.task.outputs

        elif self.ntype.prim:
            return Examplewise([self.value for _ in range(sing.num_exs)])

        elif self.ntype.var:
            # if the index is bound to something return that
            # otherwise return self.index_nm[self.i]()
            exwise = self.ctx[self.i].exwise
            if exwise is not None:
                # this branch is taken for all references to toplevel args
                return exwise
            # this branch is taken for all lambdas that arent actually applied (ie theyre HOF inputs)
            assert self.in_HOF_lambda
            if concrete_only: return None
            return Examplewise(sing.model.abstract_transformers.lambda_index_nms[self.i]().expand(sing.num_exs,-1))

        elif self.ntype.hole:
            if towards is self:
                return self.parent.propagate(self, concrete_only=concrete_only)
            if concrete_only: return None
            if self.in_HOF_lambda:
                # contextless hole as in BAS
                return Examplewise(sing.model.abstract_transformers.lambda_hole_nms[self.tp.show(True)]().expand(sing.num_exs,-1))
            # not in lambda
            ctx = sing.model.abstraction_fn.encode_known_ctx(self.ctx)
            return  Examplewise(sing.model.abstract_transformers.hole_nms[self.tp.show(True)](ctx))

        elif self.ntype.abs:
            # in terms of evaluation, abstractions do nothing but pass along data
            if towards is self.parent: # evaluate self.body
                # this handles the case where our parent is an HOF who's about to do concrete application
                # and we're the HOF so it needs us to return a python lambda that it can feed to its HOF
                # primitive function. We use execute_single() to get that python lambda version of ourselves
                # oh and if our parent is an abstraction too and they chose not to do execute_single() then clearly we shouldnt
                if self.body.in_HOF_lambda and not self.parent.ntype.abs and not self.has_holes and (sing.cfg.model.pnode.allow_concrete_eval or concrete_only):
                    assert self.parent.ntype.app
                    if not self.parent.has_holes:
                        fn = self.execute_single([])
                        return Examplewise([fn for _ in range(sing.num_exs)])
                return self.body.propagate(self,concrete_only=concrete_only)
            elif towards is self.body: # evaluate self.parent
                return self.parent.propagate(self,concrete_only=concrete_only)
            else:
                raise ValueError

        elif self.ntype.app:
            possible_args = [self.parent, *self.xs]
            mask =  [x is towards for x in possible_args]
            assert sum(mask) == 1
            propagation_direction = mask.index(True)
            # propagation_direction == 0 is how computation normally happens (output is upward)
            # propagation_direction == 1 is when the output is arg0, ==2 is when output is arg1, etc

            assert self.fn.ntype.prim, "Limitation, was there in old abstract repl too, can improve upon when it matters"
            evaluated_args = [node.propagate(self,concrete_only=concrete_only) for node in possible_args if node is not towards]

            if concrete_only and None in evaluated_args: return None # one of our chilren was abstract

            if towards is self.parent and all(arg.has_concrete for arg in evaluated_args) and (sing.cfg.model.pnode.allow_concrete_eval or concrete_only):
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
            return Examplewise(nm(*[arg.abstract() for arg in evaluated_args]))

        else:
            raise TypeError
    def execute_single(self,ctx_single):
        """
        if you call this on an abstraction youll get out a callable version of it. If you call
        this on something fully concrete youll get the actual value.
        
        you can call this on ntype.output and itll just return the result of calling it on 
        its child, so thats safe.

        It only goes upward, everything must be concrete, and it works over
        individual values instead of Examplewise sets of values.

        Calling code should always set ctx_single to `[]` because it should
        only ever get populated by recursive calls within this function (ie
        when Abstractions are encountered).

        """
        if self.ntype.output:
            assert ctx_single == [] # why would you ever pass in anything else at top level
            return self.tree.execute_single(ctx_single)

        elif self.ntype.prim:
            return self.value

        elif self.ntype.var:
            return ctx_single[self.i]

        elif self.ntype.hole:
            assert False

        elif self.ntype.abs:
            return lambda x: self.body.execute_single([x] + ctx_single)

        elif self.ntype.app:
            # execute self, execute args, apply self to args
            return uncurry(self.fn.execute_single(ctx_single), [x.execute_single(ctx_single) for x in self.xs])

        else:
            raise TypeError
    def size(self):
        """
        gets size of tree below this node
        """
        if self.ntype.output:
            return self.tree.size() # no cost
        elif self.ntype.abs:
            return self.body.size() # no cost
        elif self.ntype.app:
            return self.fn.size() + sum(x.size() for x in self.xs) # sum of fn and arg sizes
        elif self.ntype.var or self.ntype.hole or self.ntype.prim:
            return 1 # base case
        else:
            raise TypeError
    def depth_of_node(self):
        """
        gets depth of this node below the output node
        """
        if self.ntype.output:
            return 0 # zero cost when it's the output node
        elif self.ntype.abs:
            return self.parent.depth_of_node() # no cost
        elif self.ntype.var or self.ntype.hole or self.ntype.prim or self.ntype.app:
            return self.parent.depth_of_node() + 1 # parent depth + 1
        else:
            raise TypeError
    @property
    def has_holes(self):
        """
        check if we have any holes
        """
        if self.ntype.output:
            return self.tree.has_holes
        elif self.ntype.abs:
            return self.body.has_holes
        elif self.ntype.var or self.ntype.prim:
            return False
        elif self.ntype.hole:
            return True
        elif self.ntype.app:
            return self.fn.has_holes or any(x.has_holes for x in self.xs)
        else:
            raise TypeError
    def depth(self):
        """
        gets depth of tree below this node
        """
        if self.ntype.output:
            return self.tree.depth() # no cost
        elif self.ntype.abs:
            return self.body.depth() # no cost
        elif self.ntype.var or self.ntype.hole or self.ntype.prim:
            return 1 # base case
        elif self.ntype.app:
            return max([x.depth() for x in (*self.xs,self.fn)]) # max among fn and args
        else:
            raise TypeError
    def get_hole(self, ordering, tiebreaking):
        """
        returns a single hole or None if there are no holes in the subtree
        """
        if self.ntype.output:
            return self.tree.get_hole(ordering, tiebreaking)
        elif self.ntype.abs:
            return self.body.get_hole(ordering, tiebreaking)
        elif self.ntype.var or self.ntype.prim:
            return None
        elif self.ntype.hole:
            return self
        elif self.ntype.app:
            holes = [self.fn.get_hole(ordering, tiebreaking)]+ [x.get_hole(ordering,tiebreaking) for x in self.xs]
            holes = [h for h in holes if h is not None]
            if len(holes) == 0:
                return None

            options = {
                'left': holes[0],
                'right': holes[-1],
                'random': random.choice(holes),
            }

            # common cases
            if ordering in options:
                return options[ordering]


            # check for a depth based tie in which case use tiebreaker
            depths = [h.depth_of_node for h in holes]
            if all(depth==depths[0] for depth in depths):
                return options[tiebreaker]

            # normal depth based ones
            if ordering == 'deep':
                return max(holes, key=lambda h: h.depth_of_node())
            if ordering == 'shallow':
                return max(holes, key=lambda h: -h.depth_of_node())

            raise ValueError(ordering)
        else:
            raise TypeError
    def cache_friendly_clone(self):
        raise NotImplementedError
    def children(self):
        """
        returns a list of any nodes immediately below this one in the tree (empty list if leaf)
        note that this doesnt recursively get everyone, just your immediate children.
        """
        if self.ntype.output:
            return [self.tree]
        elif self.ntype.abs:
            return [self.body]
        elif self.ntype.app:
            return [self.fn,*self.xs]
        elif self.ntype.var or self.ntype.hole or self.ntype.prim:
            return []
        else:
            raise TypeError
    def get_prod(self):
        self = self.unwrap_abstractions()

        ntype = self.ntype
        if ntype.hole and self.phantom_ntype is not None:
            ntype = self.phantom_ntype

        if ntype.output:
            raise TypeError
        elif ntype.hole:
            raise TypeError
        elif ntype.app:
            return self.fn.get_prod()
        elif ntype.prim:
            return self.prim
        elif ntype.var:
            return Index(self.i)
        elif ntype.abs:
            assert False, "not possible bc we unwrapped abstractions"
        else:
            raise TypeError
    def unwrap_abstractions(self, count=False):
        """
        traverse down .body attributes until you get to the first non-abstraction PNode.
        """
        node = self
        i = 0
        while node.ntype.abs:
            node = node.body
            i += 1
        if count:
            return node,i
        return node
    def root(self):
        """
        get the root of the tree (an output node)
        """
        if self.parent is self:
            return self
        return self.parent.root()
    def clear_cache(self, mode):
        """
        Clear your own cache and if the mode is...
            - single: nobody, just your own
            - parents: you, your parent, and all their parents to the root
            - children: you, all your children, and all their children recursively
            - tree: the whole tree from the .root() downward
        """
        if mode is None:
            return
        self.cache = Cache(None,None,None)
        sing.stats.cache_cleared += 1
        if mode == 'single':
            pass # nothing more to od
        elif mode == 'parents':
            if self.parent is not self: # ie not root
                self.parent.clear_cache('parents')
        elif mode == 'children':
            for c in self.children():
                c.clear_cache('children')
        elif mode == 'tree':
            self.root().clear_cache('children')
        else:
            raise ValueError
    def hide(self,recursive=False):
        """
        Turn whole tree into Holes but without destroying the old data
        But dont do this for abstractions + output nodes bc it's impossible to "guess" these
        and theyre autofilled during search.
        Also for applications we dont hide the .fn field (which we assume to be a primitive)
        """
        if recursive:
            if self.ntype.output:
                self.tree.hide(recursive=True)
            if self.ntype.abs:
                self.body.hide(recursive=True)
            if self.ntype.app:
                for x in self.xs:
                    x.hide(recursive=True)
                # dont hide the self.fn
                assert self.fn.ntype.prim, "i imagine hiding will change if we had applications where the fn wasnt a prim"

        if self.ntype.output or self.ntype.abs:
            pass # we cant hide an output or abs, they just recurse if recursive=True
        elif self.ntype.prim and self.parent.ntype.app:
            pass # to hide an APP we hide its args and the APP node itself but not the fn (which should be revealed when the app node is unhidden)
        elif self.ntype.hole:
            raise TypeError
        elif self.ntype.prim or self.ntype.var or self.ntype.app:
            assert self.phantom_ntype is None
            self.phantom_ntype = self.ntype
            self.ntype = NType.HOLE
        else:
            raise TypeError
    def unhide(self,recursive=False):
        if self.ntype.hole:
            assert self.phantom_ntype is not None
            assert not self.phantom_ntype.hole
            self.ntype = self.phantom_ntype
            self.phantom_ntype = None

        if recursive: # must come after setting our own ntype otherwise we dont have any children if we're a hole
            for c in self.children():
                c.unhide(recursive=True)

    def pause_cache(self, children=True):
        raise NotImplementedError
        """
        pauses the cache for us and our children (unless children=False)
            - note that this makes the whole system ignore the cache and itll stay
              in whatever state you left it (it will NOT clear the cache)
        """
        self.cache_paused = True
        self.cache = Cache(None,None)
        if children:
            for c in self.children():
                c.pause_cache(children=True)



        
class NType(enum.Enum):
    # dont change the order or it changes loaded programs too
    ABS = 0
    APP = 1
    VAR = 2
    PRIM = 3
    HOLE = 4
    OUTPUT = 5
    @staticmethod
    def from_program(p):
        if p.isAbstraction:
            return NType.ABS
        elif p.isApplication:
            return NType.APP
        elif p.isIndex:
            return NType.VAR
        elif p.isPrimitive:
            return NType.PRIM
        elif p.isHole:
            return NType.HOLE
        raise TypeError
    @property
    def abs(self):
        return self == NType.ABS
    @property
    def app(self):
        return self == NType.APP
    @property
    def var(self):
        return self == NType.VAR
    @property
    def prim(self):
        return self == NType.PRIM
    @property
    def hole(self):
        return self == NType.HOLE
    @property
    def output(self):
        return self == NType.OUTPUT
