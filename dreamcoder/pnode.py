
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

def cached_propagate(propagate):
    @functools.wraps(propagate)
    def _cached_propagate(self,towards,concrete_only=False):
        if self.no_cache:
            return propagate(self, towards, concrete_only=concrete_only)

        if self.cache.towards is towards:
            return self.cache.exwise

        exwise = propagate(self, towards, concrete_only=concrete_only)
        self.cache = Cache(towards,exwise)
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

Ctx = namedtuple('Ctx','tp exwise')
Cache = namedtuple('Cache','towards exwise')

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
    
    def __init__(self, ntype, parent, ctx:list):
        super().__init__()

        if isinstance(parent,PTask):
            self.task = parent
            self.parent = self # root node. We dont use None as the parent bc we want towards=None to be reserved for our Cache emptiness
        else:
            self.task = parent.task
            self.parent = parent

        self.ntype = ntype
        self.ctx = ctx

        self.cache = Cache(towards=None, exwise=None) # can't use `None` as the cleared cache since that's a legit direciton for the output node
        
    def expand_to(self, prim, cache='kill'):
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

        """
        self = self.unwrap_abstractions()
        assert self.ntype.hole
        assert self.tp == prim.tp
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
                self.fn = self.build_hole(prim.tp.returns())
                self.xs = [self.build_hole(arg_tp) for arg_tp in prim.tp.functionArguments()]
        else:
            raise TypeError
        
    def build_hole(self, tp):
        """
        Make a new hole with `self` as parent (and `ctx` calculated from `self`)
        This also handles expanding into Abstractions if tp.isArrow()
        """
        if not tp.isArrow():
            return PNode(NType.HOLE, parent=self, ctx=self.ctx)
        # Introduce a lambda

        which_toplevel_arg = len(self.ctx) # ie if we're just entering the first level of lambda this will be 0

        # now if this arg is one of our inputs we grab that input, otherwise we just return None
        # (so for non toplevel lambdas this is always None)
        exwise = self.task.inputs[which_toplevel_arg] if len(self.task.inputs) > which_toplevel_arg else None

        arg_tp = self.tp.arguments[0] # the input arg to this arrow
        res_tp = self.tp.arguments[1] # the return arg (which may be an arrow)

        # push it onto the ctx stack
        abs = PNode(NType.ABS, parent=self, ctx=(Ctx(arg_tp,exwise),*self.ctx))
        # now make an inner hole (which may itself be an abstraction ofc)
        inner_hole = abs.build_hole(res_tp)
        abs.body = inner_hole

        return abs # and return our abstraction

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
        root = PNode(NType.OUTPUT, parent=ptask, ctx=())
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

        assert p.infer() == self.tp

        if p.isPrimitive or p.isIndex:
            # p is a Primitive or Index
            self.expand_to(p)
        elif p.isHole:
            pass # we already are a hole!
        elif p.isAbstraction:
            assert False # can't happen bc of the `while p.isAbstraction` unwrapping above
        elif p.isApplication:
            # application. We expand each hole we create (the f hole and the xs holes)
            self.expand_to(p) # expand, creating holes
            f, xs = p.applicationParse()
            self.fn.expand_from_dreamcoder(f)
            for x,x_ in zip(self.xs,xs):
                x.expand_from_dreamcoder(x_)
        else:
            raise TypeError
    
    def __repr__(self):
        if self.ntype.abs:
            return f'(lambda {self.body})'
        elif self.ntype.app:
            args = ' '.join(repr(arg) for arg in self.args)
            return f'({self.fn} {args})'
        elif self.ntype.prim:
            return f'{self.name}'
        elif self.ntype.var:
            return f'${self.i}'
        elif self.ntype.HOLE:
            return f'<HOLE>'
        else:
            raise TypeError
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
                if self.body.in_HOF_lambda and not self.has_holes and (sing.cfg.model.pnode.allow_concrete_eval or concrete_only):
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
    def get_hole(self, ordering, tiebreaker='random'):
        """
        returns a single hole or None if there are no holes in the subtree
        """
        if self.ntype.output:
            return self.tree.get_hole(ordering, tiebreaker=tiebreaker)
        elif self.ntype.abs:
            return self.body.get_hole(ordering, tiebreaker=tiebreaker)
        elif self.ntype.var or self.ntype.prim:
            return None
        elif self.ntype.hole:
            return self
        elif self.ntype.app:
            holes = [self.fn.get_hole(ordering, tiebreaker=tiebreaker)] + [x.get_hole(ordering,tiebreaker=tiebreaker) for x in self.xs]
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
        returns a list of any nodes below this one in the tree
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
    def get_prim(self):
        self = self.unwrap_abstractions()
        if self.ntype.output or self.ntype.hole:
            raise TypeError
        elif self.ntype.app:
            return self.fn.get_prim()
        if self.ntype.prim:
            return self.prim
        elif self.ntype.var:
            return Index(self.i)
        elif self.ntype.abs:
            assert False, "not possible bc we unwrapped abstractions"
        else:
            raise TypeError
    def unwrap_abstractions(self, count=False):
        """
        traverse down .body attributes until you get to the first non-abstraction PNode
        """
        assert self.ntype.abs
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
    def clear_cache(self, children=True):
        raise NotImplementedError
        """
        clears the cache of us and our children (unless children=False)
        """
        self.cache = Cache(None,None)
        if children:
            for c in self.children():
                c.clear_cache(children=True)
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




class PTrace:
    def __init__(self, root, phead, ordering, tiebreaking='random') -> None:
        assert root.ntype.output
        self.root = root
        self.phead = phead
        self.ordering = ordering
        self.tiebreaking = tiebreaking
        self.prev_hole = None

        self.into_phantom_tree(root) # leave the toplevel ntype.output node.


    def into_phantom_tree(self,node):
        """
        Turn whole tree into Holes but without destroying the old data
        But dont do this for abstractions + output nodes bc it's impossible to "guess" these
        and theyre autofilled during search.
        """
        assert not node.ntype.hole
        if not (node.ntype.abs or node.ntype.output):
            node._old_ntype = node.ntype
            node.ntype = NType.HOLE
        for c in node.children():
            self.into_phantom_tree(c)

    def process_holes(self, process_hole):
        processed_holes = []
        masks = []
        targets = []
        strings = []
        while True:
            hole = self.root.get_hole(self.ordering,self.tiebreaking)
            if hole is None:
                break
            masks.append(self.phead.build_mask(hole))
            targets.append(self.phead.build_target(hole))
            strings.append(str(self.pnode))
            processed_holes.append(process_hole(hole))
        return processed_holes, masks, targets, strings

    
    def iter_inplace(self):
        """
        be very careful, this generator mutates the original pnode so dont ask for the next element
        until youve already processed the previous one!!!!!!

        We actually dont even return the pnode, just so you cant collect those values into a list
        """
        assert False, "deprecated for process_holes"
        if self.prev_hole is not None:
            # teacher-force the hole into the proper node
            self.prev_hole.ntype = self.prev_hole._old_ntype
            del self.prev_hole._old_ntype

        hole = self.pnode.get_hole(self.ordering,self.tiebreaking)
        if hole is None:
            return # stopiteration

        self.masks.append(self.phead.build_mask(hole))
        self.targets.append(self.phead.build_target(hole))
        self.strings.append(str(self.pnode))

        yield hole





        
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


# ABS = NType.ABS
# APP = NType.APP
# VAR = NType.VAR
# HOLE = NType.HOLE
# OUTPUT = NType.OUTPUT


    # def follow_zipper(self, zipper):
    #     next,*rest = zipper
    #     if next == 'body':
    #         pass
    #     pass
    # def get_zippers(self):
    #     zippers = []
    #     def _get_zippers(node):
    #         nonlocal zippers
    #         if node.ntype.hole:

    #     pass
    # def parent_first(self,fn,acc):
    #     """
    #     simple helper that just calls fn() on every PNode in the tree.
    #     Doesn't return anything. If you want to return something have
    #     your fn maintain some state (eg a class with __call__ or 
    #     just a `nonlocal` variable closured in)
    #     """
    #     acc = fn(self,acc)
    #     if self.ntype.output:
    #         return self.tree.parent_first(fn,acc)
    #     if self.ntype.abs:
    #         self.body.recurse(fn)
    #     elif self.ntype.app:
    #         self.fn.recurse(fn)
    #         for x in self.xs:
    #             x.recurse(fn)
    #     elif self.ntype.var or self.ntype.hole or self.ntype.prim:
    #         pass
    #     else:
    #         raise TypeError





    # def has_unset_index(self):
    #     found = False
    #     def finder(node):
    #         nonlocal found
    #         if node.ntype.var and node.ctx[node.i].is_unset:
    #             found = True
    #     self.traverse(finder)
    #     return found
