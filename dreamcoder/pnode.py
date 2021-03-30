
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

#Ctx = namedtuple('Ctx','tp exwise')
#Cache = namedtuple('Cache','towards exwise string')
class FoundSolution(Exception):
    def __init__(self,p):
        self.p = p


from collections import defaultdict

def return_none(): return None

class PNodeCache:
    def __init__(self):
        self.clear()
    def clone(self):
        new = PNodeCache()
        new.__dict__.update(self.__dict__)
        # shallow duplicate the dicts
        new.labelled_arg_ews = new.labelled_arg_ews.copy()
        new.res_inverse_app = new.res_inverse_app.copy()
        return new
    def clear(self):
        # ctx: used for beval, inverse abs, inverse app
        self.ctx = None # actually used by process_hole as well!
        
        # used by beval
        self.beval_cache_str = None

        # used by inverse abs, inverse app
        self.output_ew = None

        # used by inverse app
        self.fn_ew = None # EW
        self.labelled_arg_ews = {} # zipper0 -> [EW]

        # cached results
        self.res_beval = None # EW
        self.res_inverse_abs = None # ctx to use for recursive call
        self.res_inverse_app = defaultdict(return_none) # zipper0 -> new output EW to use for recursive call
    
    def try_ctx_change(self,ctx):
        if ctx is self.ctx:
            return
        self.ctx = ctx
        self.res_beval = None
        self.res_inverse_abs = None
        # dont need to wipe the inverse app bc it doesnt use ctx except via beval of args/fn which will invalidate cache anyways
    
    def try_output_ew_change(self,output_ew):
        if output_ew is self.output_ew:
            return
        self.output_ew = output_ew
        self.res_inverse_abs = None
        self.res_inverse_app = defaultdict(return_none)

    def beval(self, pnode, no_cache, ctx):
        if sing.cfg.debug.no_cache:
            return no_cache()
        def check_in_cache():
            if self.res_beval is None:
                return False
            if ctx is not self.ctx:
                return False
            if pnode.beval_cache_str() != self.beval_cache_str:
                return False
            return True
        
        in_cache = check_in_cache()
        if in_cache:
            sing.stats.cache_beval.hit()
            if sing.cfg.debug.validate_cache:
                validate_ew_equal(self.res_beval, no_cache()), "Invalid cache"
            return self.res_beval
        sing.stats.cache_beval.miss()
        self.try_ctx_change(ctx)
        self.beval_cache_str = pnode.beval_cache_str()
        self.res_beval = no_cache()
        return self.res_beval

    def inverse_abs(self, pnode, no_cache, ctx, output_ew):
        if sing.cfg.debug.no_cache:
            return no_cache()
        assert pnode.ntype.abs
        def check_in_cache():
            if self.res_inverse_abs is None:
                return False
            if ctx is not self.ctx:
                return False
            if output_ew is not self.output_ew:
                return False
            return True
        
        in_cache = check_in_cache()
        if in_cache:
            sing.stats.cache_inverse_abs.hit()

            if sing.cfg.debug.validate_cache:
                new_ctx = no_cache()
                assert len(self.res_inverse_abs) == len(new_ctx)
                for ew1,ew2 in zip(new_ctx, self.res_inverse_abs):
                    validate_ew_equal(ew1,ew2), "Invalid cache"
            return self.res_inverse_abs

        sing.stats.cache_inverse_abs.miss()
        self.try_ctx_change(ctx)
        self.try_output_ew_change(output_ew)
        self.res_inverse_abs = no_cache()
        return self.res_inverse_abs

    def inverse_app(self, pnode, no_cache, output_ew, fn_ew, labelled_arg_ews, zipper0):
        if sing.cfg.debug.no_cache:
            return no_cache()
        assert pnode.ntype.app
        assert isinstance(zipper0,int)
        def check_in_cache():
            if self.res_inverse_app[zipper0] is None:
                return False
            # dont need to check ctx for app actually :)
            if output_ew is not self.output_ew:
                return False
            if fn_ew is not self.fn_ew:
                return False
            assert len(labelled_arg_ews) == len(self.labelled_arg_ews[zipper0]), "num args is changing???"
            for (i1,ew1),(i2,ew2) in zip(labelled_arg_ews,self.labelled_arg_ews[zipper0]):
                assert i1 == i2, "somethings strange"
                if ew1 is not ew2:
                    return False
            return True
        
        in_cache = check_in_cache()
        if in_cache:
            sing.stats.cache_inverse_app.hit()
            if sing.cfg.debug.validate_cache:
                validate_ew_equal(self.res_inverse_app[zipper0], no_cache()), "Invalid cache"
            return self.res_inverse_app[zipper0]
        sing.stats.cache_inverse_app.miss()

        if self.fn_ew is not fn_ew:
            self.fn_ew = fn_ew
            # change affects all inverses so we gotta wipe them all
            self.res_inverse_app = defaultdict(return_none)
        self.try_output_ew_change(output_ew)
        self.labelled_arg_ews[zipper0] = labelled_arg_ews
        self.res_inverse_app[zipper0] = no_cache()
        return self.res_inverse_app[zipper0]
        

        
def validate_ew_equal(cached_ew, new_ew):
    """
    for cache validation. Note that this will actually recurse
    since when we manually call propagate here then that inner
    call will call self.propagate (which is _cached_propagate) internally so
    it'll end up verifying the entire cache since validate_cache will be true
    for all these cases.
    """
    assert cached_ew.placeholder == new_ew.placeholder
    if cached_ew.placeholder:
        return # nothing to check for placeholders

    assert (cached_ew.concrete is None) == (new_ew.concrete is None)
    assert (cached_ew.closure is None) == (new_ew.closure is None)
    
    if cached_ew.concrete:
        assert cached_ew.concrete == new_ew.concrete
    
    if cached_ew.closure:
        pass # not sure what checks to do here
    
    if (cached_ew.abstract is not None
        or new_ew.abstract is not None
        or random.random() < sing.cfg.debug.p_check_abstract):
        # this check is pretty thorough but also will massively slow everything down hence the probability thing
        assert torch.allclose(cached_ew.get_abstract(), new_ew.get_abstract())



class Cached:
    def __init__(self):
        self.val = self # sentinel
    def get(self, func):
        if sing.cfg.debug.validate_cache and self.val is not self:
            new_val = func()
            if torch.is_tensor(self.val) and torch.is_tensor(new_val):
                assert torch.allclose(self.val,new_val)
            else:
                assert self.val == new_val
        if self.val is self: # empty cache
            self.val = func()
        return self.val
    def clear(self):
        self.val = self


class Context:
    vals = ()
    def __init__(self):
        raise NotImplementedError
    def __add__(self,other):
        assert isinstance(other,self.__class__)
        return self.__class__(self.vals + other.vals)
    def __iter__(self):
        return iter(self.vals)
    def __len__(self):
        return len(self.vals)
    def __getitem__(self,indexer):
        return self.vals[indexer]
    def __repr__(self):
        inner = ', '.join(compressed_str(repr(ew)) for ew in self.vals)
        return f'{self.__class__.__name__}(len={len(self)}: {inner})'

class EWContext(Context):
    is_ew = True
    def __init__(self, ews=()) -> None:
        """
        Takes an EW or tuple thereof and builds an immutable context
          (which behaves roughly like a tuple)
        
        Note [0] is the top of the ctx stack and thus the $0 argument
        Note `ctx1+ctx2` would make [0] come from ctx1
        Pushing onto the stack looks like: `ctx = EWContext(exwise) + ctx`
        """
        assert isinstance(ews,(tuple,list))
        assert all(isinstance(ew,Examplewise) for ew in ews)
        self.vals = tuple(ews)
        self.no_free = all(not ew.placeholder for ew in self.vals)
        self.cached_encode = Cached()
    def encode(self):
        """
        used in places like beval() for Hole as well as by the policy
        """
        return self.cached_encode.get(lambda:sing.model.abstraction_fn.encode_ctx(self))
    @staticmethod
    def get_placeholders(argc):
        return EWContext([Examplewise(placeholder=True) for _ in range(argc)])
    def split(self):
        """
        Turn an exwise context into a list of single contexts
        """
        assert self.no_free, "singlescontexts are fully concrete and never have free vars"
        list_of_ews = [ew.split() for ew in self.vals] # Examplewise.split()
        list_of_ctxs = list(zip(*list_of_ews)) # [len_ctx,num_exs] -> [num_exs,len_ctx]
        return [SinglesContext(ctx) for ctx in list_of_ctxs]


class SinglesContext(Context):
    is_ew = False
    def __init__(self, vals=()) -> None:
        """
        Single Context (as opposed to examplewise)
        """
        self.vals = tuple(vals)


class UncurriedFn:
    """
    Takes a normal function callable like fn(a)(b) and makes it callable
    like fn(a,b). I think you can still call it in the
    curried way but whatever it returns wont be an UncurriedFn for the record.

    equality compares by name btw
    """
    def __init__(self,fn,name=None):
        self.fn = fn
        assert callable(fn)
        self.name = name if name is not None else getattr(fn,'__name__',repr(fn))
    def __repr__(self):
        return self.name
    def __call__(self,*args):
        return uncurry(self.fn,args)
    def __eq__(self,other):
        return isinstance(other,UncurriedFn) and self.name == other.name and self.fn == other.fn





class Closure:
    def __init__(self, abs, enclosed_ctx):
        """
        Returned by beval() for ABS 

        Note enclosed_ctx can either be an Exwise list or just singles!
        """
        assert abs.ntype.abs
        self.is_ew == enclosed_ctx.is_ew, f"classes dont match, only one is examplewise: {self.__class__} {enclosed_ctx.__class__}"

        self.abs = abs # an ABS
        self.enclosed_ctx = enclosed_ctx
        self.argc = self.abs.argc

    def __repr__(self):
        return f'{self.__class__.__name__}({self.abs} with {self.enclosed_ctx})'

class SinglesClosure(Closure):
    is_ew = False
    def __call__(self, *args):
        """
        blended-exec the closure body with the context of `args + enclosed_ctx`.
        If more args are provided than the body takes, we'll (reasonably) assume that you want
        currying and will execute the body with fewer args then feed remaining args
        into whatever the body returns (hopefully another closure)
        Same as EWClosure.__call except uses beval_single_concrete and SinglesContext
        """
        assert len(args) >= self.argc
        consumed_args = args[:self.argc] # consume as many args as we can
        remaining_args = args[self.argc:]
        ctx = SinglesContext(consumed_args) + self.enclosed_ctx
        res = self.abs.body.beval_single_concrete(ctx)
        if len(remaining_args) > 0:
            assert callable(res)
            res = res(remaining_args) # recursion of its a Closure, and works fine if its a prim fn as well
        return res


class EWClosure(Closure):
    is_ew = True
    def __call__(self, *args):
        """
        blended-exec the closure body with the context of `args + enclosed_ctx`.
        If more args are provided than the body takes, we'll (reasonably) assume that you want
        currying and will execute the body with fewer args then feed remaining args
        into whatever the body returns (hopefully another closure)
        Same as SinglesClosure.__call except uses beval and EWContext

        """
        assert len(args) >= self.argc
        consumed_args = args[:self.argc] # consume as many args as we can
        remaining_args = args[self.argc:]
        ctx = EWContext(consumed_args) + self.enclosed_ctx
        res = self.abs.body.beval(ctx)
        if len(remaining_args) > 0:
            assert callable(res)
            res = res(remaining_args) # recursion of its a Closure, and works fine if its a prim fn as well
        return res
    def split(self):
        """
        Turn an exwise closure into a list of single closures by turning the
        exwise based enclosed ctx into num_exs non-exwise contexts
        """
        return [SinglesClosure(self.abs,ctx) for ctx in self.enclosed_ctx.split()]

def cached_propagate(propagate):
    @functools.wraps(propagate)
    def _cached_propagate(self,towards,concrete_only=False):
        assert False
        if concrete_only: # its fast enough to just do it directly and not have to deal with cache craziness
            return propagate(self, towards, concrete_only=concrete_only)

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
        self.inputs = [Examplewise(concrete=arg) for arg in _argwise]
        # self.cached_input_features = Cached()

        ctx = EWContext()
        for exwise in self.inputs:
            """
            inputs[0] is the arg to the outermost lambda therefore
            It should be pushed onto the ctx stack first
            """
            ctx = EWContext((exwise,)) + ctx
        self.ctx = ctx




    def output_features(self):
        return self.outputs.get_abstract()
    def input_features(self):
        # honestly i like this setup more than a decorator bc it means i get
        # my Cached object which makes garbage collection less opaque
        return self.ctx.encode()
        #return self.cached_input_features.get(self._input_features)
    # def _input_features(self):
    #     return self.ctx.encode()
    #     #return sing.model.abstraction_fn.encode_known_ctx([Ctx(None,i) for i in self.inputs])


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
    def __init__(self,/,concrete=None,abstract=None,closure=None,placeholder=False) -> None:
        super().__init__()
        assert (concrete,abstract,closure).count(None) + (placeholder is False) == 3
        self.concrete = concrete
        self.abstract = abstract
        self.closure = closure
        self.placeholder = bool(placeholder)

        if abstract is not None:
            assert torch.is_tensor(abstract)
            assert abstract.dim() == 2
            assert abstract.shape[0] == sing.num_exs
        
        if concrete is not None:
            assert isinstance(concrete,(list,tuple))
            assert len(concrete) == sing.num_exs
            self._check_in_range()
        
        if closure is not None:
            assert isinstance(closure,Closure)
    
    def get_abstract(self):
        if self.abstract is None:
            if self.concrete is not None:
                self.abstract = sing.model.abstraction_fn.encode_concrete_exwise(self)
            elif self.closure is not None:
                args = EWContext.get_placeholders(self.closure.argc)
                self.abstract = self.closure(*args).get_abstract()
            else:
                assert False
        return self.abstract
    
    def encode_placeholder(self,i):
        """
        We don't cache this since it can actually change.
        And it's not part of get_abstract() bc it's dependent on the index
        """
        assert self.placeholder
        return sing.model.abstract_transformers.lambda_index_nms[i]().expand(sing.num_exs,-1)
    
    def split(self):
        """
        Goes from an EW to a list of num_exs "singles" (python values or )
        """
        if self.concrete:
            return self.concrete
        if self.closure:
            return self.closure.split() # EWClosure.split()
        if self.abstract:
            # not sure youll ever want to split an abstract
            # guy but this is how you'd do it.
            return [inner_tensor for inner_tensor in self.abstract]
        if self.placeholder:
            assert False, "singles are fully concrete"
        assert False


    @property
    def can_be_concrete(self):
        """
        True if this can be used in fully concrete computation
        """
        if self.concrete is not None:
            return True
        elif self.closure is not None and not self.closure.abs.has_holes:
            return True
        return False


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
            return f'EW(c={compressed_str(repr(self.concrete[0]))},...)'
        if self.closure is not None:
            return f'EW({self.closure})'
        if self.abstract is not None:
            return 'EW(abstract)'
        if self.placeholder:
            return 'EW(placeholder)'


class PNode:
    """

    PNodes basically look like the Tree structure you'd expect instead
    of the wildness of dreamcoder.program.Program where you have to do
    applicationParse() etc.

    lambda abstractions are still used bc many things operating over
    PNodes will want to be aware that the EWContext is changing

    create a new one of these from a program like so:
        node = PNode(p=some_program, parent=PNode(PTask(task)))
    then run it with
        node.evaluate(output_node=None)

    """
    
    def __init__(self, ntype, tp, parent, ctx_tps):
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
        self.ctx_tps = ctx_tps
        #self.ctx = ctx

        self.pnode_cache = PNodeCache()

        #self.cache = Cache(None, None, None) # can't use `None` as the cleared cache since that's a legit direciton for the output node
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
        # if cache_mode is not None:
        #     self.clear_cache(cache_mode) # ok to do this here bc we wont be editing cache at all during expand_to
        self = self.unwrap_abstractions()
        assert self.ntype.hole
        assert not self.tp.isArrow(), "We should never have an arrow for a hole bc it'll instantly get replaced by abstractions with an inner hole"
        
        if prim.isIndex:
            assert self.ctx_tps[prim.i] == self.tp # the ctx tp is the same as our tp
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
                self.fn = PNode(NType.PRIM, tp=prim.tp, parent=self, ctx_tps=self.ctx_tps)
                self.fn.prim = prim
                self.fn.name = prim.name
                self.fn.value = UncurriedFn(prim.value,name=prim.name)
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
            return PNode(NType.HOLE, tp, parent=self, ctx_tps=self.ctx_tps)
        
        # Introduce a lambda
        
        #which_toplevel_arg = len(self.ctx) # ie if we're just entering the first level of lambda this will be 0

        # now if this arg is one of our inputs we grab that input, otherwise we just return None
        # (so for non toplevel lambdas this is always None)
        #exwise = self.task.inputs[which_toplevel_arg] if len(self.task.inputs) > which_toplevel_arg else None

        arg_tp = tp.arguments[0] # the input arg to this arrow
        res_tp = tp.arguments[1] # the return arg (which may be an arrow)

        # push it onto the ctx stack
        # abs = PNode(NType.ABS, tp, parent=self, ctx=(Ctx(arg_tp,exwise),*self.ctx))
        # now make an inner hole (which may itself be an abstraction ofc)
        abs = PNode(NType.ABS, tp, parent=self, ctx_tps=(arg_tp,*self.ctx_tps))
        inner_hole = abs.build_hole(res_tp)
        abs.body = inner_hole
        abs.argc = 1  # TODO can change

        return abs # and return our abstraction

    def into_hole(self, cache_mode='single'):
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
        # self.clear_cache(cache_mode)
        self.ntype = NType.HOLE

    def expand_from(self, other, recursive=True):
        """
        like expand_to but taking another PNode instead of a Prod.
        recursive=True means itll recurse so you can fill in whole subtrees.
        cache wont be copied, you should call self.copy_cache_from(other, recursive=True) at the
            end if you want that
        Abstractions and Outputs get unwrapped
        """
        if self.ntype.output:
            self = self.tree
        if other.ntype.output:
            other = other.tree
        self = self.unwrap_abstractions()
        other = other.unwrap_abstractions()
        assert self.ntype.hole
        assert self.tp == other.tp
        assert not other.ntype.output, "holes never expand into outputs"
        assert not other.ntype.abs, "this should have been unwrapped"

        if other.ntype.hole:
            return # we're already a hole! Nothing to change
        
        prod = other.get_prod()
        self.expand_to(prod)
        assert self.ntype == other.ntype, "expansion didnt yield the expected ntype"
        
        if recursive and self.ntype.app:
            # APP is the only case w children but we dont do the `fn` since its just a prefilled prim
            for c,o in zip(self.xs,other.xs):
                c.expand_from(o,recursive=True)


        
    def check_solve(self):
        """
        check if we're a solution to the task
        """
        if self.root().has_holes:
            return False
        try:
            res = self.root().beval(None).concrete
            assert res is not None
            return res == self.task.outputs.concrete
        except InvalidSketchError:
            return False


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
        root = PNode(NType.OUTPUT, tp=None, parent=ptask, ctx_tps=())
        root.tree = root.build_hole(ptask.request)

        # do some sanity checks
        hole, num_abs = root.tree.unwrap_abstractions(count=True)
        assert ptask.argc == num_abs, "task argc doesnt match number of toplevel abstraction"
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
        assert str(root.tree) == str(p), "these must be the same for the sake of the RNN which does str() of the pnode"
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
    
    def __str__(self):
        if self.ntype.abs:
            return f'(lambda {self.body})'
        elif self.ntype.app:
            args = ' '.join(str(arg) for arg in self.xs)
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
    def beval_cache_str(self):
        return str(self) # ideally this should change iff beval() will cahnge
    def __repr__(self):
        return f'{self.ntype.name}({self.tp}): {self.marked_str()}'
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
            res = str(self)
        elif self.ntype.output:
            res = self.tree.marked_repr(marked_node)
        else:
            raise TypeError
        if self is marked_node:
            return f'[[{res}]]'
        return res
    @property
    def in_HOF_lambda(self): # TODO worth changing, p specific to this dsl
        return self.get_zipper().count('body') > len(self.task.inputs)

    def propagate_upward(self, concrete_only=False):
        assert False
        """
        BAS-style upward-only propagation
        """
        return self.propagate(self.parent, concrete_only=concrete_only)
    def propagate_to_hole(self):
        """
        MBAS-style propagation to hole
        """
        assert False
        assert self.ntype.hole
        return self.propagate(self)




    def beval_single_concrete(self, ctx):
        """
        non-exwise beval. If you call it on an output node or an abs node youll
        get a python lambda you can feed args into.
        """

        sing.scratch.beval_print(f'beval_single {self} with ctx={ctx}', indent=True)

        def printed(res):
            assert res is not None
            sing.scratch.beval_print(f'{mlb.mk_green("->")} {res}', dedent=True)
            return res


        assert not self.has_holes


        if self.ntype.output:
            assert ctx is None
            return printed(self.tree.beval_single_concrete(SinglesContext())) # this will just convert the toplevel ABS into a closure
        
        assert ctx is not None

        if self.ntype.prim:
            """
            for both fn prims and other prims
            """
            return printed(self.value)

        elif self.ntype.var:
            return printed(ctx[self.i])

        elif self.ntype.hole:
            assert False

        elif self.ntype.abs:
            return printed(SinglesClosure(abs=self,enclosed_ctx=ctx))

        elif self.ntype.app:
            fn = self.fn.beval_single_concrete(ctx)
            args = [arg.beval_single_concrete(ctx) for arg in self.xs]
            return printed(fn(*args))
    
    def beval(self, ctx):
        """
        call like root.beval(ctx=None) and the output node will fill the right ctx for you.
        """
        sing.scratch.beval_print(f'{mlb.mk_blue("beval")}({self.ntype}) {self} with ctx={ctx}', indent=True)

        if hasattr(self,'needs_ctx'):
            self.needs_ctx = ctx
            sing.scratch.beval_print('[hit beval needs ctx]')

        def printed(res):
            assert res is not None
            sing.scratch.beval_print(f'{mlb.mk_green("->")} {short_repr(res)}', dedent=True)
            return res
                
        def no_cache():
            if self.ntype.output:
                """
                beval on the output sets up the ctx properly and skips over the abstractions into their bodies and
                executes the bodies in the proper context.
                """
                assert ctx is None
                body,i = self.tree.unwrap_abstractions(count=True)
                assert len(self.task.ctx) == i
                return body.beval(self.task.ctx)
            
            assert ctx is not None

            if self.ntype.prim:
                """
                even for fn primitives, just always return an EW(concrete=...)
                """
                return Examplewise(concrete=[self.value for _ in range(sing.num_exs)])

            elif self.ntype.var:
                ew = ctx[self.i]
                if not ew.placeholder:
                    return ew # normal case
                else:
                    # encode a free var
                    return Examplewise(abstract=ew.encode_placeholder(self.i))


            elif self.ntype.hole:
                return Examplewise(abstract=sing.model.abstract_transformers.hole_nms[self.tp.show(True)](ctx.encode()))

            elif self.ntype.abs:
                return Examplewise(closure=EWClosure(abs=self,enclosed_ctx=ctx))

            elif self.ntype.app:
                fn = self.fn.beval(ctx)
                args = [arg.beval(ctx) for arg in self.xs]
                

                if (ctx.no_free and # no free vars
                    fn.can_be_concrete and # fn can be concrete
                    all(x.can_be_concrete for x in args) # args can be concrete
                    ):
                    """
                    Concrete application!
                    This uses beval_single_concrete implicitly since it
                        does ew.split() on all args as well as the fn which
                        means even if the fn was a closure it became a list of
                        SinglesClosures with SinglesContexts (so even our current
                        context was converted even tho we dont explicitly convert
                        it here).
                    """
                    sing.scratch.beval_print(f'[concrete apply]')

                    singles_fns = fn.split()
                    singles_args = list(zip(*[ew.split() for ew in args])) # [argc,num_exs] -> [num_exs,argc]
                    assert len(singles_fns) == len(singles_args) == sing.num_exs
                    res = [fn(*args) for fn,args in zip(singles_fns,singles_args)]
                    return Examplewise(concrete=res)



                """
                Abstract call!
                Abstract the fn and args (they were already beval()'d) then label them and pass to apply_nn
                """
                ### * V1 * ###
                sing.scratch.beval_print(f'[abstract apply]')
                sing.stats.fn_called_abstractly += 1
                assert self.fn.ntype.prim, "would work even if this wasnt a prim, but for v1 im leaving this here as a warning"
                
                fn_embed = fn.get_abstract() # gets the Parameter vec for that primitive fn
                args_embed = [arg.get_abstract() for arg in args]
                labelled_args = list(enumerate(args_embed)) # TODO important to change this line once you switch to multidir bc this line to labels the args in order

                return Examplewise(abstract=sing.model.apply_nn(fn_embed, labelled_args, parent_vec=None))

            else:
                raise TypeError
        
        res = self.pnode_cache.beval(self,no_cache,ctx)
        return printed(res)


    def embed_from_above(self):
        """
        Imagine youre a hole. What would multidirectional propagation say the representation
        of you is?
        Note that this does NOT include your type or your context (beyond the fact that
        context is used other places)
        """
        sing.scratch.beval_print(f'embed_from_above {self.marked_str()} ')

        root = self.root()
        zipper = self.get_zipper()
        res = root.inverse_beval(ctx=None, output_ew=None, zipper=zipper)
        return res
    def __eq__(self,other):
        return self is other
    def inverse_beval(self, ctx, output_ew, zipper):
            """
            follow `zipper` downward starting at `self` 
            """
            sing.scratch.beval_print(f'inverse_beval {self} with zipper={zipper} and ctx={ctx}', indent=True)
        
            if hasattr(self,'needs_ctx'):
                assert self.ntype.hole, "temp"
                sing.scratch.beval_print('[hit needs_ctx]')
                assert self.needs_ctx is None, "someone forgot to garbage collect"
                self.needs_ctx = ctx

            # if (res:=self.pnode_cache.try_beval_cache(self,ctx)) is not None:
            #     sing.scratch.beval_print('[cache hit]')
            #     return printed(res) # doesnt hurt to call update_beval_cache() really anyways

            def printed(res):
                assert res is not None
                sing.scratch.beval_print(f'{mlb.mk_green("->")} {short_repr(res)}', dedent=True)
                return res

            if len(zipper) == 0:
                """
                reached end of zipper so we can just return the embedding
                without even looking at what node we're pointing to
                """
                sing.scratch.beval_print(f'[end of zipper]')
                return printed(output_ew)
            
            if self.ntype.output:
                """
                set up the ctx right, set output_ew to task.outputs, and strip off the abstractions
                
                caching: no need. No new Context or EW objects get created here
                """
                assert zipper[0] == 'tree'
                zipper = zipper[1:]
                assert ctx is None
                assert output_ew is None

                body,i = self.tree.unwrap_abstractions(count=True)
                assert len(self.task.ctx) == i
                assert len(zipper) >= i, "zippers must pass all the way thru the toplevel abstraction"
                assert all(x == 'body' for x in zipper[:i])
                zipper = zipper[i:]

                return printed(body.inverse_beval(
                    ctx=self.task.ctx,
                    output_ew=self.task.outputs,
                    zipper=zipper
                    ))

            assert ctx is not None
            assert output_ew is not None

            """
            no children to invert into for these types
            """
            if self.ntype.prim:
                assert False
            elif self.ntype.var:
                assert False
            elif self.ntype.hole:
                assert False

            elif self.ntype.abs:
                """
                We will pass our own ctx into the body since this is the enclosed_ctx anyways since
                    rn we're at definition time not application time.
                We will add placeholders for the lambda args to our context so they can be encoded
                    and referenced by anyone.
                Then we simply do the inverse of the body

                caching: yes, since a new Context is created this will invalidate everyones cache if we dont reuse our existing Context instead
                """
                assert zipper[0] == 'body'

                    # sing.scratch.beval_print(f'[trivial abs cache hit]')

                def no_cache():
                    # no cache hit
                    return EWContext.get_placeholders(self.argc) + ctx
                
                new_ctx = self.pnode_cache.inverse_abs(self, no_cache, ctx, output_ew)
                
                res = self.body.inverse_beval(new_ctx, output_ew, zipper[1:])
                
                return printed(res)

            elif self.ntype.app:
                """
                Applications can be inverted towards and arg (zipper[0] is an int)
                or towards the function (zipper[0] == 'fn')
                """
                if zipper[0] == 'fn':
                    assert False, "not a v1 thing and cant ever show up in v1 anyways"
                
                assert isinstance(zipper[0],int)
                assert 0 <= zipper[0] < len(self.xs)

                """
                Abstract inversion! Bc theres no concrete inversion.
                Beval and abstract the fn and args except the one we're inverting into.
                label the args and run apply_nn with output_ew as the parent vector
                """

                assert self.fn.ntype.prim, "feel free to remove post V1"

                sing.scratch.beval_print(f'[inverting application]')

                sing.scratch.beval_print(f'[beval fn]')
                fn_embed = self.fn.beval(ctx)
                
                sing.scratch.beval_print(f'[beval {len(self.xs)-1} args]')
                labelled_args = [(i,arg.beval(ctx)) for i,arg in enumerate(self.xs) if i!=zipper[0]]

                
                def no_cache():
                    sing.scratch.beval_print(f'[get_abstract calls]') # these dont come until here bc of caching
                    fn_embed_vec = fn_embed.get_abstract()
                    labelled_args_vecs = [(i,arg.get_abstract()) for i,arg in labelled_args]
                    output_ew_vec = output_ew.get_abstract()

                    sing.scratch.beval_print(f'[apply_nn]')
                    return Examplewise(abstract=sing.model.apply_nn(fn_embed_vec, labelled_args_vecs, parent_vec=output_ew_vec))
                
                new_output_ew = self.pnode_cache.inverse_app(self,no_cache,output_ew,fn_embed,labelled_args,zipper[0])

                res = self.xs[zipper[0]].inverse_beval(ctx, output_ew=new_output_ew, zipper=zipper[1:])
                return printed(res)

            else:
                raise TypeError

    # def get_concrete_subtrees(self):
    #     if self.ntype.output:
            
    #     elif self.ntype.abs:
    #         assert False
    #     elif self.ntype.prim:
    #         assert False
    #     elif self.ntype.var:
    #         assert False
    #     elif self.ntype.hole:
    #         assert False
    #     elif self.ntype.app:
    #         assert False
    #     else:
    #         raise TypeError
    
    @cached_propagate
    def propagate(self, towards, concrete_only=False):
        assert False
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


            if sing.model.cfg.apply_mode == 'nmns':
                assert self.fn.ntype.prim, "Limitation, was there in old abstract repl too, can improve upon when it matters"
                nm = sing.model.abstract_transformers.fn_nms[self.fn.name][propagation_direction]
                return Examplewise(nm(*[arg.abstract() for arg in evaluated_args]))
            elif sing.model.apply_mode == 'apply_nn':
                # get the direction numbers
                directions = [dir for dir,node in enumerate(possible_args) if node is not towards]
                directed_args = [(exwise.abstract(),dir) for exwise,dir in zip(evaluated_args,directions)]
                # TODO you gotta actually have a way of embedding a functino
                self.fn.propagate(self).abstract()
            else:
                assert False



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
        assert False
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
        assert False
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
    def reset_cache(self,recursive=True):
        self.pnode_cache.clear()
        if recursive:
            for c in self.children():
                c.reset_cache(recursive=True)

    def get_zipper(self):
        """
        Get a zipper to yourself from your self.root()

        Zipper looks like ('tree','body','body',1,2,1) for example where ints are used for which arg of a fn ur talking about
        """
        root = self.root()
        if self is root:
            return ()
        parent = self.parent
        if parent.ntype.output:
            attr = 'tree'
        elif parent.ntype.abs:
            attr = 'body'
        elif parent.ntype.hole or parent.ntype.prim or parent.ntype.var:
            raise TypeError
        elif parent.ntype.app:
            if parent.fn is self:
                attr = 'fn'
            for i,arg in enumerate(parent.xs):
                if arg is self:
                    attr = i
        elif parent.ntype.abs:
            attr = 'body'
        elif parent.ntype.abs:
            attr = 'body'
        else:
            raise TypeError
        
        return (*parent.get_zipper(),attr)
    def apply_zipper(self, zipper):
        """
        Returns the node retrieved by the zipper
        """
        if len(zipper) == 0:
            return self
        if isinstance(zipper[0],str):
            return getattr(self,zipper[0]).apply_zipper(zipper[1:])
        elif isinstance(zipper[0],int):
            return self.xs[zipper[0]].apply_zipper(zipper[1:])
        else:
            raise TypeError

    def clone(self, no_cache_copy=False):
        """

        [description out of date]

        Clone the tree from self.root() down, and return the node corresponding to `self` thats in
        the newly cloned tree.
            * no pointers in the newly cloned tree will point to PNodes in the old tree
                ie a new copy of every PNode will be made, and all parent and child pointers
                will be adjusted to point to things in this new tree.
            * the original tree will be unaffected
            * this is cache safe, it'll actually copy over the same `cache.exwise` and adjust the `cache.towards` properly
                this doesnt duplicate the exwise object, it just means now both new and old trees point to the same one.
                since exwises dont change inplace (beyond turning their .concrete into a .abstract()) this will be fine! As
                soon as one of the caches invalidates itll just change which exwise it points to and wont disrupt the other
                tree.
            * the PTask will be shared (even if no_cache_copy=True)
        """
        zipper = self.get_zipper() # so we can find our way back to this node in the new tree
        root = self.root()
        assert root.ntype.output

        cloned_root = PNode.from_ptask(root.task) # share same ptask (includes cache)
        cloned_root.expand_from(root)
        if not no_cache_copy:
            cloned_root.copy_cache_from(root, recursive=True)

        cloned_self = cloned_root.apply_zipper(zipper)
        assert self.marked_str() == cloned_self.marked_str()
        return cloned_self

    def copy_cache_from(self, other, recursive=True):
        self.pnode_cache = other.pnode_cache.clone()
        if recursive:
            for c,o in zip(self.children(),other.children()):
                c.copy_cache_from(o,recursive=True)

        # assert False
        # assert self.ntype == other.ntype
        # assert self.marked_str() == other.marked_str(), "maaaaybe you can relax this but be careful. Esp given that cache.string is getting copied (tho u could change that)"
        # cache = other.cache
        # if cache.towards is None:
        #     self.cache = Cache(None,None,None)
        #     return

        # new_towards = None
        # if cache.towards is other:
        #     new_towards = self
        # elif cache.towards is other.parent:
        #     new_towards = self.parent
        # else:
        #     for i,c in enumerate(other.children()):
        #         if cache.towards is c:
        #             new_towards = self.children()[i]
        # assert new_towards is not None
        # self.cache = Cache(new_towards,cache.exwise,cache.string)
        # if recursive:
        #     for c,o in zip(self.children(),other.children()):
        #         c.copy_cache_from(o,recursive=True)
            
    def hide(self,recursive=False):
        """
        Turn whole tree into Holes but without destroying the old data
        But dont do this for abstractions + output nodes bc it's impossible to "guess" these
        and theyre autofilled during search.
        Also for applications we dont hide the .fn field (which we assume to be a primitive)
        We also dont touch the cache
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
        assert False
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
