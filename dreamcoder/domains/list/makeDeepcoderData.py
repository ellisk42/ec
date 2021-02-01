import sys
import mlb
import numpy as np
import os
from hydra import utils
from dreamcoder.Astar import Astar
from dreamcoder.SMC import SMC
import itertools
from torch import nn
#import mlb
import contextlib
import multiprocessing as mp
import queue
import contextlib
# sys.path.append(ospath.abspath('./'))
# sys.path.append(os.path.abspath('./ec'))

import pickle
from dreamcoder import valueHead
#from util.algolisp_util import make_holey_algolisp
#from util.deepcoder_util import basegrammar
import time
from collections import namedtuple
#Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])

from dreamcoder.domains.misc.deepcoderPrimitives import int_to_int, int_to_bool, int_to_int_to_int, deepcoderPrimitivesPlusPlus, get_lambdas

if __name__ == '__main__':
    try:
        import binutil  # required to import from dreamcoder modules
    except ModuleNotFoundError:
        import bin.binutil  # alt import if called as module

#from dreamcoder.domains.list.main import main, list_options
#from dreamcoder.dreamcoder import commandlineArguments
#from dreamcoder.utilities import numberOfCPUs

from dreamcoder.task import Task


from dreamcoder.utilities import flatten

import math
import random
from itertools import zip_longest, chain, islice
from functools import reduce
import torch

from dreamcoder.domains.list.dc_program import generate_IO_examples, compile
from dreamcoder.domains.list.deepcoder_util import parseprogram, make_holey_deepcoder
from dreamcoder.grammar import Grammar, NoCandidates
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderProductions, flatten_program
from dreamcoder.program import Application, Hole, Primitive, Index, Abstraction, ParseFailure
from dreamcoder.type import Context, arrow, tint, tlist, tbool, UnificationFailure
from dreamcoder.matt.util import *

def convert_dc_program_to_ec(dc_program, tp):
    source = dc_program.src
    source = source.split('\n')
    source = [line.split(' ') for line in source]
    # print(source)
    num_inputs = len(tp.functionArguments())
    # print(num_inputs)
    del source[:num_inputs]
    source = [[l for l in line if l != '<-'] for line in source]
    last_var = source[-1][0]
    prog = source[-1][1:]
    del source[-1]
    variables = list('abcdefghigklmnop')
    del variables[variables.index(last_var):]  # check this line
    lookup = {variables[i]: ["input_" + str(i)] for i in range(num_inputs)}
    for line in source:
        lookup[line[0]] = line[1:]
    for variable in reversed(variables):
        p2 = []
        for x in prog:
            if x == variable:
                p2 += lookup[variable]
            else:
                p2.append(x)
        prog = p2
    return prog


def task_of_line(line, N, L, V, num_tasks=1):
    line = line.replace(' | ', '\n')
    # compile(L) just uses L for bounds. We'll just feed in the max value
    # in L bc thats close enough, and we double check our input examples with rejection
    # sampling later anyways
    dc_program = compile(line, V=V, L=(L if isinstance(L,int) else max(L)))

    if dc_program is None:
        return None,None

    # find IOs
    IO = [tuple(generate_IO_examples(dc_program, N=N, L=L, V=V)) for _ in range(num_tasks)]

    # find tp
    ins = [tint if inp == int else tlist(tint) for inp in dc_program.ins]
    if dc_program.out == int:
        out = tint
    else:
        assert dc_program.out == [int]
        out = tlist(tint)
    tp = arrow(*(ins+[out]))

    # find program p
    pseq = tuple(convert_dc_program_to_ec(dc_program, tp))
    p = parseprogram(pseq, tp)
    tasks = [Task(str(p), tp, io) for io in IO]
    return p, tasks


_pp = deepcoderPrimitivesPlusPlus()
prims_pp = {prim.name:prim for prim in _pp}
g_pp = Grammar.uniform(_pp)
for name,prim in prims_pp.items():
    Primitive.GLOBALS[name] = prim # so Program.parse() etc will use it

class ToPlusPlusVisitor:
    def __init__(self):
        super().__init__()
        self.has_lambda = False
    def primitive(self,prim):
        if '_to_' not in prim.tp.name: #having a "_to_" in the type means its a lambda like int_to_int
            return prims_pp[prim.name] # convert to plus plus version of primitive (might be pointless)
        self.has_lambda = True
        new_tp = {
            int_to_int:arrow(tint,tint),
            int_to_int_to_int:arrow(tint,tint,tint),
            int_to_bool:arrow(tint,tbool),
        } [prim.tp]
        hole = Hole(tp=new_tp)
        return hole
    def invented(self,e):
        assert False
    def index(self,e):
        return e
    def application(self,e):
        return Application(e.f.visit(self),e.x.visit(self))
    def abstraction(self,e):
        return Abstraction(e.body.visit(self))



from collections import defaultdict
def preprocess(test_frontiers,cfg):
    """
    Filters out tasks that we don't want, e.g. ones with out of range values, etc
    """
    filtered = []
    reqs = defaultdict(int)
    numbers = defaultdict(int)
    global_min = 10000
    global_max = -100000
    seen = set()

    print(f"{len(test_frontiers)=}")

    for f in test_frontiers:
        reqs[f.t.request] += 1

        if f.t.request != arrow(tlist(tint),tlist(tint)):
            continue

        outputs = [o for [i],o in f.t.examples]
        inputs = [i for [i],o in f.t.examples]
        if all([len(o)==1 for o in outputs]):
            print("[TASK] rejecting singleton")
            continue

        new_exs = []
        for ex in f.t.examples:
            [i],o = ex
            if len(i) < 1 or len(i) > 10:
                print("reject ex bc input too long")
                continue
            if max(i+o,default=global_max) > global_max:
                global_max = max(i+o)
            if min(i+o,default=global_min) < global_min:
                global_min = min(i+o)
            if min(i+o) < -cfg.data.test.V:
                print("reject ex bc small")
                continue
            if max(i+o) > cfg.data.test.V:
                print("reject ex bc big")
                continue
            new_exs.append(ex)
        new_exs = new_exs[:5]
        if len(new_exs) < 5:
            print("[TASK] rejecting bc too few valid examples")
            continue
        f.t.examples = new_exs

        name = f.t.name
        if name[:3].isdigit() and len(name) == 5 and name[4].isdigit():
            # name is like 004_2 and we wanna throw out everything beyond *_1
            name = name[:3] # for `seen` purposes

        if name in seen:
            print(f"[TASK] skipping bc already seen this task: {f.t.name}")
            continue

        seen.add(name)

        filtered.append(f)

            # if max(i,default=0) > 16:
            #     mlb.red(f"throwing out a big boy {max(i)}")
            #     continue
            # if min(i,default=0) < -16:
            #     mlb.red(f"throwing out a negative big boy {min(i)}")
            #     continue
        


        # filter request type
        # if not cfg.data.test.allow_complex_requests:
        #     # only [int] -> [int] allowed
        #     if task.request != arrow(tlist(tint),tlist(tint)):
        #         continue
        # else:
        #     # only [int] -> [int] or [int] -> int allowed
        #     if task.request not in [arrow(tlist(tint),tlist(tint)), arrow(tlist(tint),tint)]:
        #         continue
        
        # adjust number of examples
        # for ([i],o) in task.examples:
        #     print(len(i), end=' ')

        #print()
    for k,v in reqs.items():
        print(f"{k}: {v}")
    print(f"{global_min=}")
    print(f"{global_max=}")
    print(f"{len(filtered)=}")
    mlb.cyan(f"VERY IMPORTANT: you are pruning examples during preprocessing and search based on this limit: {cfg.data.test.V}")
    return filtered





def get_primitive(app):
    if app.isPrimitive:
        return app
    if app.isApplication:
        return get_primitive(app.f)
    if app.isAbstraction:
        assert False, "Abstractions shouldnt be applied in this DSL, theyre only ever either top level or passed into higher order functions"
    assert False

def has_index(e, i):
    if e.isIndex:
        if i is None:
            return True
        return e.i == i # true if the index is right else false
    if e.isPrimitive:
        return False
    if e.isAbstraction:
        return has_index(e.body,i)
    if e.isApplication:
        return has_index(e.f,i) or has_index(e.x,i)
    assert False

def is_constant(e):
    return e.isPrimitive and e.value in list(range(-10,10))

def insert_index(e,i):
    """
    Replaces a single numeric constant in the tree with an index (uniformly)
    Note that this constant will always be an argument to an application (by nature of the DSL)
    Also note that this recurses into Abstractions directly without incrementing i or anything like that

    Logic:
        - count number of constants in e
        - traverse e (in any order) and each time you see a new constant replace it with an index
            with probability (1/unseen_constants). This will end up being a uniform distribution.
    """
    num_constants = count_constants(e)
    if num_constants == 0:
        raise InvalidSketchError("no constants to replace with indices")
    unseen_constants = num_constants # reamining_constants will be a shared nonlocal value

    def helper(e): # recursive helper
        nonlocal unseen_constants
        assert unseen_constants > 0
        if e.isIndex or e.isPrimitive:
            return False # ignore these. Note that the primitive constant case is handled by the abtraction or application above it.
        assert e.isApplication or e.isAbstraction
        if e.isAbstraction and not is_constant(e.body):
            return helper(e.body) # simply recurse on body
        elif e.isApplication and not is_constant(e.x):
            return helper(e.f) or helper(e.x) # note that actually both f and x can be complicated subtrees so we must recurse on both

        # we found a constant! Now let's flip a weighted coin
        if random.random() <= 1 - (1/unseen_constants):
            # 1/unseen_constants is the probability that we accept. This if-branch is the rejection branch.
            unseen_constants -= 1 # we've seen a new constant
            if e.isApplication:
                return helper(e.f) # still need to recurse on f, though e.x and e.body are clearly constants so no recursion needed
            return # no need to recurse, everything around us is constants
        
        # we're changing the constant!
        if e.isAbstraction:
            e.body = Index(i)
        else:
            e.x = Index(i)
        unseen_constants = None # ensure an error if we do anything other than recursing all the way out
        return True
    
    success = helper(e)
    assert success
    assert count_constants(e) == num_constants-1
    assert has_index(e,i)


def count_constants(e):
    if e.isIndex:
        return 0
    elif e.isPrimitive:
        if is_constant(e):
            return 1
        return 0
    elif e.isAbstraction:
        return count_constants(e.body)
    elif e.isApplication:
        return count_constants(e.f) + count_constants(e.x)
    else:
        raise ValueError

def verify_tree(e):
    """
    verifies that a Program has:
        - for each lambda, there's an index in the subtree (we don't actually check the index number)
        - we also handle (lambda (lambda ...)) must contain both $0 and $1
            - note we specifically handle this exact case where the body of the outer abstraction is itself an abstraction, it wont work
                for any more complicated ways of nesting lambdas. The entire body of the first lambda must be the second.

    """
    if e.isIndex:
        return
    if e.isPrimitive:
        return
    if e.isAbstraction:
        # all abstractions need to use their variable somewhere
        # this handles both lambdas like those that are passed to MAP
        # as well as top level lambdas (tho those should already be fine)
        if not has_index(e,0):
            insert_index(e,0)
        if e.body.isAbstraction: # lambda( lambda( ...))
            if not has_index(e,1):
                insert_index(e,1)
        if e.body.isIndex: # the identity lambda function
            raise InvalidSketchError("identity lambda")
        verify_tree(e.body)
        return
    if e.isApplication:
        verify_tree(e.f)
        verify_tree(e.x)
        return
    assert False


def check_in_range(res,V):
    """
    check if an int, [int], or [[int]] has all values within the range [-V,V]
    Raises an InvalidSketchError if this is not the case
    """
    if isinstance(res,(int,np.integer)):
        if res > V or res < -V:
            #mlb.yellow(f'rejecting sketch bc concrete evaluation out of range: {res}')
            raise InvalidSketchError(f"integer outside a list had value {res}")
    if isinstance(res,(list,tuple)):
        if len(res) == 0:
            return
        if isinstance(res[0],(list,tuple)):
            return [check_in_range(x,V) for x in res]
        maxx = max(res)
        minn = min(res)
        if maxx > V or minn < -V:
            #mlb.yellow(f'rejecting sketch bc concrete evaluation out of range: list min={minn}; list max={maxx}')
            raise InvalidSketchError(f"{minn} {maxx}")

def ctxs_of_examples(examples):
    """
    convert task.examples to a list of contexts that evaluate() will take
    """
    inputs = [ex[0] for ex in examples] # nested list w shape (num_examples,argc)
    ctxs = tuple([list(reversed(args)) for args in inputs])
    return ctxs

def strip_lambdas(sk):
    i = 0
    while sk.isAbstraction:
        sk = sk.body
        i += 1
    return sk,i



def concrete_rep(sk,task,ctxs,in_lambda,V):
    """
    Basically rep() modified to not do the neural bit. You recurse on lambdas and children.

    ctxs :: a list of tuples. The outer list iterates over examples, and the
        inner tuple is a context where the 0th thing is the value of $0 etc.
        Pass in None initially to initialize it. Note that this isn't a "default"
        argument for ctxs because that would make it very easy to forget to
        pass in the existing one when recursing.
    returns :: Tensor[num_exs,H] or a list of concrete values (one per example)
        in which case the type is [int] or [[int]] or [bool] where the outermost list
        is always iterating over examples.
    """

    # first, if this was called at the top level (ctx=None),
    # we clear out as many abstractions as there are top level inputs
    if ctxs is None: # pull initial context out of the task inputs
        assert not in_lambda
        ctxs = ctxs_of_examples(task.examples)
        sk,num_lambdas = strip_lambdas(sk)
        assert len(ctxs[0]) == num_lambdas, "Mismatch between num args passed in and num lambda abstractions"

    if sk.isHole:
        return

    if not sk.hasHoles:
        if not (in_lambda and has_index(sk,None)):
            # we dont run this if we're inside a lambda and we contain an index
            # since those can't be concrete evaluated in a lambda
            # in that lambda case we do wanna just continue to the Application branch and not return early tho!
            res = evaluate_ctxs(sk,ctxs,V)
            # if sk.size() > 1:
            #     #print(f"ran concrete eval on sk of size {sk.size()}: {sk}")
            #     if self._curr_task_concrete_count == task:
            #         self.concrete_count += sk.size()
            return
    
    if sk.isPrimitive:
        return
    if sk.isIndex:
        return

    if sk.isAbstraction:
        #assert not in_lambda, "nested lambda should never happen"
        sk,i = strip_lambdas(sk)
        assert i <= 2
        return concrete_rep(sk,task,ctxs,True,V)

    if sk.isApplication:
        fn, args = sk.applicationParse()
        assert len(args) > 0
        # recurse on children
        for arg in args:
            concrete_rep(arg,task,ctxs,in_lambda,V)
        concrete_rep(fn,task,ctxs,in_lambda,V)
        return
    if fn.isAbstraction:
        assert False
    assert False

    
def evaluate_ctxs(e, ctxs, V):
    """
    evaluate multiple contexts (e.g. all examples)
    """
    try:
        res = [evaluate(e,ctx, V) for ctx in ctxs]
    except (ZeroDivisionError,FloatingPointError):
        raise InvalidSketchError("zerodiv")
    return res

def evaluate(e, ctx, V):
    """
    like Program.evaluate() but calls check_in_range() on each intermediate result
    """
    if e.isIndex:
        res = ctx[e.i]
    elif e.isAbstraction:
        res = lambda x: evaluate(e.body,[x] + ctx, V)
    elif e.isPrimitive:
        res = e.value
    elif e.isApplication:
        res = evaluate(e.f,ctx,V)(evaluate(e.x,ctx, V))
    else:
        assert False, "should never happen"
    check_in_range(res,V) # may raise InvalidSketchError
    return res

class InvalidSketchError(Exception): pass

class FakeRecognitionModel(nn.Module):
    # pretends to be whatever Astar wants from its RecognitionModel. Which isn't much lol
    def __init__(self,valueHead,policyHead):
        super().__init__()
        self.policyHead = policyHead
        self.valueHead = valueHead

def depth_nolambda(e):
    """
    Depth in the T1/T2/T3 sense
    """
    while e.isAbstraction:
        e = e.body

    def helper(e):
        if e.isIndex:
            return 0
        elif e.isAbstraction:
            return 0
        elif e.isPrimitive:
            return 0
        elif e.isHole:
            return 0
        elif e.isApplication:
            f, xs = e.applicationParse() # important bc actually changes depth
            return 1 + max([helper(f)] + [helper(x) for x in xs])
        assert False
    return helper(e)

def depth_lambda(e):
    """
    Depth in the lambda_depth=1,2,3 sense
    """
    while e.isAbstraction:
        e = e.body

    def helper_outside_lambda(e):
        if e.isIndex:
            return 0
        elif e.isAbstraction:
            while e.isAbstraction:
                e = e.body # burrow into the lambda
            return helper_inside_lambda(e)
        elif e.isPrimitive:
            return 0
        elif e.isHole:
            return 0
        elif e.isApplication:
            f, xs = e.applicationParse() # important bc actually changes depth
            return max([helper_outside_lambda(f)] + [helper_outside_lambda(x) for x in xs])
        assert False
    def helper_inside_lambda(e):
        if e.isIndex:
            return 1
        elif e.isAbstraction:
            assert False
        elif e.isHole:
            return 1
        elif e.isPrimitive:
            return 1
        elif e.isApplication:
            f, xs = e.applicationParse() # important bc actually changes depth
            return 1 + max([helper_inside_lambda(f)] + [helper_inside_lambda(x) for x in xs])
        assert False
    return helper_outside_lambda(e)

def depth_for_solver(e):
    """
    Depth in the astar or smc maximumDepth sense
    """

    def helper(e):
        if e.isIndex:
            return 0
        elif e.isAbstraction:
            return helper(e.body) # ignores abstractions
        elif e.isPrimitive:
            return 0
        elif e.isHole:
            return 0
        elif e.isApplication:
            # doesnt do an applicationParse
            return 1 + max([helper(e.f),helper(e.x)])
        assert False
    return helper(e)
    

def get_depth(p):
    T = depth_nolambda(p)
    d = depth_lambda(p)
    astar = depth_for_solver(p)
    return T,d,astar


class FakeFrontier:
    # pretends to be whatever valueLossFromFrontier wants for simplicity
    def __init__(self,program,task, scaffold=None):
        self.task = task # satisfies frontier.task call
        self._fullProg = program
        self.program = self # trick for frontier.sample().program._fullProg
        self.scaffold = scaffold

    # for my own use
    @property
    def p(self):
        return self._fullProg
    @property
    def t(self):
        return self.task
    def sample(self):
        return self

from dreamcoder.matt.sing import sing

def dcfg():
    return sing.cfg.data

class DeepcoderTaskloaderInner:
    def __init__(self,mode):
        cfg = dcfg()
        self.mode = mode

        if cfg.L is None:
            assert cfg.L_min is not None and cfg.L_max is not None
            self.L = list(range(cfg.L_min,cfg.L_max+1))
            self.L_big = max(self.L)
        else:
            assert cfg.L_min is  None and cfg.L_max is  None
            self.L = cfg.L
            self.L_big = self.L
        
        self.premade_templates = None
        if dcfg().premade_templates is not None:
            self.premade_templates = torch.load(utils.to_absolute_path('list_tests/'+dcfg().premade_templates))
            #self.premade_templates = itertools.cycle(self.premade_templates)
            self.premade_i = 0
            mlb.purple(f'using templates for {self.mode}')


        self.threaded = sing.cfg.loader.threaded
        assert not self.threaded, "We're not allowing threading for now"
        
        # if cfg.t4:
        #     self.file = utils.to_absolute_path(f'dreamcoder/domains/list/DeepCoder_data/T4_A2_V512_L10_train_perm.txt')
        # else:
        self.file = utils.to_absolute_path(f'dreamcoder/domains/list/DeepCoder_data/T{cfg.T}_A2_V512_L10_{mode}_perm.txt')
        self.allowed_requests = [arrow(tlist(tint),tlist(tint)), arrow(tlist(tint),tint)] if dcfg().allow_complex_requests else [arrow(tlist(tint),tlist(tint))]

        self.buf = []
        self.templates_seen = 0 # number of tasks seen (pre mutation)

        with open(self.file,'r') as f:
            f.readline() # skip first line of file
            self.offset_in_file = f.tell()
            self.file_start = self.offset_in_file
        
        # make a lambda grammar to sample lambdas from
        self.g_lambdas = Grammar.uniform(get_lambdas(), max_hole_depth= cfg.lambda_depth)

    def reloadBuffer(self):
        """
        Refills the buffer.
            If `cfg.repeat=True` -> will loop back to start of file at EOF (and will keep filling buffer). Otherwise EOFError will be raised.
            `dcfg().num_mutated_tasks` gives the number of mutations of each template that will be given in a row.
        How much the buffer will be filled:
            It'll be filled up to `sing.cfg.loader.buf_size`
        """
        with open(self.file,'r') as f:
            f.seek(self.offset_in_file) # pick up where we left off
            while True: # loops until `queue.Full` gets raised by the .put() line (or buf.full() happens)
                try:
                    if len(self.buf) >= sing.cfg.loader.buf_size:
                        assert len(self.buf) == sing.cfg.loader.buf_size, "bug in code"
                        return
                    
                    # note we can't use next(f) as this disables f.tell(), so we do f.readline()
                    line = f.readline() # do NOT call .strip() on this until after the EOF check below!

                    # EOF
                    if line == '': # readline never errors out, it returns empty string on EOF (as opposed to when it encounters a normal empty line in which case in returns '\n')
                        if sing.cfg.loader.repeat:
                            f.seek(self.file_start) # repeat
                            continue
                        else:
                            raise EOFError
                    line = line.rstrip()

                    # get program and task
                    # purposefully self.L not dcfg().L
                    program,tasks = task_of_line(line,N=dcfg().N,L=self.L,V=dcfg().V, num_tasks=dcfg().num_mutated_tasks)
                    if program is None:
                        continue
                    if hasattr(self,'premade_templates') and self.premade_templates is not None:
                        program = self.premade_templates[self.premade_i]
                    ff = FakeFrontier(program,tasks[0])

                    task = tasks[0]

                    # filter out bad programs/tasks
                    if self.allowed_requests is not None and task.request not in self.allowed_requests:
                        continue
                    if all([ex[0] == task.examples[0][0] for ex in task.examples]):
                        continue # for some reason this happens sometimes (all inputs are the same)
                    try:
                        check_in_range(task.examples,dcfg().V)
                    except InvalidSketchError:
                        continue

                    self.templates_seen += 1 # number of templates seen

                    # deepcoder++ conversion
                    if dcfg().expressive_lambdas:
                        frontiers = self.convert_to_deepcoder_plus_plus(ff,tasks)
                        if frontiers is None:
                            continue
                    else:
                        print(ff.p)

                        for [i],o in ff.t.examples:
                            print(f"\t{i} -> {o}")
                        frontiers = [ff]

                    # add to buffer and potentially exit by throwing exception (this is the only exit point)
                    for frontier in frontiers:
                        self.buf.append(frontier)
                        if hasattr(self,'premade_templates') and self.premade_templates is not None:
                            self.premade_i += 1
                            if self.premade_i >= len(self.premade_templates):
                                self.premade_i = 0
                        if len(self.buf) == sing.cfg.loader.buf_size:
                            return
                        #print(f"buf {self.mode}:",self.buf.qsize())
                    
                    # if we've seen the first `cfg.num_templates` programs in the file (pre-mutation)
                    # then loop back to the start of the file
                    if sing.cfg.loader.max_tasks is not None and self.templates_seen == dcfg().num_templates:
                        f.seek(self.file_start)
                finally:
                    self.offset_in_file = f.tell()

    def getTask(self):
        ret = self.getTasks(n=1)
        if len(ret) == 0:
            raise ValueError("Out of tasks, can't getTask()")
        return ret

    def getTasks(self, n=None):
        """
        if n is None: reload buf and return `sing.cfg.loader.buf_size` tasks
        else: return n tasks (even if n>buf_size)
        Note that it may return fewer tasks if repeat=False and we hit EOF
        """
        # If n is none, return `sing.cfg.loader.buf_size` items
        if n is None:
            #with cm:
            with contextlib.suppress(EOFError):
                self.reloadBuffer()
                assert len(self.buf) > 0
            ret = self.buf[:]
            self.buf = []
            print(f"yielding {len(ret)} tasks")
            return ret
        
        # If buf is longer than n (or equal), we dont need to reload and can just return
        if n <= len(self.buf):
            ret = self.buf[:n]
            self.buf = self.buf[n:]
            print(f"yielding {len(ret)} tasks")
            return ret
        
        # buf is smaller than n so we need to keep reloading
        ret = []
        remaining = n
        while True:
            ret += self.buf[:remaining]
            self.buf = []
            remaining = n-len(ret)
            if remaining <= 0:
                assert remaining == 0, "bug in code"
                return ret
            with contextlib.suppress(EOFError):
                self.reloadBuffer()
            if len(self.buf) == 0:
                mlb.yellow(f"warning: ran out of tasks and repeat=False, returning fewer tasks than requested")
                print(f"yielding {len(ret)} tasks")
                return ret
    def convert_to_deepcoder_plus_plus(self, f, tasks):
        program = f.p

        visitor = ToPlusPlusVisitor()
        if not hasattr(self,'premade_templates') or self.premade_templates is None:
            program_plus_plus = program.visit(visitor)
            if not visitor.has_lambda:
                # if the program has no lambdas to mutate we should ignore `n`
                # and just return the unmodified program in a singleton list
                return None # actually lets just only allow programs wiht lambdas
                return [f]
        else:
            program_plus_plus = program

        assert program_plus_plus.hasHoles
        res = []

        num_generated = 0
        assert dcfg().num_mutated_tasks == 1, "just bc the failure stuff i guess"
        failures = -1
        while True:
            failures += 1

            if failures > 100:
                print(f"Giving up on {program_plus_plus}")
                return None

            task = tasks[num_generated]
            assert self.g_lambdas.max_hole_depth is not None
            sampled = self.g_lambdas.sampleFromSketch(task.request, program_plus_plus, maximumDepth = 20) # this max depth wont be hit bc of Grammar.max_hole_depth
            if sampled.size() > 30:
                #breakpoint()
                print("you got some big trees there man")
            try:
                verify_tree(sampled)
            except InvalidSketchError:
                #print(f"rejecting {sampled} bc missing index or has (lambda $0) identity")
                continue # rejection sample

            # calculate correct outputs for inputs now that we've modified the program
            assert not sampled.hasHoles
            ctxs = ctxs_of_examples(task.examples)
            try:
                #outputs = valueHead.ListREPLValueHead.rep(fake_vhead, sampled, task, None)
                stripped,num_lambdas = strip_lambdas(sampled)
                assert len(ctxs[0]) == num_lambdas, "Mismatch between num args passed in and num lambda abstractions"
                outputs = evaluate_ctxs(stripped,ctxs,dcfg().V)
            except InvalidSketchError:
                #print(f"rejecting {sampled} bc out of range values or zero division")
                continue # e.g. division by zero during concrete eval

            # check if its a constant function (output same for any input)
            if all([output == outputs[0] for output in outputs[1:]]):
                #print(f"rejecting {sampled} bc output is constant")
                continue # rejection sample

            # if its a [int] -> [int] function, check if its the identity
            if task.request == arrow(tlist(tint),tlist(tint)):
                # note ex is an input,output tuple, ex[0] is the tuple of input arguments which is a singleton list in this case so we do ex[0][0] to get the actual [int] input
                if all([ex[0][0] == output for ex,output in zip(task.examples,outputs)]):
                    #print(f"rejecting {sampled} bc it's the identity function")
                    continue # rejection sample

            new_examples = [(ex[0],output) for ex,output in zip(task.examples,outputs)]
            new_task = Task(str(sampled), task.request, new_examples)
            res.append(FakeFrontier(sampled,new_task,scaffold=get_scaffold(program_plus_plus)))
            #mlb.green(f"accepting {sampled}")
            if dcfg().print_data:
                print(f"accepting {sampled}")
                for ex in new_task.examples:
                    print(f"\t{ex[0][0]} -> {ex[1]}")
            num_generated += 1
            if num_generated >= dcfg().num_mutated_tasks:
                break
        assert len(res) == dcfg().num_mutated_tasks
        return res


def get_scaffold(e):
    if e.isIndex:
        return e
    elif e.isAbstraction:
        return Abstraction(get_scaffold(e.body))
    elif e.isPrimitive:
        return e
    elif e.isHole:
        #print(e.tp)
        if not e.tp.isArrow():
            return e
        e.tp = e.tp.arguments[1] # for int->int->bool this shd be int->bool
        return Abstraction(get_scaffold(e))
    elif e.isApplication:
        # doesnt do an applicationParse
        return Application(get_scaffold(e.f),get_scaffold(e.x))
    assert False

if __name__ == '__main__':
    pass
