# generate deepcoder data
import sys
import mlb
import numpy as np
import os
#import mlb
import contextlib
# sys.path.append(os.path.abspath('./'))
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


def task_of_line(line, N=5, L=10, V=63):
    line = line.replace(' | ', '\n')
    dc_program = compile(line, V=V, L=L)

    if dc_program is None:
        return None,None

    # find IO
    IO = tuple(generate_IO_examples(dc_program, N=N, L=L, V=V))

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
    task = Task(str(p), tp, IO)
    return p, task


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


def get_primitive(app):
    if app.isPrimitive:
        return app
    if app.isApplication:
        return get_primitive(app.f)
    if app.isAbstraction:
        assert False, "Abstractions shouldnt be applied in this DSL, theyre only ever either top level or passed into higher order functions"
    assert False
# class PostprocessVisitor:
#     def __init__(self, g_lambdas):
#         super().__init__()
#         self.g_lambdas = g_lambdas
#     def primitive(self,prim):
#         #if prim.name in [str(num) for num in range(-10,10)]:
#         return False
#     def invented(self,e):
#         assert False
#     def index(self,e):
#         return True
#     def application(self,e):
#         p = get_primitive(e)
#         if p in self.g_lambdas.primitives:
#             if p.name in ['+','*']:
#                 # these primitives can have arbitrary children
#                 # so we just ignore them and recurse on them
#                 return e.f.visit(self) or e.x.visit(self)
#             elif p.name in ['MIN','MAX','>','DIVISIBLE','AND','OR']:
#                 # these primitives need at least one $0 one of their subtrees
#                 pass
#             else:
#                 assert False, "please add the new primitive to a branch in this function"
#         return e.f.visit(self) or e.x.visit(self)
#     def abstraction(self,e):
#         return Abstraction(e.body.visit(self))



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
        raise InvalidSketchError
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
            raise InvalidSketchError()
        verify_tree(e.body)
        return
    if e.isApplication:
        #f,xs = e.applicationParse()
        verify_tree(e.f)
        verify_tree(e.x)
        #prim = get_primitive(e)
        # if prim.name in ['MIN','MAX','>','DIVISIBLE','AND','OR']:
        #     # these primitives must have a $0 in their subtree
        #     if not verified:
        #         raise NeedsIndexException()

        return
    assert False


def check_in_range(res):
    """
    check if all the values in a len(num_examples) list of values (eg ints or int lists) are within the range -99,99.
    Raises an InvalidSketchError if this is not the case
    """
    if isinstance(res,(int,np.integer)):
        if res > 99 or res < -99:
            mlb.yellow(f'rejecting sketch bc concrete evaluation out of range: {res}')
            raise InvalidSketchError
    if isinstance(res,(list,tuple)):
        if len(res) == 0:
            return
        assert not isinstance(res[0],(list,tuple)), "check_in_range should not be used examplewise"
        maxx = max(res)
        minn = min(res)
        if maxx > 99 or minn < -99:
            mlb.yellow(f'rejecting sketch bc concrete evaluation out of range: list min={minn}; list max={maxx}')
            raise InvalidSketchError


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
    
def evaluate_ctxs(e, ctxs):
    """
    evaluate multiple contexts (e.g. all examples)
    """
    try:
        res = [evaluate(e,ctx) for ctx in ctxs]
    except (ZeroDivisionError,FloatingPointError):
        raise InvalidSketchError
    return res

def evaluate(e, ctx):
    """
    like Program.evaluate() but calls check_in_range() on each intermediate result
    """
    if e.isIndex:
        res = ctx[e.i]
    elif e.isAbstraction:
        res = lambda x: evaluate(e.body,[x] + ctx)
    elif e.isPrimitive:
        res = e.value
    elif e.isApplication:
        res = evaluate(e.f,ctx)(evaluate(e.x,ctx))
    else:
        assert False, "should never happen"
    check_in_range(res) # may raise InvalidSketchError
    return res


class InvalidSketchError(Exception): pass


def convert_to_deepcoder_plus_plus(program,task, n, g_lambdas, mutate=True):
    assert mutate

    visitor = ToPlusPlusVisitor()
    program_plus_plus = program.visit(visitor)
    if not visitor.has_lambda:
        # if the program has no lambdas to mutate we should ignore `n`
        # and just return the unmodified program in a singleton list
        return [(program_plus_plus,task)]

    assert program_plus_plus.hasHoles
    res = []

    while True:
        sampled = g_lambdas.sampleFromSketch(arrow(tlist(tint), tlist(tint)), program_plus_plus, maximumDepth = 20) # this max depth wont be hit bc of Grammar.max_hole_depth
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
            outputs = evaluate_ctxs(stripped,ctxs)
        except InvalidSketchError:
            #print(f"rejecting {sampled} bc out of range values or zero division")
            continue # e.g. division by zero during concrete eval


        # check if its a constant function (output same for any input)
        if all([output == outputs[0] for output in outputs[1:]]):
            #print(f"rejecting {sampled} bc output is constant")
            continue # rejection sample

        # check for really large or small numbers
        # if max([max(output) for output in outputs]) > 99 or min([min(output) for output in outputs]) < -99:
        #     continue # rejection sample

        # if its a [int] -> [int] function, check if its the identity
        if task.request == arrow(tlist(tint),tlist(tint)):
            # note ex is an input,output tuple, ex[0] is the tuple of input arguments which is a singleton list in this case so we do ex[0][0] to get the actual [int] input
            if all([ex[0][0] == output for ex,output in zip(task.examples,outputs)]):
                #print(f"rejecting {sampled} bc it's the identity function")
                continue # rejection sample
        

        new_examples = [(ex[0],output) for ex,output in zip(task.examples,outputs)]
        new_task = Task(task.name, task.request, new_examples)


        res.append((sampled,new_task))
        #mlb.green(f"accepting {sampled}")
        print(f"accepting {sampled}")
        for ex in new_task.examples:
            print(f"\t{ex[0][0]} -> {ex[1]}")
        if len(res) >= n:
            break
    # g.sampleFromSketch(arrow(list, list), sk)

    return res

class DeepcoderTaskloader:
    def __init__(self,file,allowed_requests,N=5,L=10,V=63,buf_size=500,repeat=False,num_tasks=None, num_mutated_tasks=10, expressive_lambdas=False, lambda_depth = 3):
        self.file = file
        self.allowed_requests = allowed_requests
        self.N = N
        self.num_mutated_tasks = num_mutated_tasks
        self.L = L
        self.V = V
        self.buf_size = buf_size
        self.expressive_lambdas = expressive_lambdas
        self.lambda_depth = lambda_depth
        self.repeat = repeat
        self.num_tasks = num_tasks

        self.buf = [] # buffer of (program,task) tuples
        with open(self.file,'r') as f:
            f.readline() # skip first line of file
            self.offset_in_file = f.tell()
            self.file_start = self.offset_in_file
        
        # make a lambda grammar to sample lambdas from
        _lambdas = get_lambdas()
        g_lambdas = Grammar.uniform(_lambdas)
        g_lambdas.max_hole_depth = self.lambda_depth
        self.g_lambdas = g_lambdas

    def reloadBuffer(self):
        assert len(self.buf) == 0
        with open(self.file,'r') as f:
            f.seek(self.offset_in_file) # pick up where we left off
            while True:
                try:
                    if len(self.buf) >= self.buf_size:
                        return # we've filled our buffer
                    # note we can't use next(f) as this disables f.tell(), so we do f.readline()
                    line = f.readline().rstrip()
                    if line == '': # readline never errors out, it returns empty string on EOF
                        if not self.repeat:
                            raise EOFError
                        f.seek(self.file_start) # repeat
                        continue
                    program,task = task_of_line(line,N=self.N,L=self.L,V=self.V)
                    if program is None:
                        continue
                    if self.allowed_requests is not None and task.request not in self.allowed_requests:
                        continue
                    if all([ex[0] == task.examples[0][0] for ex in task.examples]):
                        continue # for some reason this happens sometimes (all inputs are the same)
                    if self.expressive_lambdas:
                        programs_tasks = convert_to_deepcoder_plus_plus(program,task, g_lambdas=self.g_lambdas, n=self.num_mutated_tasks)
                        self.buf.extend(programs_tasks) # this may exceed buf_len and that's ok
                    else:
                        self.buf.append((program,task))
                    if self.num_tasks is not None and len(self.buf) % self.num_tasks == 0:
                        assert len(self.buf) != 0
                        f.seek(self.file_start)
                finally:
                    self.offset_in_file = f.tell()
    def getTask(self):
        if len(self.buf) == 0:
            self.reloadBuffer() # may raise EOFError
        return self.buf.pop()
    def getTasks(self, n, ignore_eof=False):
        ret = []
        for i in range(n):
            try:
                ret.append(self.getTask())
            except EOFError:
                if ignore_eof:
                    return ret
                raise
        return ret
            

if __name__ == '__main__':
    pass
