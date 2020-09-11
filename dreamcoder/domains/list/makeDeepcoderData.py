# generate deepcoder data
import sys
import os
import contextlib
# sys.path.append(os.path.abspath('./'))
# sys.path.append(os.path.abspath('./ec'))

import pickle
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
_lambdas = get_lambdas()
prims_pp = {prim.name:prim for prim in _pp}
g_pp = Grammar.uniform(_pp)
g_lambdas = Grammar.uniform(_lambdas)
#g_pp.max_hole_depth = 3
g_lambdas.max_hole_depth = 3

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

class PostprocessVisitor:
    def __init__(self):
        super().__init__()
    def primitive(self,prim):
        if prim.name in [str(num) for num in range(-10,10)]:
        return prim
    def invented(self,e):
        assert False
    def index(self,e):
        return e
    def application(self,e):
        raise NotImplementedError
        if (prim := e.f) in g_lambdas.primitives:
            if prim.name in ['+','*']:
                # these primitives can have arbitrary children
                # so we just ignore them and recurse on them
                return Application(e.f.visit(self),e.x.visit(self))
            elif prim.name in ['MIN','MAX','>','DIVISIBLE','AND','OR']:
                
                pass
            else:
                assert False, "this new primitive needs"
        return Application(e.f.visit(self),e.x.visit(self))
    def abstraction(self,e):
        return Abstraction(e.body.visit(self))


def convert_to_deepcoder_plus_plus(program,task, n, mutate=True):
    assert mutate

    visitor = ToPlusPlusVisitor()
    program_plus_plus = program.visit(visitor)
    if not visitor.has_lambda:
        # if the program has no lambdas to mutate we should ignore `n`
        # and just return the unmodified program in a singleton list
        return [(program_plus_plus,task)]

    assert program_plus_plus.hasHoles
    res = []



    for i in range(n):
        sampled = g_lambdas.sampleFromSketch(arrow(tlist(tint), tlist(tint)), program_plus_plus, maximumDepth = 20) # this max depth wont be hit bc of Grammar.max_hole_depth
        visitor = PostprocessVisitor()
        processed = sampled.visit(visitor)
        #TODO mutate the task to be right
        res.append((processed,task))
        print(processed)
    # g.sampleFromSketch(arrow(list, list), sk)

    return res

class DeepcoderTaskloader:
    def __init__(self,file,allowed_requests,N=5,L=10,V=63,repeat=False,num_tasks=None, num_mutated_tasks=10, expressive_lambdas=False):
        self.buf = [] # buffer of (program,task) tuples
        self.file = file
        self.allowed_requests = allowed_requests
        self.N = N
        self.num_mutated_tasks = num_mutated_tasks
        self.L = L
        self.V = V
        self.expressive_lambdas = expressive_lambdas
        self.repeat = repeat
        self.num_tasks = num_tasks
        self.eof = False
        with open(self.file,'r') as f:
            f.readline() # skip first line of file
            self.offset_in_file = f.tell()
            self.file_start = self.offset_in_file
    def reloadBuffer(self):
        assert len(self.buf) == 0
        with open(self.file,'r') as f:
            f.seek(self.offset_in_file) # pick up where we left off
            # note we can't use next(f) as this disables f.tell(), so we do f.readline()
            while True:
                if len(self.buf) >= 500:
                    break
                line = f.readline().rstrip()
                if line == '': # readline never errors out, it returns empty string on EOF
                    if not self.repeat:
                        self.eof = True
                        return # only paritally filled the buffer
                    f.seek(self.file_start) # repeat
                    continue
                program,task = task_of_line(line,N=self.N,L=self.L,V=self.V)
                if program is None: continue
                if self.allowed_requests is not None and task.request not in self.allowed_requests: continue
                if self.expressive_lambdas:
                    programs_tasks = convert_to_deepcoder_plus_plus(program,task, n=self.num_mutated_tasks)
                    self.buf.extend(programs_tasks)
                else:
                    self.buf.append((program,task))
                if self.num_tasks and len(self.buf) % self.num_tasks == 0:
                    assert len(self.buf) != 0
                    f.seek(self.file_start) # this should always
            self.offset_in_file = f.tell()
    def getTask(self):
        if len(self.buf) == 0:
            if self.eof:
                raise EOFError
            self.reloadBuffer()
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
