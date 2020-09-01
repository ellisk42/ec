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
from dreamcoder.type import Context, arrow, tint, tlist, UnificationFailure

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


class DeepcoderTaskloader:
    def __init__(self,file,allowed_requests,N=5,L=10,V=63,repeat=False,num_tasks=None):
        self.buf = [] # buffer of (program,task) tuples
        self.file = file
        self.allowed_requests = allowed_requests
        self.N = N
        self.L = L
        self.V = V
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
                if task.request not in self.allowed_requests: continue
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