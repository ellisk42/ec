import sys
import mlb
import numpy as np
import os
from hydra import utils
#import mlb
import contextlib
import multiprocessing as mp
import queue
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


def task_of_line(line, N=5, L=10, V=63, num_tasks=1):
    line = line.replace(' | ', '\n')
    dc_program = compile(line, V=V, L=L)

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
            raise InvalidSketchError
    if isinstance(res,(list,tuple)):
        if len(res) == 0:
            return
        if isinstance(res[0],(list,tuple)):
            return [check_in_range(x,V) for x in res]
        maxx = max(res)
        minn = min(res)
        if maxx > V or minn < -V:
            #mlb.yellow(f'rejecting sketch bc concrete evaluation out of range: list min={minn}; list max={maxx}')
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
    
def evaluate_ctxs(e, ctxs, V):
    """
    evaluate multiple contexts (e.g. all examples)
    """
    try:
        res = [evaluate(e,ctx, V) for ctx in ctxs]
    except (ZeroDivisionError,FloatingPointError):
        raise InvalidSketchError
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


class FakeFrontier:
    # pretends to be whatever valueLossFromFrontier wants for simplicity
    def __init__(self,program,task):
        self.task = task # satisfies frontier.task call
        self._fullProg = program
        self.program = self # trick for frontier.sample().program._fullProg

        # for my own use
        self.p = program
        self.t = task
    def sample(self):
        return self

class DeepcoderTaskloader:
    def __init__(self, cfg, mode):
        if mode == 'train':
            self.cfg = cfg.data.train
        elif mode == 'test':
            self.cfg = cfg.data.test
        self.mode = mode
        self.parent_cfg = cfg # in case it's useful
        cfg = self.cfg # bc we're more likely to use it in the rest of __init__

        # if cfg.repeat is False:
        #     if cfg.expressive_lambdas:
        #         cfg.buf_len = max([cfg.buf_len,cfg.num_templates*cfg.num_mutated_tasks])
        #     else:
        #         cfg.buf_len = max([cfg.buf_len,cfg.num_templates])

        self.threaded = cfg.threaded
        assert not self.threaded, "We're not allowing threading for now"
        
        self.file = utils.to_absolute_path(f'dreamcoder/domains/list/DeepCoder_data/T{cfg.T}_A2_V512_L10_{mode}_perm.txt')
        self.allowed_requests = None if self.cfg.allow_complex_requests else [arrow(tlist(tint),tlist(tint))]

        #self.buf = queue.Queue(self.cfg.buf_size) # turns out Queues also cant be pickled:(
        self.buf = []
        #self.lock = None
        #self.exception = 0
        self.templates_seen = 0 # number of tasks seen (pre mutation)
        #self.p = None

        with open(self.file,'r') as f:
            f.readline() # skip first line of file
            self.offset_in_file = f.tell()
            self.file_start = self.offset_in_file
        
        # make a lambda grammar to sample lambdas from
        _lambdas = get_lambdas()
        g_lambdas = Grammar.uniform(_lambdas)
        g_lambdas.max_hole_depth = self.cfg.lambda_depth
        self.g_lambdas = g_lambdas
        #self.post_load()
    # @contextlib.contextmanager
    # def saveable(self):
    #     if self.lock is not None:
    #         print("locking...")
    #         self.lock.acquire() # so the worker doesnt get confused
    #         print("acquired lock")
    #     l,q,v,e,p = self.lock, self.buf, self.offset_in_file, self.exception, self.p
    #     self.lock, self.buf, self.offset_in_file, self.exception,p = None, None, v.value, e.value, None
    #     try:
    #         yield None
    #     finally:
    #         self.lock,self.buf,self.offset_in_file,self.exception, self.p = l,q,v,e, p
    #         if self.lock is not None:
    #             self.lock.release()
    #             print("lock released")
    # def check(self):
    #     if self.exception.value == 1:
    #         raise Exception("Worker thread died")
    # def post_load(self):
    #     if self.lock is not None:
    #         return # ignore redundant loads
    #     self.offset_in_file = mp.Value('i',self.offset_in_file)
    #     self.exception = mp.Value('i',self.exception)
    #     self.lock = mp.Lock()
    #     self.buf = mp.Queue(self.cfg.buf_size) # buffer of frontiers
    #     if self.cfg.threaded:
    #         self.launch_worker()
    #     self.check()

    def reloadBuffer(self):
        """
        Refills the buffer.
            If `self.cfg.repeat=True` -> will loop back to start of file at EOF (and will keep filling buffer). Otherwise EOFError will be raised.
            `self.cfg.num_mutated_tasks` gives the number of mutations of each template that will be given in a row.
        How much the buffer will be filled:
            It'll be filled up to `self.cfg.buf_size`
        """
        with open(self.file,'r') as f:
            #cm = lock if lock is not None else contextlib.nullcontext()
            #with cm:
            f.seek(self.offset_in_file) # pick up where we left off
            while True: # loops until `queue.Full` gets raised by the .put() line (or buf.full() happens)
                try:
                    # if lock is not None:
                    #     lock.acquire()
                    #if self.buf.full():
                    #    break
                    if len(self.buf) >= self.cfg.buf_size:
                        assert len(self.buf) == self.cfg.buf_size, "bug in code"
                        return
                    # note we can't use next(f) as this disables f.tell(), so we do f.readline()
                    line = f.readline().rstrip()

                    # EOF
                    if line == '': # readline never errors out, it returns empty string on EOF
                        if self.cfg.repeat:
                            f.seek(self.file_start) # repeat
                            continue
                        else:
                            raise EOFError

                    # get program and task
                    program,tasks = task_of_line(line,N=self.cfg.N,L=self.cfg.L,V=self.cfg.V, num_tasks=self.cfg.num_mutated_tasks)
                    if program is None:
                        continue
                    ff = FakeFrontier(program,tasks[0])
                    task = tasks[0]

                    # filter out bad programs/tasks
                    if self.allowed_requests is not None and task.request not in self.allowed_requests:
                        continue
                    if all([ex[0] == task.examples[0][0] for ex in task.examples]):
                        continue # for some reason this happens sometimes (all inputs are the same)
                    try:
                        check_in_range(task.examples,self.cfg.V)
                    except InvalidSketchError:
                        continue

                    self.templates_seen += 1 # number of templates seen

                    # deepcoder++ conversion
                    if self.cfg.expressive_lambdas:
                        frontiers = self.convert_to_deepcoder_plus_plus(ff,tasks)
                    else:
                        frontiers = [ff]

                    # add to buffer and potentially exit by throwing exception (this is the only exit point)
                    for frontier in frontiers:
                        self.buf.append(frontier) # may raise queue.Full exception
                        if len(self.buf) == self.cfg.buf_size:
                            return
                        #print(f"buf {self.mode}:",self.buf.qsize())
                    
                    # if we've seen the first `cfg.num_templates` programs in the file (pre-mutation)
                    # then loop back to the start of the file
                    if self.cfg.num_templates is not None and self.templates_seen == self.cfg.num_templates:
                        f.seek(self.file_start)
                finally:
                    self.offset_in_file = f.tell()
                    #assert len(self.buf) == self.buf_len
                    #time.sleep(.1) # fixes a weird BrokenPipeError from https://stackoverflow.com/questions/36359528/broken-pipe-error-with-multiprocessing-queue
                    #if lock is not None:
                    #    lock.release()

    def getTask(self):
        ret = self.getTasks(n=1)
        if len(ret) == 0:
            raise ValueError("Out of tasks, can't getTask()")
        return ret

        # #self.check()
        # #if not self.cfg.threaded and self.buf.empty():
        # if len(self.buf) == 0:
        #     self.reloadBuffer() # may raise EOFError
        #     assert len(self.buf) == self.buf_size
        # return buf.pop()
        # # while True:
        # #     try:
        # #         return self.buf.get() # may block
        # #     except queue.Empty:
        # #         print("buf is empty...")
        # #         time.sleep(.5)
        # #         self.check()

    def getTasks(self, n=None):
        """
        if n is None: reload buf and return `self.cfg.buf_size` tasks
        else: return n tasks
        Note that it may return fewer tasks if repeat=False and we hit EOF
        """
        # If n is none, return `self.cfg.buf_size` items
        if n is None:
            #cm = contextlib.nullcontext() if ignore_eof else contextlib.suppress(EOFError)
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
        
        

        # print(f"serving {n} tasks")
        # ret = self.buf[:n] # note that this will be a shallow clone of buf even if 

        # self.check()
        # if n is None:
        #     if self.cfg.threaded:
        #         n = self.buf.qsize()
        #         while n == 0:
        #             print("waiting on buffer to fill...")
        #             time.sleep(.5)
        #             self.check()
        #             n = self.buf.qsize()
        #     else:
        #         if self.buf.qsize() == 0:
        #             print(f"[main: {self.mode}] filling buffer")
        #             self.reloadBuffer()
        #             print(f"[main: {self.mode}] buffer filled")
        #         n = self.buf.qsize()
        # ret = []
        # for i in range(n):
        #     try:
        #         ret.append(self.getTask())
        #     except EOFError: # Hit end of file and self.repeat=False
        #         if ignore_eof:
        #             return ret
        #         raise
        # return ret

    # def launch_worker(self):
    #     assert False, "we're not doing threading for now"
    #     print("launching worker")
    #     p = mp.Process(target=self._worker, daemon=True)
    #     p.start()
    #     self.p = p
    #     print("launched")

    # def _worker(self):
    #     assert False, "we're not doing threading for now"
    #     def set_exc():
    #         self.exception.value = 1
    #     with mlb.debug(crash=set_exc, ctrlc=set_exc):
    #         lock = self.lock
    #         assert lock is not None
    #         while True:
    #             if not self.buf.full():
    #                 mlb.gray(f"[worker: {self.mode}] reloading buf")
    #                 self.reloadBuffer(lock=lock)
    #                 mlb.gray(f"[worker: {self.mode}] reloaded buf")
    #             #else:
    #                 #print(f"buf {self.mode} seems full, not reloading")
    #             time.sleep(1) # alternatively could just have reloadBuffer *never* exit
    def convert_to_deepcoder_plus_plus(self, f, tasks):
        program = f.p

        visitor = ToPlusPlusVisitor()
        program_plus_plus = program.visit(visitor)
        if not visitor.has_lambda:
            # if the program has no lambdas to mutate we should ignore `n`
            # and just return the unmodified program in a singleton list
            return [f]

        assert program_plus_plus.hasHoles
        res = []

        num_generated = 0
        while True:
            task = tasks[num_generated]
            sampled = self.g_lambdas.sampleFromSketch(arrow(tlist(tint), tlist(tint)), program_plus_plus, maximumDepth = 20) # this max depth wont be hit bc of Grammar.max_hole_depth
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
                outputs = evaluate_ctxs(stripped,ctxs,self.cfg.V)
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
            res.append(FakeFrontier(sampled,new_task))
            #mlb.green(f"accepting {sampled}")
            print(f"accepting {sampled}")
            for ex in new_task.examples:
                print(f"\t{ex[0][0]} -> {ex[1]}")
            num_generated += 1
            if num_generated >= self.cfg.num_mutated_tasks:
                break
        assert len(res) == self.cfg.num_mutated_tasks
        return res
            

if __name__ == '__main__':
    pass
