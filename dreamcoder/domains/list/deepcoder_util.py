#deepcoder_util.py
import sys
import os
#sys.path.append(os.path.abspath('./'))

from builtins import super
import pickle
import string
import argparse
import random

import torch
from torch import nn, optim

#from pinn import RobustFill
#from pinn import SyntaxCheckingRobustFill
import random
import math
import time

from collections import OrderedDict
#from util import enumerate_reg, Hole

from dreamcoder.grammar import Grammar, NoCandidates
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderProductions, flatten_program

from dreamcoder.program import Application, Hole, Primitive, Index, Abstraction, ParseFailure

import math
from dreamcoder.type import Context, arrow, tint, tlist, tbool, UnificationFailure

from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives
dc_primitives = {prim.name:prim for prim in deepcoderPrimitives()}

productions = deepcoderProductions()  # TODO - figure out good production probs ... 
basegrammar = Grammar.fromProductions(productions, logVariable=0.0)  # TODO

def deepcoder_vocab(grammar, n_inputs=2): 
    return [prim.name for prim in grammar.primitives] + ['input_' + str(i) for i in range(n_inputs)] + ['<HOLE>']  # TODO

def tokenize_for_robustfill(IOs):
    """
    tokenizes a batch of IOs
    """
    newIOs = []
    for examples in IOs:
        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            serializedInputs = []
            for x in xs:
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                serializedInputs.extend(x)
            tokenized.append((serializedInputs, y))
        newIOs.append(tokenized)
    return newIOs

def buildCandidate(request, context, environment, parsecontext, index_dict={}):
    """Primitives that are candidates for being used given a requested type
    If returnTable is false (default):
    returns [((log)likelihood, tp, primitive, context)]
    if returntable is true: returns {primitive: ((log)likelihood, tp, context)}"""
    variable_list = ['input_' + str(i) for i in range(4)]

    if len(parsecontext) == 0: raise NoCandidates()
    chosen_str = parsecontext[0]
    parsecontext = parsecontext[1:] #is this right?

    candidate = None

    #for l, t, p in self.productions:
    #print(sys.path)
    #print("PRIMITIVE GLOBALS:", Primitive.GLOBALS)
    if chosen_str in dc_primitives: #if it is a primtive
        p = dc_primitives[chosen_str]
        t = p.tp 
        try:
            newContext, t = t.instantiate(context)
            newContext = newContext.unify(t.returns(), request)
            t = t.apply(newContext)
            #candidates.append((l, t, p, newContext))
            candidate = (t, p, newContext)

        except UnificationFailure:
            raise ParseFailure()
    
    elif chosen_str in variable_list:
        try:
            j = index_dict[chosen_str]
        except KeyError: 
            raise ParseFailure()
        t = environment[j]
    #for j, t in enumerate(environment):
        try:
            newContext = context.unify(t.returns(), request)
            t = t.apply(newContext)
            candidate = (t, Index(j), newContext)
        except UnificationFailure:
            raise ParseFailure()
    else: #if it is a hole:
        try: assert chosen_str == '<HOLE>' #TODO, choose correct representation of program
        except AssertionError as e:
            raise
            print("bad string:", chosen_str)
            assert False
        p = Hole()
        t = request #[try all possibilities and backtrack] #p.inferType(context, environment, freeVariables) #TODO
        # or hole is request.
        try:
            newContext, t = t.instantiate(context)
            newContext = newContext.unify(t.returns(), request)
            t = t.apply(newContext)
            #candidates.append((l, t, p, newContext))
            candidate = (t, p, newContext)

        except UnificationFailure:
            raise ParseFailure()

    if candidate == None:
        raise NoCandidates()


    return parsecontext, candidate


def parseprogram(pseq, request): #TODO 
    num_inputs = len(request.functionArguments())

    index_dict = {'input_' + str(i): num_inputs-i-1 for i in range(num_inputs)}

    #request = something #TODO

    def _parse(request, parsecontext, context, environment):
        if request.isArrow():
            parsecontext, context, expression = _parse(
                request.arguments[1], parsecontext, context, [
                    request.arguments[0]] + environment)
            return parsecontext, context, Abstraction(expression) #TODO

        parsecontext, candidate = buildCandidate(request, context, environment, parsecontext, index_dict=index_dict)

        newType, chosenPrimitive, context = candidate
   
        # Sample the arguments
        xs = newType.functionArguments()
        returnValue = chosenPrimitive

        for x in xs:
            x = x.apply(context)
            parsecontext, context, x = _parse(
                x, parsecontext, context, environment)
            returnValue = Application(returnValue, x)

        return parsecontext, context, returnValue
        
    _, _, e = _parse(
                    request, pseq, Context.EMPTY, [])
    return e

def make_holey_deepcoder(prog,
                            k,
                            g,
                            request,
                            inv_temp=1.0,
                            reward_fn=None,
                            sample_fn=None,
                            verbose=False,
                            use_timeout=False):
    #need to add improved_dc_model=False, nHoles=1
    """
    inv_temp==1 => use true mdls
    inv_temp==0 => sample uniformly
    0 < inv_temp < 1 ==> something in between
    """ 
    choices = g.enumerateHoles(request, prog, k=k)

    if len(list(choices)) == 0:
        #if there are none, then use the original program 
        choices = [(prog, 0)]
    #print("prog:", prog, "choices", list(choices))
    progs, weights = zip(*choices)

    # if verbose:
    #     for c in choices: print(c)

    if sample_fn is None:
        sample_fn = lambda x: inv_temp*math.exp(inv_temp*x)

    if use_timeout:
        # sample timeout
        r = random.random()
        t = -math.log(r)/inv_temp

        cs = list(zip(progs, [-w for w in weights]))
        if t < list(cs)[0][1]: return prog, None, None

        below_cutoff_choices = [(p, w) for p,w in cs if t > w]


        _, max_w = max(below_cutoff_choices, key=lambda item: item[1])

        options = [(p, None, None) for p, w in below_cutoff_choices if w==max_w]
        x = random.choices(options, k=1)
        return x[0]

        # cdf = lambda x: 1 - math.exp(-inv_temp*(x))
        # weights = [-w for w in weights]
        # probs = list(weights)
        # #probs[0] = cdf(weights[0])
        # for i in range(0, len(weights)-1):
        #     probs[i] = cdf(weights[i+1]) - cdf(weights[i])

        # probs[-1] = 1 - cdf(weights[-1])
        # weights = tuple(probs)
    else:
    #normalize weights, and then rezip
    
        weights = [sample_fn(w) for w in weights]
        #normalize_weights
        w_sum = sum(w for w in weights)
        weights = [w/w_sum for w in weights]
    

    if reward_fn is None:
        reward_fn = math.exp
    rewards = [reward_fn(w) for w in weights]

    prog_reward_probs = list(zip(progs, rewards, weights))

    if verbose:
        for p, r, prob in prog_reward_probs:
            print(p, prob)

    if k > 1:
        x = random.choices(prog_reward_probs, weights=weights, k=1)
        return x[0] #outputs prog, prob
    else:
        return prog_reward_probs[0] #outputs prog, prob




# ####unused####
# def sample_request(): #TODO
#     requests = [
#             arrow(tlist(tint), tlist(tint)),
#             arrow(tlist(tint), tint),
#             #arrow(tint, tlist(tint)),
#             arrow(tint, tint)
#             ]
#     return random.choices(requests, weights=[4,3,1])[0] #TODO

# def isListFunction(tp):
#     try:
#         Context().unify(tp, arrow(tlist(tint), tint)) #TODO, idk if this will work
#         return True
#     except UnificationFailure:
#         try:
#             Context().unify(tp, arrow(tlist(tint), tlist(tint))) #TODO, idk if this will work
#             return True
#         except UnificationFailure:
#             return False


# def isIntFunction(tp):
#     try:
#         Context().unify(tp, arrow(tint, tint)) #TODO, idk if this will work
#         return True
#     except UnificationFailure:
#         try:
#             Context().unify(tp, arrow(tint, tlist(tint))) #TODO, idk if this will work
#             return True
#         except UnificationFailure:
#             return False

# def sampleIO(program, tp, k_shot=4, verbose=False): #TODO
#     #needs to have constraint stuff
#     N_EXAMPLES = 5
#     RANGE = 30 #TODO
#     LIST_LEN_RANGE = 8
#     OUTPUT_RANGE = 128

#     #stolen from Luke. Should be abstracted in a class of some sort.
#     def _featuresOfProgram(program, tp, k_shot=4):
#         e = program.evaluate([])
#         examples = []
#         if isListFunction(tp):
#             sample = lambda: random.sample(range(-RANGE, RANGE), random.randint(0, LIST_LEN_RANGE))
#         elif isIntFunction(tp):
#             sample = lambda: random.randint(-RANGE, RANGE-1)
#         else:
#             return None
#         for _ in range(N_EXAMPLES*3):
#             x = sample()
#             #try:
#             #print("program", program, "e", e, "x", x)
#             y = e(x)
#             #eprint(tp, program, x, y)

#             if x == [] or y == []: 
#                 if verbose: print("tripped empty list continue ")
#                 continue   

#             if type(y) == int:
#                 y = [y] #TODO fix this dumb hack ...    
#             if type(x) == int:
#                 x = [x] #TODO fix this dumb hack ...

#             if any((num >= OUTPUT_RANGE) or (num < -OUTPUT_RANGE) for num in y): #this is a total hack
#                 if verbose: print("tripped range continue", flush=True)
#                 continue

#             examples.append( (x, y) )


#             # except:
#             #     print("tripped continue 2", flush=True)
#             #     continue

#             if len(examples) >= k_shot: break
#         else:
#             return None #What do I do if I get a None?? Try another program ...
#         return examples

#     return _featuresOfProgram(program, tp, k_shot=k_shot)

# def getInstance(k_shot=4, max_length=30, verbose=False, with_holes=False, k=None):
#     """
#     Returns a single problem instance, as input/target strings
#     """
#     #TODO
#     assert False, "this function has been depricated"
#     while True:
#         #request = arrow(tlist(tint), tint, tint)
#         #print("starting getIntance loop")
#         request = sample_request()
#         #print("request", request)
#         p = grammar.sample(request, maximumDepth=4) #grammar not abstracted well in this script
#         #print("program:", p)
        
#         IO = sampleIO(p, request, k_shot, verbose=verbose)
#         if IO == None: #TODO, this is a hack!!!
#             if verbose: print("tripped IO==None continue")
#             continue
#         if any(y==None for x,y in IO):
#             if verbose: print("tripped y==None continue")
#             assert False
#             continue

#         pseq = flatten_program(p)
        

#         if all(len(x)<max_length and len(y)<max_length for x, y in IO): 
#             d = {'IO':IO, 'pseq':pseq, 'p':p, 'tp': request}
#             if with_holes:
#                 d['sketch'] = make_holey_deepcoder(p, k, grammar, request)
#                 d['sketchseq'] = flatten_program(d['sketch'])
#             break
#         if verbose: print("retry sampling program")
#     return d

# def getBatch():
#     """
#     Create a batch of problem instances, as tensors
#     """
#     k_shot = random.choice(range(3,6)) #this means from 3 to 5 examples

#     instances = [getInstance(k_shot=k_shot, max_length=max_length) for i in range(batch_size)]
#     IO = [inst['IO'] for inst in instances]
#     p = [inst['p'] for inst in instances]
#     pseq = [inst['pseq'] for inst in instances]
#     tp = [inst['tp'] for inst in instances]
#     return IO, pseq, p, tp
