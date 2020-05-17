import json
import os
import subprocess
import random

from pathos.multiprocessing import Pool

from dreamcoder.domains.arithmetic.arithmeticPrimitives import k1, k0, addition, subtraction, multiplication
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import tuplify, timing, eprint, get_root_dir, mean, unfrozendict


def helmholtzEnumeration(g, request, inputs, timeout, _=None,
                         special=None, evaluationTimeout=None,
                         use_vars_in_tokenized=False, executable=None,
                         maximum_size=5000000):
    """Returns json (as text)"""
    message = {"request": request.json(),
               "timeout": timeout,
               "DSL": g.json(),
               "extras": inputs}
    if maximum_size: message["maximumSize"] = maximum_size
    if evaluationTimeout: message["evaluationTimeout"] = evaluationTimeout
    if special: message["special"] = special
    if use_vars_in_tokenized: message["use_vars_in_tokenized"] = use_vars_in_tokenized
    message = json.dumps(message)
    with open('/tmp/hm', 'w') as handle:
        handle.write(message)
    try:
        binary_name = 'helmholtz' if executable is None else executable
        binary = os.path.join(get_root_dir(), binary_name)
        process = subprocess.Popen(binary,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        response, error = process.communicate(bytes(message, encoding="utf-8"))
    except Exception as exc:
        print("ERROR: %s", exc)
        return ""
    return response

def backgroundHelmholtzEnumeration(tasks, g, timeout, _=None,
                                   special=None, evaluationTimeout=None,
                                   use_vars_in_tokenized=False, dedup=True,
                                   executable=None,
                                   serialize_special=None,
                                   maximum_size=None):
    requests = list({t.request for t in tasks})
    if serialize_special is None:
         def _s(x): return x
         serialize_x_fn = _s
    else: 
        serialize_x_fn = serialize_special
    inputs = {r:  [serialize_x_fn(unfrozendict(x)) for x in list({tuplify(xs) for t in tasks if t.request == r
                   for xs, y in t.examples})]
              for r in requests}
    workers = Pool(len(requests))
    promises = [workers.apply_async(helmholtzEnumeration,
                                    args=(g, r, inputs[r], float(timeout)),
                                    kwds={'special': special,
                                          'evaluationTimeout': evaluationTimeout,
                                          'use_vars_in_tokenized' : use_vars_in_tokenized,
                                          'executable' : executable})
                for r in requests]

    def get():
        results = [p.get() for p in promises]
        frontiers = []
        with timing("(Helmholtz enumeration) Decoded json into frontiers"):
            for request, result in zip(requests, results):
                try:
                    response = json.loads(result.decode("utf-8"))
                    for b, entry in enumerate(response):
                        frontiers.append(Frontier([FrontierEntry(program=Program.parse(p),
                                                                 logPrior=entry["ll"],
                                                                 logLikelihood=0.,
                                                                 tokens=g.escape_tokens_string(tokens).split())
                                                   for p, tokens in zip(entry["programs"], entry["tokens"])],
                                                  task=Task(str(b),
                                                            request,
                                                            [])))
                except:
                    continue
        eprint("Total number of Helmholtz frontiers:", len(frontiers))
        if maximum_size is not None and len(frontiers) > maximum_size: 
            # Take randomly.
            frontiers = random.sample(frontiers, maximum_size)
            eprint("Taking %d random Helmholtz frontiers." % len(frontiers))
        return frontiers

    return get


if __name__ == "__main__":
    g = Grammar.uniform([k1, k0, addition, subtraction, multiplication])
    frontiers = helmholtzEnumeration(g,
                                     arrow(tint, tint),
                                     [[0], [1], [2]],
                                     10.)
    eprint("average frontier size", mean(len(f.entries) for f in frontiers))
    f = DummyFeatureExtractor([])
    r = RecognitionModel(f, g, hidden=[], contextual=True)
    r.trainBiasOptimal(frontiers, frontiers, steps=70)
    g = r.grammarOfTask(frontiers[0].task).untorch()
    frontiers = helmholtzEnumeration(g,
                                     arrow(tint, tint),
                                     [[0], [1], [2]],
                                     10.)
    for f in frontiers:
        eprint(f.summarizeFull())
    eprint("average frontier size", mean(len(f.entries) for f in frontiers))
