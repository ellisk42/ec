import datetime
import json
import os
import pickle
import subprocess
import sys

from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program, Invented
from dreamcoder.utilities import eprint, timing, callCompiled, get_root_dir
from dreamcoder.vs import induceGrammar_Beta


def induceGrammar(*args, **kwargs):
    if sum(not f.empty for f in args[1]) == 0:
        eprint("No nonempty frontiers, exiting grammar induction early.")
        return args[0], args[1]
    with timing("Induced a grammar"):
        backend = kwargs.pop("backend", "pypy")
        if backend == "pypy":
            g, newFrontiers = callCompiled(pypyInduce, *args, **kwargs)
        elif backend == "rust":
            g, newFrontiers = rustInduce(*args, **kwargs)
        elif backend == "vs":
            g, newFrontiers = rustInduce(*args, vs=True, **kwargs)
        elif backend == "pypy_vs":
            kwargs.pop('iteration')
            kwargs.pop('topk_use_only_likelihood')
            fn = '/tmp/vs.pickle'
            with open(fn, 'wb') as handle:
                pickle.dump((args, kwargs), handle)
            eprint("For debugging purposes, the version space compression invocation has been saved to", fn)
            g, newFrontiers = callCompiled(induceGrammar_Beta, *args, **kwargs)
        elif backend == "ocaml":
            kwargs.pop('iteration')
            kwargs.pop('topk_use_only_likelihood')
            kwargs['topI'] = 300
            kwargs['bs'] = 1000000
            g, newFrontiers = ocamlInduce(*args, **kwargs)
        elif backend == "memorize":
            g, newFrontiers = memorizeInduce(*args, **kwargs)
        else:
            assert False, "unknown compressor"
    return g, newFrontiers

def memorizeInduce(g, frontiers, **kwargs):
    existingInventions = {p.uncurry()
                          for p in g.primitives }
    programs = {f.bestPosterior.program for f in frontiers if not f.empty}
    newInventions = programs - existingInventions
    newGrammar = Grammar.uniform([p for p in g.primitives] + \
                                 [Invented(ni) for ni in newInventions])
    
    # rewrite in terms of new primitives
    def substitute(p):
        nonlocal newInventions
        if p in newInventions: return Invented(p).uncurry()
        return p
    newFrontiers = [Frontier([FrontierEntry(program=np,
                                            logPrior=newGrammar.logLikelihood(f.task.request, np),
                                            logLikelihood=e.logLikelihood)
                           for e in f
                           for np in [substitute(e.program)] ],
                             task=f.task)
                 for f in frontiers ]
    return newGrammar, newFrontiers
    
    
        
    

def pypyInduce(*args, **kwargs):
    kwargs.pop('iteration')
    return FragmentGrammar.induceFromFrontiers(*args, **kwargs)


def ocamlInduce(g, frontiers, _=None,
                topK=1, pseudoCounts=1.0, aic=1.0,
                structurePenalty=0.001, a=0, CPUs=1,
                bs=1000000, topI=300):
    # This is a dirty hack!
    # Memory consumption increases with the number of CPUs
    # And early on we have a lot of stuff to compress
    # If this is the first iteration, only use a fraction of the available CPUs
    if all(not p.isInvented for p in g.primitives):
        if a > 3:
            CPUs = max(1, int(CPUs / 6))
        else:
            CPUs = max(1, int(CPUs / 3))
    else:
        CPUs = max(1, int(CPUs / 2))
    CPUs = 2

    # X X X FIXME X X X
    # for unknown reasons doing compression all in one go works correctly and doing it with Python and the outer loop causes problems
    iterations = 99  # maximum number of components to add at once

    while True:
        g0 = g

        originalFrontiers = frontiers
        t2f = {f.task: f for f in frontiers}
        frontiers = [f for f in frontiers if not f.empty]
        message = {"arity": a,
                   "topK": topK,
                   "pseudoCounts": float(pseudoCounts),
                   "aic": aic,
                   "bs": bs,
                   "topI": topI,
                   "structurePenalty": float(structurePenalty),
                   "CPUs": CPUs,
                   "DSL": g.json(),
                   "iterations": iterations,
                   "frontiers": [f.json()
                                 for f in frontiers]}

        message = json.dumps(message)
        if True:
            timestamp = datetime.datetime.now().isoformat()
            os.system("mkdir  -p compressionMessages")
            fn = "compressionMessages/%s" % timestamp
            with open(fn, "w") as f:
                f.write(message)
            eprint("Compression message saved to:", fn)

        try:
            # Get relative path
            compressor_file = os.path.join(get_root_dir(), 'compression')
            process = subprocess.Popen(compressor_file,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE)
            response, error = process.communicate(bytes(message, encoding="utf-8"))
            response = json.loads(response.decode("utf-8"))
        except OSError as exc:
            raise exc

        g = response["DSL"]
        g = Grammar(g["logVariable"],
                    [(l, p.infer(), p)
                     for production in g["productions"]
                     for l in [production["logProbability"]]
                     for p in [Program.parse(production["expression"])]],
                    continuationType=g0.continuationType)

        frontiers = {original.task:
                         Frontier([FrontierEntry(p,
                                                 logLikelihood=e["logLikelihood"],
                                                 logPrior=g.logLikelihood(original.task.request, p))
                                   for e in new["programs"]
                                   for p in [Program.parse(e["program"])]],
                                  task=original.task)
                     for original, new in zip(frontiers, response["frontiers"])}
        frontiers = [frontiers.get(f.task, t2f[f.task])
                     for f in originalFrontiers]
        if iterations == 1 and len(g) > len(g0):
            eprint("Grammar changed - running another round of consolidation.")
            continue
        else:
            eprint("Finished consolidation.")
            return g, frontiers


def rustInduce(g0, frontiers, _=None,
               topK=1, pseudoCounts=1.0, aic=1.0,
               structurePenalty=0.001, a=0, CPUs=1, iteration=-1,
               topk_use_only_likelihood=False,
               vs=False):
    def finite_logp(l):
        return l if l != float("-inf") else -1000

    message = {
        "strategy": {"version-spaces": {"top_i": 50}}
        if vs else
        {"fragment-grammars": {}},
        "params": {
            "structure_penalty": structurePenalty,
            "pseudocounts": int(pseudoCounts + 0.5),
            "topk": topK,
            "topk_use_only_likelihood": topk_use_only_likelihood,
            "aic": aic if aic != float("inf") else None,
            "arity": a,
        },
        "primitives": [{"name": p.name, "tp": str(t), "logp": finite_logp(l)}
                       for l, t, p in g0.productions if p.isPrimitive],
        "inventions": [{"expression": str(p.body),
                        "logp": finite_logp(l)}  # -inf=-100
                       for l, t, p in g0.productions if p.isInvented],
        "variable_logprob": finite_logp(g0.logVariable),
        "frontiers": [{
            "task_tp": str(f.task.request),
            "solutions": [{
                "expression": str(e.program),
                "logprior": finite_logp(e.logPrior),
                "loglikelihood": e.logLikelihood,
            } for e in f],
        } for f in frontiers],
    }

    eprint("running rust compressor")

    messageJson = json.dumps(message)

    with open("jsonDebug", "w") as f:
        f.write(messageJson)

    # check which version of python we are using
    # if >=3.6 do:
    if sys.version_info[1] >= 6:
        p = subprocess.Popen(
            ['./rust_compressor/rust_compressor'],
            encoding='utf-8',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
    elif sys.version_info[1] == 5:
        p = subprocess.Popen(
            ['./rust_compressor/rust_compressor'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

        messageJson = bytearray(messageJson, encoding='utf-8')
        # convert messageJson string to bytes
    else:
        eprint("must be python 3.5 or 3.6")
        assert False

    p.stdin.write(messageJson)
    p.stdin.flush()
    p.stdin.close()

    if p.returncode is not None:
        raise ValueError("rust compressor failed")

    if sys.version_info[1] >= 6:
        resp = json.load(p.stdout)
    elif sys.version_info[1] == 5:
        import codecs
        resp = json.load(codecs.getreader('utf-8')(p.stdout))

    productions = [(x["logp"], p) for p, x in
                   zip((p for (_, _, p) in g0.productions if p.isPrimitive), resp["primitives"])] + \
                  [(i["logp"], Invented(Program.parse(i["expression"])))
                   for i in resp["inventions"]]
    productions = [(l if l is not None else float("-inf"), p)
                   for l, p in productions]
    g = Grammar.fromProductions(productions, resp["variable_logprob"], continuationType=g0.continuationType)
    newFrontiers = [
        Frontier(
            [
                FrontierEntry(
                    Program.parse(
                        s["expression"]),
                    logPrior=s["logprior"],
                    logLikelihood=s["loglikelihood"]) for s in r["solutions"]],
            f.task) for f,
                        r in zip(
            frontiers,
            resp["frontiers"])]
    return g, newFrontiers
