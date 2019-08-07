try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import DummyFeatureExtractor, RecognitionModel
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import eprint


def main():
    trainingTimeout = 600
    rt = arrow(tint,tint)
    g = Grammar.uniform([Program.parse(p)
                         for p in ["1","0","+"] ])
    tasks = [Task("dummy",rt,
                  [(tuple([n]),9)
                   for n in range(3) ])]
    hf = backgroundHelmholtzEnumeration(tasks, g, 60,
                                        evaluationTimeout=0.001)
    hf = hf()
    f = DummyFeatureExtractor(tasks)
    for contextual in [True,False]:
        for trainingSource in ['random','enumerated']:
            eprint("Training contextual =",contextual,"w/",trainingSource,"data")
            r = RecognitionModel(f, g, hidden=[], contextual=contextual)
            r.train([], helmholtzFrontiers=hf if trainingSource == 'enumerated' else [],
                    helmholtzRatio=1.,
                    defaultRequest=rt,
                    timeout=trainingTimeout)
            cg = r.grammarOfTask(tasks[0]).untorch()
            eprint(cg)
            eprint("Samples from recognition model:")
            N = 50
            for _ in range(N):
                eprint(cg.sample(rt,maximumDepth=20))
            eprint()

    eprint("Samples from generative model:")
    for _ in range(N):
        eprint(g.sample(rt, maximumDepth=20))


if __name__ == "__main__":
    main()
