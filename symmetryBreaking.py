from program import *
from recognition import *
from grammar import *
from arithmeticPrimitives import *
import numpy as np

def associations(p):
    """(left associations,right associations, zeros, leaves)"""
    if p.isAbstraction: return associations(p.body)
    if p.isPrimitive or p.isIndex: return np.array([0,0,int(str(p) == "0"),1])
    f = p.f.f
    assert str(f) == "+"
    a = p.f.x
    b = p.x
    this = np.array([int(str(a.applicationParse()[0]) == "+"),
                     int(str(b.applicationParse()[0]) == "+"),
                     0,
                     1])
    return this + associations(a) + associations(b)
    

if __name__ == "__main__":
    trainingTimeout = 600
    enumerationTimeout = 10
    rt = tint
    g = Grammar.uniform([Program.parse(p)
                         for p in ["1","0","+"] ])
    tasks = [Task("dummy",rt,
                  [([],9)])]
    hf = backgroundHelmholtzEnumeration(tasks, g, 60,
                                        evaluationTimeout=0.001)
    hf = hf()
    eprint("number of Helmholtz frontiers",len(hf))
    f = DummyFeatureExtractor(tasks)
    for contextual in [True,False]:
        for trainingSource in ['enumerated','random']:
            eprint("Training contextual =",contextual,"w/",trainingSource,"data")
            r = RecognitionModel(f, g, hidden=[], contextual=contextual)
            r.train([], helmholtzFrontiers=hf if trainingSource == 'enumerated' else [],
                    helmholtzRatio=1.,
                    defaultRequest=rt,
                    timeout=trainingTimeout)
            cg = r.grammarOfTask(tasks[0]).untorch()
            eprint(cg)
            eprint("Samples from recognition model:")
            N = 100
            aggregateAssociations = np.zeros([4])
            for _ in range(N):
                s = cg.sample(rt,maximumDepth=20)
                eprint(s)
                aggregateAssociations += associations(s)
            eprint("[left associations, right associations, zeros, leaves] = ",aggregateAssociations)
            [l,r,z,total] = list(aggregateAssociations)
            eprint("%f left"%(l/(l + r)))
            eprint("%f zero"%(z/total))

                
            eprint()

    eprint("Samples from generative model:")
    aggregateAssociations = np.zeros([4])
    for _ in range(N):
        s = g.sample(rt, maximumDepth=20)
        eprint(s)
        aggregateAssociations += associations(s)
    eprint(aggregateAssociations)

