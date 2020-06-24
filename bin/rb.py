try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.rb.main import main, RBFeatureExtractor, rb_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs
import dreamcoder.ROB as ROB

if __name__ == '__main__':

    # 

    # ps = robustFillPrimitives()
    # print()
    # from dreamcoder.domains.rb.rbPrimitives import *
    # from dreamcoder.ROB import generate_FIO, executeProg
    # from dreamcoder.grammar import Grammar
    # baseGrammar = Grammar.uniform( robustFillPrimitives(), continuationType=texpression)    
    # from dreamcoder.zipper import getTracesFromProg



    # lens = []
    # buttonLens = []
    # for i in range(1000):
    #     p, I, O = generate_FIO(4)
    #     #print(p)
    #     pp = p.ecProg()
    #     pos, _ = getTracesFromProg(pp, arrow(texpression, texpression), baseGrammar, onlyPos=True, canonicalOrdering=True)
    #     lens.append (len(pos))

    #     buttonLens.append( len(p.flatten()) )


    # print("len", sum(lens)/len(lens))
    # print("button len", sum(buttonLens)/len(buttonLens))
    # assert 0

    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=RBFeatureExtractor,
        pseudoCounts=30.0,
        extras=rb_options
        )
    main(arguments)
