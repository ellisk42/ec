# from dreamcoder.domains.draw.makeDrawTasks import drawDrawings
from dreamcoder.domains.draw.drawPrimitives import *
from dreamcoder.domains.draw.drawPrimitives import primitives
from dreamcoder.domains.draw.primitives import _repeat, _line, _makeAffine, _circle,_connect
from dreamcoder.domains.draw.makeDrawTasks import makeSupervisedTasks, SupervisedDraw
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
# from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.recognition import ImageFeatureExtractor
# from dreamcoder.utilities import eprint, testTrainSplit, loadPickle
import datetime
import os


class DrawCNN(ImageFeatureExtractor):
    special = "draw"
    def __init__(self, tasks, testingTasks=[], cuda=False):
        super(DrawCNN, self).__init__(inputImageDimension=128,
                                            resizedDimension=64,
                                            cuda=cuda,
                                            channels=1)
        print("output dimensionality",self.outputDimensionality)
    def taskOfProgram(self, p, t):
        if t.isArrow:
            # continuation passing
            i = p.evaluate([])([])
        else:
            i = p.evaluate([])
        return SupervisedDraw("dream", i)

    def featuresOfTask(self, t):
        return self(t.rendered_strokes)

g0 = Grammar.uniform(primitives, continuationType=tstroke)

def dreamFromGrammar(g=g0, directory = "", N=25):
   # request = taxes # arrow9turtle turtle) just for logl.
   # request = arrow(taxes, taxes) # arrow9turtle turtle) just for logl.
   request = tstroke # arrow9turtle turtle) just for logl.
   request = arrow(tstroke, tstroke)
   programs = [ p for _ in range(N) for p in [g.sample(request, maximumDepth=15)] if p is not None]
   return programs
   # drawDrawings(*programs, filenames)

def main_dummy(N=25):
    ps = dreamFromGrammar(N=N)
    for n,p in enumerate(ps):
        print(n,p)
        a = p.evaluate([])
        savefig(a, f"/tmp/draw{n}.png")
        # a.savefig(f"/tmp/draw{n}.png")
                                    
                       
from dreamcoder.domains.draw.drawPrimitives import *
import dreamcoder.domains.draw.primitives as PP
from dreamcoder.type import arrow
from itertools import product


def visualizePrimitives(primitives, export="/tmp/draw_primitives"):
    from math import ceil
    matrix = []
    print("ONLY PLOTS INVENTED PRIMITIVES")
    print("FILLS ARGUMENT HOLES WITH A HANDFUL OF VALUES")
    stringlist = [] # to collect primitives
    for i, p in enumerate(primitives):
        # print("--- prim {}".format(i))
        stringlist.append("--- prim {}".format(i))
        stringlist.append(str(p))
        if not p.isInvented: continue
        t = p.tp
        # print(p,":",p.tp)
        if t.returns() != tstroke:
            # print("\t(does not return a tstroke)")
            stringlist.append("\t(does not return a tstroke)")
            continue

        def argumentChoices(t):
            if t in [tmaybe(tangle), tangle]:
                return [j*(2*pi/4) for j in range(4)]
            elif t in [tmaybe(tstroke), tstroke]:
                return [PP._line, PP._circle, PP.polygon(3)]
            elif t in [tmaybe(tscale), tscale]:
                return [2., 4.]
            elif t in [tmaybe(tdist), tdist]:
                return [-2., -1., 0, 1., 2.]
            elif t in [ttrorder, tmaybe(ttrorder)]:
                return PP.ORDERS
            elif t in [tmaybe(trep), trep]:
                return [j+1 for i, j in enumerate(range(7))]
            elif t == arrow(tmaybe(ttrorder), ttransmat):
                return [[PP.ORDERS[0], _makeAffine()]]
            else: return []

        ts = [] # holds all cases for this primitive
        # print(t.functionArguments())
        # print([argumentChoices(t) for t in t.functionArguments() ])
        stringlist.append(str(t.functionArguments()))
        stringlist.extend([str(argumentChoices(t)) for t in t.functionArguments() ])

        for arguments in product(*[argumentChoices(t) for t in t.functionArguments() ]):
            t = p.evaluate([])
            for a in arguments: t = t(a)
            ts.append(t)
            
        matrix.append(ts)
    

    # ==== make and save plot
    def save(ts, j):
        n = len(ts)
        ncol = 6
        nrow = ceil(n+1/6)
        fig = plt.figure(figsize=(ncol*2, nrow*2))
        for ii, nn in enumerate(ts):
            ax = plt.subplot(nrow, ncol, ii+1)
            PP.plotOnAxes(nn, ax)
            plt.title("prim {}".format(j))
        fig.savefig("{}_p{}.pdf".format(export, j))
        
    for j, ts in enumerate(matrix):
        save(ts, j)
        
#     # Only visualize if it has something to visualize.
#     if len(matrix) > 0:
#         matrix = montageMatrix(matrix)
#         # imshow(matrix)
        
#         import scipy.misc
#         scipy.misc.imsave(fn, matrix)
#         #    show()
#     else:
#         eprint("Tried to visualize primitives, but none to visualize.")

    for s in stringlist:
        print(s)
    return matrix, stringlist



def main(arguments):
        if arguments["dopruning"]:
            print("PRUNING PRIMITIES, using trainset {}".format(arguments["trainset"]))
        else:
            print("NOT DOING PRUNING")
        primitives = getPrimitives(trainset=arguments["trainset"], prune=arguments["dopruning"])
        g0 = Grammar.uniform(primitives)

        print("As an example, here's the amount of ink used by a line, a circle, and  connected to a circle:")
        print(program_ink(_line))
        print(program_ink(_circle))
        print(program_ink(_circle + _line))
        print()

        if False:
            #p = Program.parse("(repeat line 4 ")
            p = Program.parse("(transform circle #(transmat (Some scale4) None None None None))")
            p.evaluate([])
            print(p.infer())
            #p = _repeat(_connect(_line,_circle), 3, _makeAffine(x=1.0))
            Parse.animate_all(Parse.ofProgram(p), "/tmp/parses.png")
            for i in range(50):
                                            p = g0.sample(tstroke, maximumDepth=10)
                                            if p is None: continue
                                            Parse.animate_all(Parse.ofProgram(p), f"/tmp/parses{i}.png")
            print("Primitives:")
            print(primitives)
        
        train, test = makeSupervisedTasks(trainset=arguments["trainset"], doshaping=arguments["doshaping"])[:2]

        # For the testing tasks we are going to use Euclidean distance as a likelihood function
        # The likelihood is -5*(Euclidean distance over all pixels)
        # Change the number five below to make it something else.
        for t in test:
            t.specialTask[1]["l2"] = 5.

        # ==== remove bad shaping tasks
        train = [t for t in train if t.name not in ["shaping_4", "shaping_6"]]
        print("------")
        print("removed shaping_4 and shaping_6 from shaping tasks (goes outside canvas, ocaml code doesnt allow")
        print("new tasks:")
        print([t.name for t in train])

        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "experimentOutputs/draw/%s"%timestamp
        evaluationTimeout = 0.001 # seconds, how long allowed

        os.system(f"mkdir -p {outputDirectory}")
        arguments["featureExtractor"] = DrawCNN
        if arguments["skiptesting"]==False and len(test)>0:
                generator = ecIterator(g0, train, testingTasks=test,
                        outputPrefix="%s/draw"%outputDirectory,
                        evaluationTimeout=evaluationTimeout,
                        **arguments) # 
        else:
                print("NO TESTING TASKS INCLUDED")
                generator = ecIterator(g0, train,
                        outputPrefix="%s/draw"%outputDirectory,
                        evaluationTimeout=evaluationTimeout,
                        **arguments) # 

        for result in generator:
                continue
