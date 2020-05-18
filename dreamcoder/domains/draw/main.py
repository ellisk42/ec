# from dreamcoder.domains.draw.makeDrawTasks import drawDrawings
from dreamcoder.domains.draw.drawPrimitives import *
# from dreamcoder.domains.draw.drawPrimitives import primitives
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


USENEWPRIM = False # i..e, contuinuation.
print("Note: should change this dependeing on hwether using new or old primtiives.")

class DrawCNN(ImageFeatureExtractor):
    special = "draw"
    def __init__(self, tasks, testingTasks=[], cuda=False, USE_NEW_PRIMITIVES=USENEWPRIM):
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
        return SupervisedDraw("dream", i, USE_NEW_PRIMITIVES)

    def featuresOfTask(self, t):
        return self(t.rendered_strokes)


def dreamFromGrammar(g=None, directory = "", N=25, USE_NEW_PRIMITIVES=True, 
    maximumDepth=15, returnLogLikelihoods=False):
   # request = taxes # arrow9turtle turtle) just for logl.
   # request = arrow(taxes, taxes) # arrow9turtle turtle) just for logl.if USE_NEW_PRIMITIVES:

    if g is None:
        primitives = primitiveList(USE_NEW_PRIMITIVES = USE_NEW_PRIMITIVES)
        if USE_NEW_PRIMITIVES:
            g = Grammar.uniform(primitives, continuationType=tstroke)
        else:
            g = Grammar.uniform(primitives)
    if USE_NEW_PRIMITIVES:
        request = arrow(tstroke, tstroke)
    else:
        request = tstroke # arrow9turtle turtle) just for logl.
    programs = [ p for _ in range(N) for p in [g.sample(request, maximumDepth=maximumDepth)] if p is not None]
    if returnLogLikelihoods:

        ll = [g.logLikelihood(request, d) for d in programs]
        return programs, ll

    else:
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


def visualizePrimitives(primitives, export="/tmp/draw_primitives", saveon=True):
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
        
        stringlist.append(f"\n- type {t}, takes in {t.arguments} and returns {t.returns()}")
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
                return [2.]
            elif t in [tmaybe(tdist), tdist]:
                return [-1.5, 0, 1., 1.5]
            elif t in [ttrorder, tmaybe(ttrorder)]:
                return PP.ORDERS
            elif t in [tmaybe(trep), trep]:
                return [j+1 for i, j in enumerate(range(4))]
            elif t == arrow(tmaybe(ttrorder), ttransmat):
                return [[PP.ORDERS[0], _makeAffine()]]
            elif t in [ttransmat]:
                return [_makeAffine(theta=th) for th in [pi/3]] + [_makeAffine(s=s) for s in [0.5]]
            elif t == arrow(arrow(ttransmat, tstroke), tstroke):
                # return [[_makeAffine(), p1, p2] for p1, p2 in zip([PP.polygon(3), PP._line], [PP._circle, PP.polygon(3)])]               
                print("this is not correct..") 
                return [[_makeAffine(), PP._line, PP._circle]]
            else:
                print(t)
                import pdb
                pdb.set_trace()
                assert False

        ts = [] # holds all cases for this primitive
        # print(t.functionArguments())
        # print([argumentChoices(t) for t in t.functionArguments() ])
        stringlist.append(str(t.functionArguments()))
        stringlist.extend([str(argumentChoices(t)) for t in t.functionArguments() ])

        # print([argumentChoices(t) for t in t.functionArguments()])
        # print(p)
        # print(p.evaluate([]))
        # print(dir(p.evaluate([])))
        # assert False
        for arguments in product(*[argumentChoices(t) for t in t.functionArguments() ]):
            t = p.evaluate([])

            try:
                for a in arguments: 
                    t = t(a)
                # print(t)
                # assert False
                ts.append(t)
            except TypeError:
                ts.append([])
            except:
                raise
            
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
        
    if saveon:
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

        if arguments["use_cogsci_primitives"] in [1, 2]:
            USE_NEW_PRIMITIVES=False
        else:
            USE_NEW_PRIMITIVES=True

        # ========== LOAD STARTING PRIMITIVES
        if arguments["use_cogsci_primitives"]==0:
            # then ignore cogsci. use latest primtiives whatever theya re
            print('--- using teh latest primitives')
            primitives = getPrimitives(trainset=arguments["trainset"], prune=arguments["dopruning"])
        elif arguments["use_cogsci_primitives"]==1:
            # then use the exact same primtiives from cogsci submission (2020)
            print('--- using exact cogsci 2020 primtiives')
            primitives = getPrimitives(trainset = arguments["trainset"], prune=True, USE_NEW_PRIMITIVES=False, suppress_print=True) 
        elif arguments["use_cogsci_primitives"]==2:
            # then use exact same, but add 3 more primtiives which are 
            # crucial for adding on inventions (hand built).
            print('--- using exact cogsci 2020 primtiives + a few (3) extra needed for using hand built inventions.')
            primitives = getPrimitivesUpdated(arguments["trainset"])
        else:
            assert False, "not coded... make sure in [0,1,2]"


        # ========= LOAD HAND-BUILT INVENTIONS
        if not arguments["invention_set"] is None:
            print(f"== Getting hand-built inventions from set: {arguments['invention_set']}")
            Inventions = getHandcodedInventions(arguments["invention_set"])
            print("List of added inventiosn:")
            for I in Inventions:
                print(I)
            primitives.extend(Inventions)

        # ========= BUILD STARTING GRAMMAR
        print(" *** FINAL PRIMITIVES + INVENTIONS: ")
        [print(P) for P in primitives]
        if USE_NEW_PRIMITIVES:
            g0 = Grammar.uniform(primitives, continuationType=tstroke)
        else:
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
        
        train, test = makeSupervisedTasks(trainset=arguments["trainset"], doshaping=arguments["doshaping"], USE_NEW_PRIMITIVES=USE_NEW_PRIMITIVES)[:2]

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
