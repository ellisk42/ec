from dreamcoder.dreamcoder import *
from dreamcoder.domains.sketch.sketchPrimitives import *
from dreamcoder.domains.sketch.sketchPrimitives import _empty_sketch
from dreamcoder.utilities import *
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.utilities import *


class SupervisedSketch(Task):
    def __init__(self, name, program):
        super(SupervisedSketch, self).__init__(name, arrow(tsketch,tsketch), []) # TODO: LT, needs this, i.e., a request. 
        if isinstance(program,str):
            program = par
            try:
                program = parseSketch(program)
            except:
                eprint("Parse failure:")
                eprint(program)
                assert False
        self.original = program
        self.hand, self.trace = executeSketch(program)
        # hand, self.trace = program.evaluate([])(_empty_sketch)(SketchState())
        # self.hand = state.hand
        # self.hand = hand
        self.specialTask = ("sketch",
                            {"trace": self.trace})
        self.image = None
        self.handImage = None

    # def getImage(self, drawHand=False, pretty=False):
    #     if not drawHand:
    #         if not pretty:
    #             if self.image is not None: return self.image
    #             self.image = renderPlan(self.plan, pretty=pretty)
    #             return self.image
    #         else:
    #             return renderPlan(self.plan, pretty=True)
    #     else:
    #         if self.handImage is not None: return self.handImage
    #         self.handImage = renderPlan(self.plan,
    #                                     drawHand=self.hand,
    #                                     pretty=pretty)
    #         return self.handImage
                

    
    # # do not pickle the image
    # def __getstate__(self):
    #     return self.specialTask, self.plan, self.request, self.cache, self.name, self.examples
    # def __setstate__(self, state):
    #     self.specialTask, self.plan, self.request, self.cache, self.name, self.examples = state
    #     self.image = None


    # def animate(self):
    #     from pylab import imshow,show
    #     a = renderPlan(self.plan)
    #     imshow(a)
    #     show()

    # @staticmethod
    # def showMany(ts):
    #     from pylab import imshow,show
    #     a = montage([renderPlan(t.plan, pretty=True, Lego=True, resolution=256,
    #                             drawHand=False)
    #                  for t in ts]) 
    #     imshow(a)
    #     show()

    # @staticmethod
    # def exportMany(f, ts, shuffle=True, columns=None):
    #     import numpy as np
        
    #     ts = list(ts)
    #     if shuffle:
    #         assert all( t is not None for t in ts  )
    #         random.shuffle(ts)
    #     a = montage([renderPlan(t.plan, pretty=True, Lego=True, resolution=256) if t is not None \
    #                  else np.zeros((256,256,3))
    #                  for t in ts],
    #                 columns=columns)        
    #     import scipy.misc
    #     scipy.misc.imsave(f, a)
        

    # def exportImage(self, f, pretty=True, Lego=True, drawHand=False):
    #     a = renderPlan(self.plan,
    #                    pretty=pretty, Lego=Lego,
    #                    drawHand=t.hand if drawHand else None)
    #     import scipy.misc
    #     scipy.misc.imsave(f, a)

    def logLikelihood(self, e, timeout=None): # TODO, LT, given expression, calculates distance.
        def simpleTrace(prog):
            """ gets trace and simplifies"""
            trace = p.evaluate([])(T._empty_sketch)(SketchState(hand=starthand, history=[]))[1]
            # any LL convert to multiple 
            # TODO: didnt do anything. should convert LL to little lines. Before do that should figure out whether to 
            # allow rotation of horizontal to vert line...

        def loss(prog1, prog2):
            """compares prog1 and prog2"""
            trace1 = prog1.evaluate([])(T._empty_sketch)(SketchState(hand=HANDSTARTPOS, history=[]))[1]
            trace2 = prog2.evaluate([])(T._empty_sketch)(SketchState(hand=HANDSTARTPOS, history=[]))[1]
            
            if set(trace1)==set(trace2):
                return 0.0
            else:
                return NEGATIVEINFINITY

        trace = e.evaluate([])(_empty_sketch)(SketchState(hand=HANDSTARTPOS, history=[]))[1]

        if set(self.trace)==set(trace):
            return 0.0
        else:
            return NEGATIVEINFINITY


def makeSupervisedTasks(trainset="S8full", doshaping=False): # TODO, LT, make these tasks.

    print("DRAW TASK training set: {}".format(trainset))

    programs = []
    programnames = []

    # ===== train on basic stimuli like lines
    if doshaping:
        print("INCLUDING SHAPING STIMULI")
        ll = transform(_line, theta=pi/2, s=4, y=-2.)
        programs.extend([
            _line,
            transform(_circle, s=2.),
            transform(_circle, theta=pi/2),
            transform(_line, theta=pi/2),
            transform(_line, s=4),
            transform(_line, y=-2.),
            transform(_line, theta=pi/2, s=4.),
            transform(_line, theta=pi/2, y=-2.),
            transform(_line, theta=pi/2, s=4, y=-2.)]
            )
        programnames.extend(["shaping_{}".format(n) for n in range(9)])

    ##############################################
    ################# TRAINING SETS
    if trainset=="S8_nojitter":
        libname = "dreamcoder/domains/draw/trainprogs/S8_nojitter_shaping"
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs.extend(P)
        programnames.extend(["S8_nojitter_shaping_{}".format(n) for n in range(len(P))])

        libname = "dreamcoder/domains/draw/trainprogs/S8_nojitter"
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs.extend(P)
        programnames.extend(["S8_nojitter_{}".format(n) for n in range(len(P))])


    def addPrograms(lib, programs, programnames, nameprefix=[]):
        if not nameprefix:
            nameprefix=lib
            # note: assumes that name prefix is lib. here tell it otherwise.

        # ========= 1) SHAPING:
        # ---- get programs
        libname = "dreamcoder/domains/draw/trainprogs/{}".format(lib)
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs.extend(P)

        # ---- get program names
        with open("{}_stimnum.pkl".format(libname), 'rb') as fp:
            stimnum = pickle.load(fp)
        names = ["{}_{}".format(nameprefix, s) for s in stimnum]
        programnames.extend(names)

        return programs, programnames

    if trainset in ["S12", "S13"]:

        programs, programnames = addPrograms("S12_13_shaping", programs, programnames)
        programs, programnames = addPrograms(trainset, programs, programnames)


    # ===== make programs
    if userealnames:
        assert len(programs) == len(programnames)
        names = programnames
    else:
        names = ["task{}".format(i) for i in range(len(programs))]
    print("training task names:")
    print(names)
    alltasks = []
    for name, p in zip(names, programs):
    # for i, p in enumerate(programs):
        # name = "task{}".format(i)
        alltasks.append(SupervisedDraw(name, p))



    ##############################################
    ################# make test tasks?
    programs_test = []
    programs_test_names = []
    testtasks = []
    if trainset in ["S8full", "S8", "S9full", "S9", "S8_nojitter", "S9_nojitter"]:
        
        libname = "dreamcoder/domains/draw/trainprogs/S8_test"
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs_test.extend(P) 
        programs_test_names.extend(["S8_{}".format(n) for n in [0, 2, 59, 65, 94]])

        libname = "dreamcoder/domains/draw/trainprogs/S9_test"
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs_test.extend(P) 
        programs_test_names.extend(["S9_{}".format(n) for n in [14, 15, 17, 18, 29, 43, 55, 59, 61, 86, 96, 99, 140]])

        libname = "dreamcoder/domains/draw/trainprogs/S8_nojitter_test"
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs_test.extend(P) 
        programs_test_names.extend(["S8_nojitter_{}".format(n) for n in [69, 73, 134, 137, 139]])

        libname = "dreamcoder/domains/draw/trainprogs/S9_nojitter_test"
        with open("{}.pkl".format(libname), 'rb') as fp:
            P = pickle.load(fp)
        programs_test.extend(P) 
        programs_test_names.extend(["S9_nojitter_{}".format(n) for n in [56, 59, 76, 80, 108, 112, 135, 139, 144, 147]])

    if programs_test:
        if userealnames:
            assert len(programs_test) == len(programs_test_names)
            names = programs_test_names
        else:
            names = ["test{}".format(i) for i in range(len(programs_test))]

        for name, p in zip(names, programs_test):
        # for i, p in enumerate(programs_test):
            # name = "test{}".format(i)
            testtasks.append(SupervisedDraw(name, p))
    print("test tasks:")
    print(names)
    
    return alltasks, testtasks, programnames, programs_test_names

