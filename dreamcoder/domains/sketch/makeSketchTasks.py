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
        self.rendered_image = renderProgram(program)
        # self.image = None
        # self.handImage = None

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

        # trace = e.evaluate([])(_empty_sketch)(SketchState(hand=HANDSTARTPOS, history=[]))[1]
        try:
            trace = executeSketch(e, timeout=0.05)[1] # identical to above.
        except RunWithTimeout:
            return NEGATIVEINFINITY
        except RecursionError:
            return NEGATIVEINFINITY
            

        if set(self.trace)==set(trace):
            return 0.0
        else:
            return NEGATIVEINFINITY


def makeSupervisedTasks(trainset=["practice"], Nset=[]):
    """give a list of sets, and will output one
    long list of tasks, in the order given in trainset.
    -Nset is list of sample sizes. if empty will default to 
    20 tasks per trainset"""
    assert isinstance(Nset, list)
    if not Nset:
        Nset = [20 for _ in range(len(trainset))]
    elif len(Nset)!=len(trainset):
        Nset = [Nset[0] for _ in range(len(trainset))]

    Tasks = []
    for tset, N in zip(trainset, Nset):
        print(f"SKETCH TASK, getting {N} tasks for training set: {tset}")
        Tasks.extend(getTasks(tset, N))

    return Tasks


## ============= TASK LIBRARY - DEVELOPING CODE.

def getTasks(taskset, N):
    """
    get list of Tasks, defined by Task sets (like what I called "lbiraries" for draw DSL)
    you can always input N, but it only matters for some tasksets
    that allow to give variable sampel size 
    """
    def grid(N):
        G = f"(loop {N} (lambda (i k) (LL (r 1 k))) k)"
        return G

    if taskset=="practice_shaping":
        ## ====== SIMPLE SHAPING TASKS
        programs = []

        # =========== 1) add grids (1 to 4)
        programs = [progFromHumanString(f"(lambda (k) (embed (lambda (k) {grid(n+1)}) (k)))") for n in range(4)]
        
        # =========== 2) as sanity check, add long veritcal line (should be identical to grid1, even though program much shoerter - CHECK)
        programs.append(progFromHumanString("(lambda (k) (LL k))"))

        # =========== 3 add all the vertical skewer types
        def vertSampler2():
            V = lambda p1, p2, p3: f"lambda (k) (d 1 ({p1} (d 1 ({p2} (d 1 ({p3} k))))))"

            v1 = V("L", "L", "L")
            v2 = V("C", "C", "C")
            import random 
            prand = lambda: random.sample(["L", "C", "E"], 1)[0]
            v3 = lambda: V(prand(), prand(), prand())
            v = lambda: random.sample([v1, v2, v3()], 1)[0]
            return v

        v = vertSampler2()
        programs.extend([progFromHumanString(f"({v()})") for _ in range(15)])

        # ============= CONVERT ALL TO TASKS
        Tasks = [SupervisedSketch(f"{taskset}_{i}", p) for i, p in enumerate(programs)]


    elif taskset=="practice":
        ## VERTICALLY STRUCTURED, WITH 4 GRID LINES
        def grid(N):
            G = f"(loop {N} (lambda (i k) (LL (r 1 k))) k)"
            return G
        grid = lambda N:f"(loop {N} (lambda (i k) (LL (r 1 k))) k)"

        def vertSampler():
            V = lambda p1, p2, p3: f"embed (lambda (k) (d 1 ({p1} (d 1 ({p2} (d 1 ({p3} k)))))))"
            
            v1 = V("L", "L", "L")
            v2 = V("C", "C", "C")
            import random 
            prand = lambda: random.sample(["L", "C", "E"], 1)[0]
            v3 = lambda: V(prand(), prand(), prand())
            v = lambda: random.sample([v1, v2, v3()], 1)[0]
            return v

        v = vertSampler()
        p4 = lambda: Program.parseHumanReadable(f"(lambda (k) (embed (lambda (k) {grid(4)}) ({v()} (r 1 ({v()} (r 1 ({v()} (r 1 ({v()} k)))))))))")
        p3 = lambda: Program.parseHumanReadable(f"(lambda (k) (embed (lambda (k) {grid(3)}) ({v()} (r 1 ({v()} (r 1 ({v()} k)))))))")
        p2 = lambda: Program.parseHumanReadable(f"(lambda (k) (embed (lambda (k) {grid(2)}) ({v()} (r 1 ({v()} k)))))")
        p1 = lambda: Program.parseHumanReadable(f"(lambda (k) (embed (lambda (k) {grid(1)}) ({v()} k)))")
        ## make a library of horizontal things

        # ==== make tasks
        Nsub = int(np.floor(N/4))
        Tasks = []
        Tasks.extend([SupervisedSketch(f"{taskset}1_{i}", p1()) for i in range(Nsub)])
        Tasks.extend([SupervisedSketch(f"{taskset}2_{i}", p2()) for i in range(Nsub)])
        Tasks.extend([SupervisedSketch(f"{taskset}3_{i}", p3()) for i in range(Nsub)])
        Tasks.extend([SupervisedSketch(f"{taskset}4_{i}", p4()) for i in range(N-3*Nsub)])

    else:
        assert False, "not yet codede other tasks..."

    return Tasks
