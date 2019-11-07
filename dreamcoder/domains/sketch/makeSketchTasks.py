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
            try:
                program = parseSketch(program)
            except:
                eprint("Parse failure:")
                eprint(program)
                assert False
            self.original = program
            plan = executeSketch(program)
        elif isinstance(program,Program):
            self.original = program
            plan = executeSketch(program)
        else:
            plan = program
        self.original = program
        state, self.plan = program.evaluate([])(_empty_sketch)(SketchState())
        self.hand = state.hand
        self.specialTask = ("sketch",
                            {"plan": self.plan})
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
        # from dreamcoder.domains.tower.tower_common import centerTower
        # def k():
        #     plan = e.evaluate([])(lambda s: (s,[]))(0)[1]
        #     if centerTower(plan) == centerTower(self.plan): return 0.
        #     return NEGATIVEINFINITY
        # try: return runWithTimeout(k, timeout)
        # except RunWithTimeout: return NEGATIVEINFINITY        
        pass
