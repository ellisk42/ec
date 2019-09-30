from dreamcoder.domains.draw.drawPrimitives import *
from dreamcoder.domains.draw.drawPrimitives import _tform, _line, _circle, _repeat, _makeAffine
# from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.utilities import *
import math


class SupervisedDraw(Task):
    def __init__(self, name, program):
        super(SupervisedDraw, self).__init__(name, tstroke, [],
                                              features=[]) # TODO: LT, needs this, i.e., a request. 

        self.strokes = program # list of np arrays.
        self.rendered_strokes = prog2pxl(program)


    def logLikelihood(self, e, timeout=None): # TODO, LT, given expression, calculates distance.
        # from dreamcoder.domains.tower.tower_common import centerTower
        if False:
            p1 = self.rendered_strokes
            p2 = fig2pixel(e.evaluate([]))

            # l = loss(p1, p2, smoothing=2) 
            l = loss_pxl(p1, p2)

            if l>0.1:
                return NEGATIVEINFINITY
            else:
                return 0.0
        else:
            print("doing it!")
            p1 = self.rendered_strokes
            p2 = prog2pxl(e.evaluate([]))

            # l = loss(p1, p2, smoothing=2) 
            l = loss_pxl(p1, p2)

            if l>0.1:
                return NEGATIVEINFINITY
            else:
                return 0.0

    
def makeSupervisedTasks(): # TODO, LT, make these tasks.
    # arches = [SupervisedTower("arch leg %d"%n,
    #                           "((for i %d v) (r 4) (for i %d v) (l 2) h)"%(n,n))
    #           for n in range(1,9)
    # ]
                     
    # everything = arches + simpleLoops + Bridges + archesStacks + aqueducts + offsetArches + pyramids + bricks + staircase2 + staircase1 + compositions
    alltasks = []

    programs = [_line + _circle,
    _circle + _line,
    _line + _tform(_line, _makeAffine(x=2.)) + t_form(_circle, _makeAffine(x=-1.))
    _repeat(_line+_tform(_circle, _makeAffine(x=1.)), 3, _makeAffine(theta=math.pi/2))
    ]
    for i, p in enumerate(programs):
        name = "task{}".format(i)
        alltasks.append(SupervisedDraw(name, p))

    return alltasks

