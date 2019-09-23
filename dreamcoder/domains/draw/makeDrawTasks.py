from dreamcoder.domains.draw.drawPrimitives import tstroke, tangle, tscale, tdist, ttrorder, ttransmat, trep, primitives, loss
# from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
# from dreamcoder.utilities import eprint, testTrainSplit, loadPickle


class SupervisedDraw(Task):
    def __init__(self, name, program, mustTrain=False):
        if isinstance(program,str):
            try:
                program = parseTower(program)
            except:
                eprint("Parse failure:")
                eprint(program)
                assert False
            self.original = program
            plan = executeTower(program)
        elif isinstance(program,Program):
            self.original = program
            plan = executeTower(program)
        else:
            plan = program
        self.original = program
        state, self.plan = program.evaluate([])(_empty_tower)(TowerState())
        self.hand = state.hand
        super(SupervisedTower, self).__init__(name, arrow(ttower,ttower), [],
                                              features=[]) # TODO: LT, needs this, i.e., a request. 
        self.mustTrain = mustTrain

    def logLikelihood(self, e, timeout=None): # TODO, LT, given expression, calculates distance.
        # from dreamcoder.domains.tower.tower_common import centerTower
        p1 = self.program.evaluate([])
        p2 = e.evaluate([])
        l = loss(p1, p2, smoothing=2)

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


