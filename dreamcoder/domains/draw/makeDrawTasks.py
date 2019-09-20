from dreamcoder.domains.draw.makeDrawTasks import drawDrawings
from dreamcoder.domains.draw.drawPrimitives import primitives, taxes, tartist, tangle, tscale, tdist
# from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
# from dreamcoder.utilities import eprint, testTrainSplit, loadPickle


class SupervisedTower(Task):
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
        self.specialTask = ("supervisedTower",
                            {"plan": self.plan})
        self.image = None
        self.handImage = None
        self.mustTrain = mustTrain

    def logLikelihood(self, e, timeout=None): # TODO, LT, given expression, calculates distance.
        from dreamcoder.domains.tower.tower_common import centerTower
        def k():
            plan = e.evaluate([])(lambda s: (s,[]))(0)[1]
            if centerTower(plan) == centerTower(self.plan): return 0.
            return NEGATIVEINFINITY
        try: return runWithTimeout(k, timeout)
        except RunWithTimeout: return NEGATIVEINFINITY        
        
   

    
def makeSupervisedTasks(): # TODO, LT, make these tasks.
    arches = [SupervisedTower("arch leg %d"%n,
                              "((for i %d v) (r 4) (for i %d v) (l 2) h)"%(n,n))
              for n in range(1,9)
    ]
                     
    everything = arches + simpleLoops + Bridges + archesStacks + aqueducts + offsetArches + pyramids + bricks + staircase2 + staircase1 + compositions


