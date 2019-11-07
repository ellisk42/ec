from dreamcoder.dreamcoder import *
from dreamcoder.domains.sketch.sketchPrimitives import *
from dreamcoder.domains.sketch.sketchPrimitives import tsketch
from dreamcoder.utilities import *
from dreamcoder.grammar import Grammar

import os
import datetime


def dreamOfSketches(grammar=Grammar.uniform(primitives), N=50, make_montage=True):
    request = arrow(tsketch, tsketch)
    programs = [p for _ in range(N) for p in [grammar.sample(request, maximumDepth=15)] if p is not None]

    # randomTowers = [tuple(centerTower(t))
    #                 for _ in range(N)
    #                 for program in [grammar.sample(request,
    #                                                maximumDepth=12,
    #                                                maxAttempts=100)]
    #                 if program is not None
    #                 for t in [executeTower(program, timeout=0.5) or []]
    #                 if len(t) >= 1 and len(t) < 100 and towerLength(t) <= 360.]
    # matrix = [renderPlan(p,Lego=True,pretty=True)
    #           for p in randomTowers]

    # # Only visualize if it has something to visualize.
    # if len(matrix) > 0:
    #     import scipy.misc
    #     if make_montage:
    #         matrix = montage(matrix)
    #         scipy.misc.imsave('%s.png'%prefix, matrix)
    #     else:
    #         for n,i in enumerate(matrix):
    #             scipy.misc.imsave(f'{prefix}/{n}.png', i)
    # else:
    #     eprint("Tried to visualize dreams, but none to visualize.")
    return programs