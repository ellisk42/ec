# =========== [NEWER VERSION, NOT USING MATPLOTLIB]
import math
import numpy as np
# import matplotlibplt

from scipy.ndimage import gaussian_filter as gf
from skimage import color
from scipy.ndimage import gaussian_filter as gf
import cairo

from dreamcoder.program import Primitive, Program
from dreamcoder.utilities import Curried
from dreamcoder.grammar import Grammar
from dreamcoder.type import baseType, arrow, tmaybe, t0, t1, t2

from dreamcoder.domains.draw.primitives import *
from dreamcoder.domains.draw.primitives import _makeAffine, _tform, _reflect, _repeat, _connect, _line, _circle, _tform_wrapper, _reflect_wrapper, _emptystroke
from dreamcoder.domains.draw.primitives import _lineC, _circleC, _finishC, _repeatC, _transformC, _reflectC, _emptystrokeC


# ========================== WHETHER TO USE NEW (DRAWBETTER) PRIMTIVIES
# USE_NEW_PRIMITIVES = True

# ======= DEFINE ALL PRIMITIVES
tstroke = baseType("tstroke")
tangle = baseType("tangle")
tscale = baseType("tscale")
tdist = baseType("tdist")
ttrorder = baseType("ttorder")
ttransmat = baseType("ttransmat")
trep = baseType("trep")


def _givemeback(thing):
    return thing


def primitiveList(USE_NEW_PRIMITIVES = True):
    """ gets the default primitivies """
    p0 = [Primitive("None", tmaybe(t0), None), Primitive("Some", arrow(t0, tmaybe(t0)), _givemeback)]

    if USE_NEW_PRIMITIVES:
        if False:
            p1 = [
                Primitive("emptystroke", tstroke, _emptystroke),
                Primitive("line", tstroke, _line), 
                Primitive("circle", tstroke, _circle),
                Primitive("transmat", arrow(tmaybe(tscale), tmaybe(tangle), tmaybe(tdist), tmaybe(tdist), tmaybe(ttrorder), ttransmat), Curried(_makeAffine)),
                Primitive("transform", arrow(tstroke, ttransmat, tstroke), Curried(_tform_wrapper)),
                Primitive("reflect", arrow(tstroke, tangle, tstroke), Curried(_reflect_wrapper)), 
                Primitive("connect", arrow(tstroke, tstroke, tstroke), Curried(_connect)),
                Primitive("repeat", arrow(tstroke, trep, ttransmat, tstroke), Curried(_repeat))
            ]

        else:
            # new version , with contiuation
            p1 = [
                Primitive("emptystrokeC", arrow(tstroke, tstroke), _emptystrokeC),
                Primitive("lineC", arrow(tstroke, tstroke), _lineC), 
                Primitive("circleC", arrow(tstroke, tstroke), _circleC),
                Primitive("transmat", arrow(tmaybe(tscale), tmaybe(tangle), tmaybe(tdist), tmaybe(tdist), tmaybe(ttrorder), ttransmat), Curried(_makeAffine)),
                Primitive("transformC", arrow(arrow(tstroke, tstroke), ttransmat, tstroke, tstroke), Curried(_transformC)),
                Primitive("reflectC", arrow(arrow(tstroke, tstroke), tangle, tstroke, tstroke), Curried(_reflectC)),
                Primitive("repeatC", arrow(arrow(tstroke, tstroke), trep, ttransmat, tstroke, tstroke), Curried(_repeatC))
            ]
    else:
        p1 = [
            Primitive("line", tstroke, _line), 
            Primitive("circle", tstroke, _circle),
            Primitive("transmat", arrow(tmaybe(tscale), tmaybe(tangle), tmaybe(tdist), tmaybe(tdist), tmaybe(ttrorder), ttransmat), Curried(_makeAffine)),
            Primitive("transform", arrow(tstroke, ttransmat, tstroke), Curried(_tform_wrapper)),
            Primitive("reflect", arrow(tstroke, tangle, tstroke), Curried(_reflect_wrapper)), 
            Primitive("connect", arrow(tstroke, tstroke, tstroke), Curried(_connect)),
            Primitive("repeat", arrow(tstroke, trep, ttransmat, tstroke), Curried(_repeat))
        ]


    # p2 = [Primitive("scale{}".format(i), tscale, j) for i, j in enumerate(np.linspace(1.0, 4.0, 7))]
    p2 = [Primitive("scale{}".format(i), tscale, j) for i, j in enumerate(SCALES)] 
    # p3 = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(np.linspace(-4, 4, 9))]
    p3 = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(DISTS)]
    NANGLE = 8
    # p4 = [Primitive("angle{}".format(i), tangle, j*2*math.pi/NANGLE) for i, j in enumerate(range(NANGLE))]
    p4 = [Primitive("angle{}".format(i), tangle, j) for i, j in enumerate(THETAS)]
    # p5 = [Primitive("angle{}".format(i), tangle, (j+1)*2*math.pi/3) for j,i in enumerate(range(NANGLE, NANGLE+2))]
    p5 = []
    p6 = [Primitive(j, ttrorder, j) for j in ORDERS]
    p7 = [Primitive("rep{}".format(i), trep, j+1) for i, j in enumerate(range(7))]

    primitives = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7

    return primitives



def getPrimitives(trainset="", prune=False, primitives=None, fullpruning=True, USE_NEW_PRIMITIVES = True, 
    suppress_print=False):
    """ get primitives for each specific experiment"""

    primitives = primitiveList(USE_NEW_PRIMITIVES = USE_NEW_PRIMITIVES)

    if prune:
        assert len(trainset)>0, "Have to tell me which trainset to use for primtives"
        # -- then only keep a subset of primtiives
        if trainset in ["S12", "S13"]:

            if not suppress_print:
                print("Full primitives:")
                for p in primitives:
                    print("{} = {}".format(p.name, p.evaluate([])))

            # ------- list of primtiives to remove
            primitives_to_remove = [] + \
            ["scale{}".format(i) for i in [0, 1, 2, 3, 4, 5, 6]] + \
            ["dist{}".format(i) for i in [0, 2, 5, 7, 10, 12, 13, 19, 20, 21, 22]] + \
            ["angle{}".format(i) for i in [1,3,5,7,8,9]] + \
            ["tsr", "srt", "str", "rts", "rst"] + \
            ["rep{}".format(i) for i in [4,5,6]]
            
            if fullpruning:
                # then really careful remove anything not useful
                # partly motivated by seeing what DC actually uses given the partial pruning above.
                #primitives_to_remove.extend(["dist11", "dist8", "reflect", "angle4", "angle6"])
                primitives_to_remove.extend(["dist11", "dist8", "angle4", "angle6"])

            if not suppress_print:
                print("removing these primitives:")
                print(primitives_to_remove)

            # ----- do removal
            primitives = [p for p in primitives if p.name not in primitives_to_remove]

            if not suppress_print:
                print("Primtives, after pruning:")
                for p in primitives:
                    print("{} = {}".format(p.name, p.evaluate([])))

            return primitives
        else:
            print("DO NOT KNOW HOW TO PRUNE PRIMITIVES FOR THIS TRAINSET")
            raise
    else:
        return primitives


def getNewPrimitives():
    """ gets primtiives that are needed for doing the latest hand built stuff.
    - expected that the base primtiives you pass in are USE_NEW_PRIMITIVES=False, which
    makes them the old versions"""

    primitives = []
    primitives.append(Primitive("emptystroke", tstroke, _emptystroke))
    primitives.append(Primitive("dist98", tdist, -0.2))
    primitives.append(Primitive("dist99", tdist, 0.2))
        
    return primitives

def getPrimitivesUpdated(trainset):
    """ 5/2020, these are for second round of experiments with this DSL, 
    using the original DSL for cogsci paper, but with slight modifictiosn:
    adding a couple primtiives that are required for the hand built inventions
    to work. motviation is that this i the minimal set to run all models 
    fairly starting form the same primitives (but with different inventions).
    NOTE: this is going back to before new updated DSL with continuation, and
    the updated DSL that has emptystroke
    - trainset in {"S12", "S13"}"""
    
    primitives = getPrimitives(trainset=trainset, prune=True, USE_NEW_PRIMITIVES=False)
    primitives.extend(getNewPrimitives())

    return primitives



# ============================= NEW COMPLEX PRIMITIVES "HAND-BUILT" INTO MODELS
def getHandcodedInventions(model, plot_inventions = False):
    """
    - hand coded inventions that are inspired by generative
    model used for the stimuli themselves. 
    - model indicates which hand-built model to use. There are 
    three so far: model = ["S12skew", "S12grate", "S13grate"].
    - These inventions can be added to primtiives used for initiating dreamcoder:
    primitives.extend(Inventions)
    """
    from dreamcoder.program import Invented

    Inventions = []

    ########### USEFUL VARIABLES
    def Tp(p, s=None, th=None, x=None, y=None, o=None):
        """
        - p is primitive, e.g., circle, or k (if variable)
        - s, th, x, y, and o, are transmat stuff
        - all arguments can be string or variable. either way will convert it 
        veridically to a string.
        """
        if s is not None:
            s = f"(Some {s})"
        if th is not None:
            th = f"(Some {th})"
        if x is not None:
            x = f"(Some {x})"
        if y is not None:
            y = f"(Some {y})"
        if o is not None:
            o = f"(Some {o})"
            
        s = f"(transform {p} (transmat {s} {th} {x} {y} {o}))"
    #     print(s)
        return s
    ll = f"{Tp('line', th='angle2', s='scale7', y='dist1')}"
    lh = f"{Tp('line', x='dist4')}"
    
    
    
    ########### MAKE INVENTIONS
    # --- for all models
    # 2) ==== centered horizontal line
    s = lh
    p = Program.parseHumanReadable(s)
    if plot_inventions:
        plot(p.runWithArguments([]))
    Inventions.append(Invented(p))

    
    if model in ["S12skew", "S12grate"]:
        # 2) ==== long vertical line
        s = ll
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([]))
        Inventions.append(Invented(p))

        # 4) ==== four x position slots
        s = f"(lambda (P1 P2 P3 P4) (connect {Tp('P1', x='dist14')} (connect {Tp('P2', x='dist15')} (connect {Tp('P3', x='dist16')} {Tp('P4', x='dist17')}))))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([primitives[3].evaluate([]), primitives[3].evaluate([]), primitives[2].evaluate([]), primitives[3].evaluate([])]))
        Inventions.append(Invented(p))

        # 5) ==== repeat P N times (horizontal)
        s = f"(lambda (P N) (repeat {Tp('P', x='dist14')} N (transmat None None (Some dist18) None None)))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([primitives[2].evaluate([]), 2]))
        Inventions.append(Invented(p))

    
    if model in ["S12grate", "S13grate"]:
        # 4) ==== repeat vertical line (grating)
        s = f"(lambda (N) (repeat {Tp(ll, x='dist14')} N (transmat None None (Some dist18) None None)))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([3]))
        Inventions.append(Invented(p))
  

    if model=="S12skew":
        # 2) ==== skewer (+ vertical line)
        s = f"(lambda (k l m) (connect {ll} (connect {Tp('k', y='dist3')} (connect {Tp('l', y='dist6')} {Tp('m', y='dist9')}))))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([primitives[2].evaluate([]), primitives[3].evaluate([]),primitives[3].evaluate([])]))
        Inventions.append(Invented(p))


    if model=="S12grate":
        # 1) ==== skewer (no vertical line)
        s = f"(lambda (k l m) (connect {Tp('k', y='dist3')} (connect {Tp('l', y='dist6')} {Tp('m', y='dist9')})))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([primitives[2].evaluate([]), primitives[3].evaluate([]),primitives[3].evaluate([])]))
        Inventions.append(Invented(p))
    
    
    if model=="S13grate":
        # 6) ==== lolli (1,2)
        def Rp(p, N):
            """ repeat p N times"""
            s = f"(repeat {Tp(p, x='dist14')} {N} (transmat None None (Some dist18) None None))"
            return s

        s = f"(lambda (N) (connect {Tp('circle', x='dist14')} {Tp(Rp(lh, N='N'), x='dist18')}))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([3]))
        Inventions.append(Invented(p))

        # 7) ==== dumbell (1-2-3)
        # s = f"(lambda (N) (connect {Tp('circle', x=} (connect {Tp('circle', x='dist14')} {Tp(Rp(lh), x='dist18')})))"
        s = f"(connect {Tp('circle', x='dist16')} (connect {Tp('circle', x='dist14')} {Tp(Rp(lh, N='rep0'), x='dist18')}))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([]))
        Inventions.append(Invented(p))

        s = f"(connect {Tp('circle', x='dist17')} (connect {Tp('circle', x='dist14')} {Tp(Rp(lh, N='rep1'), x='dist18')}))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([]))
        Inventions.append(Invented(p))

        # 7) ==== horizontal line (variable length [123] and position)
        s = f"(lambda (N) (repeat {Tp(lh, x='dist14')} N (transmat None None (Some dist18) None None)))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([4]))
        Inventions.append(Invented(p))

        # 8) ==== reflect across vertical axis
        s = f"(lambda (P) {Tp('(reflect P angle2)', x='dist98')})"
        print(s)
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            ss = f"(lambda (N) (connect {Tp('circle', x='dist14')} {Tp(Rp(lh, N='N'), x='dist18')}))" # get another primitive to perform refelction on.
            plot(p.runWithArguments([Program.parseHumanReadable(ss).runWithArguments([2])]))
        Inventions.append(Invented(p))
        
        # 8) ==== 3 slots on y axis
        s = f"(lambda (P1 P2 P3) (connect {Tp('P1', y='dist3')} (connect {Tp('P2', y='dist6')} {Tp('P3', y='dist9')})))"
        p = Program.parseHumanReadable(s)
        if plot_inventions:
            plot(p.runWithArguments([primitives[3].evaluate([]), primitives[2].evaluate([]), primitives[3].evaluate([])]))
        Inventions.append(Invented(p))
    return Inventions
