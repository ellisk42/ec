# coding: utf8

import os
import random
import sys

from dreamcoder.domains.logo.logoPrimitives import primitives, turtle
from dreamcoder.task import Task
from dreamcoder.program import Abstraction, Application, Index, Program
from dreamcoder.type import arrow
from dreamcoder.utilities import eprint, jsonBinaryInvoke, random_seed, montage
from dreamcoder.grammar import Grammar


def drawLogo(*programs,
             timeout=None,
             resolution=None,
             pretty=False, smoothPretty=False,
             filenames=[],
             animate=False,
             cost=False):
    message = {}
    if pretty: message["pretty"] = pretty
    if smoothPretty: message["smoothPretty"] = smoothPretty
    if timeout: message["timeout"] = timeout
    assert resolution is not None, "resolution not provided in drawLogo"
    if isinstance(resolution, list):
        assert len(resolution) == len(programs), "must provide a resolution for each program"
    elif isinstance(resolution, int):
        resolution = [resolution]*len(programs)
    else: assert False
    jobs = []
    for p, size in zip(programs, resolution):
        entry = {"program": str(p),
                 "size": size}
        if animate: entry["animate"] = True
        if len(filenames) > 0:
            entry["export"] = filenames[0]
            filenames = filenames[1:]
        jobs.append(entry)        
    message["jobs"] = jobs
    response = jsonBinaryInvoke("./logoDrawString", message)
    if cost:
        # include the cost and return tuples of (pixels, cost)
        response = [programResponse if isinstance(programResponse,str) else (programResponse["pixels"], programResponse["cost"])
                    for programResponse in response ]
    else:
        response = [programResponse if isinstance(programResponse,str) else programResponse["pixels"]
                    for programResponse in response ]
    if len(programs) == 1:            
        return response[0]
    return response

def makeTasks(subfolders, proto):
    return manualLogoTasks()

def parseLogo(s):
        
    _ua = Program.parse("logo_UA")
    _ul = Program.parse("logo_UL")

    _za = Program.parse("logo_ZA")
    _zl = Program.parse("logo_ZL")

    _da = Program.parse("logo_DIVA")
    _ma = Program.parse("logo_MULA")
    _dl = Program.parse("logo_DIVL")
    _ml = Program.parse("logo_MULL")

    _aa = Program.parse("logo_ADDA")
    _sa = Program.parse("logo_SUBA")
    _al = None#Program.parse("logo_ADDL")
    _sl = None#Program.parse("logo_SUBL")

    _pu = None#Program.parse("logo_PU")
    _pd = None#Program.parse("logo_PD")
    _p = Program.parse("logo_PT")
    _move = Program.parse("logo_FWRT")
    _embed = Program.parse("logo_GETSET")

    _addition = Program.parse("+")
    _infinity = Program.parse("logo_IFTY")
    _ea = Program.parse("logo_epsA")
    _el = Program.parse("logo_epsL")
    _loop = Program.parse("logo_forLoop")

    from sexpdata import loads, Symbol
    s = loads(s)
    def command(k, environment, continuation):
        assert isinstance(k,list)
        if k[0] == Symbol("move"):
            return Application(Application(Application(_move,
                                                       expression(k[1],environment)),
                                           expression(k[2],environment)),
                               continuation)
        if k[0] == Symbol("for") or k[0] == Symbol("loop"):
            v = k[1]
            b = expression(k[2], environment)
            newEnvironment = [None, v] + environment
            body = block(k[3:], newEnvironment, Index(0))
            return Application(Application(Application(_loop,b),
                                           Abstraction(Abstraction(body))),
                               continuation)
        if k[0] == Symbol("embed"):
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(_embed,Abstraction(body)),continuation)
        if k[0] == Symbol("p"):
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(_p,Abstraction(body)),continuation)

        assert False
    def expression(e, environment):
        for n, v in enumerate(environment):
            if e == v: return Index(n)

        if isinstance(e,int): return Program.parse(str(e))

        mapping = {"1a": _ua,
                   "1d": _ul, "1l": _ul,
                   "0a": _za,
                   "0d": _zl, "0l": _zl,
                   "/a": _da,
                   "/l": _dl, "/d": _dl,
                   "*a": _ma,
                   "*l": _ml, "*d": _ml,
                   "+a": _aa,
                   "+d": _al, "+l": _al,
                   "-a": _sa,
                   "-d": _sl, "-l": _sl,
                   "+": _addition,
                   "infinity": _infinity,
                   "epsilonAngle": _ea,
                   "epsilonDistance": _el,
                   "epsilonLength": _el}
        if e == float('inf'): return _infinity
        for name, value in mapping.items():
            if e == Symbol(name): return value
            
        assert isinstance(e,list), "not a list %s"%e
        for name, value in mapping.items():
            if e[0] == Symbol(name):
                f = value
                for argument in e[1:]:
                    f = Application(f, expression(argument, environment))
                return f
        assert False
        
    def block(b, environment, continuation):
        if len(b) == 0: return continuation
        return command(b[0], environment, block(b[1:], environment, continuation))

    try: return Abstraction(command(s, [], Index(0)))
    except: return Abstraction(block(s, [], Index(0)))


def manualLogoTask(name, expression, proto=False, needToTrain=False,
                   supervise=False, lambdaCalculus=False):
    p = Program.parse(expression) if lambdaCalculus else parseLogo(expression)
    from dreamcoder.domains.logo.logoPrimitives import primitives
    from dreamcoder.grammar import Grammar
    g = Grammar.uniform(primitives, continuationType=turtle)
    gp = Grammar.uniform(primitives)
    try:
        l = g.logLikelihood(arrow(turtle,turtle),p)
        lp = gp.logLikelihood(arrow(turtle,turtle),p)
        assert l >= lp
        eprint(name,-l,"nats")
        
    except: eprint("WARNING: could not calculate likelihood of manual logo",p)

    attempts = 0
    while True:
        [output, highresolution] = drawLogo(p, p, resolution=[28,128], cost=True)
        if output == "timeout" or highresolution == "timeout":
            attempts += 1
        else:
            break
    if attempts > 0:
        eprint(f"WARNING: Took {attempts} attempts to render task {name} within timeout")

    cost = output[1]
    output = output[0]
    assert highresolution[1] == cost
    highresolution = highresolution[0]
            
    shape = list(map(int, output))
    highresolution = list(map(float, highresolution))
    t = Task(name, arrow(turtle,turtle),
             [(([0]), shape)])
    t.mustTrain = needToTrain
    t.proto = proto
    t.specialTask = ("LOGO", {"proto": proto})
    t.specialTask[1]["cost"] = cost*1.05

    t.highresolution = highresolution

    if supervise:
        t.supervisedSolution = p

    return t

def dSLDemo():
    n = 0
    demos = []
    def T(source):
        demos.append(manualLogoTask(str(len(demos)), source,
                                    lambdaCalculus="lambda" in source))
    # this looks like polygons - verify and include
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 4) 3)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 6) 4)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 5) 5)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 3) 6)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 2) 7)")

    # Spirals!
    for spiralSize in [1,2,3,4,5]:
        T(f"((lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT (logo_MULL logo_epsL $1) (logo_MULA logo_epsA $2) $0))))) {spiralSize})")
    for spiralSize in [5,6,7,8,9]:
        #T(f"(lambda (#(lambda (logo_forLoop $0 (lambda (lambda (#(lambda (logo_FWRT (logo_MULL logo_UL $0) (logo_DIVA logo_UA 4))) $1 $0))))) {spiralSize} $0))")
        T("(loop i " + str(spiralSize) + " (move (*d 1l i) (/a 1a 4)))")# (#(lambda (logo_forLoop $0 (lambda (lambda (#(lambda (logo_FWRT (logo_MULL logo_UL $0) (logo_DIVA logo_UA 4))) $1 $0))))) {spiralSize} $0))")

    # CIRCLES
    #(lambda (#(lambda (logo_forLoop 6 (lambda (lambda (#(lambda (lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT $2 $3 $0)))))) logo_epsA (logo_MULL logo_epsL $2) $0))))) 6 $0))
    for circleSize in [1,3,5,7,9]:
        T(f"(lambda (#(lambda (logo_forLoop 6 (lambda (lambda (#(lambda (lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT $2 $3 $0)))))) logo_epsA (logo_MULL logo_epsL $2) $0))))) {circleSize} $0))")
    
    T("(loop i 3 (move (*d 1l 3) (/a 1a 4)))")
    T("(loop i 5 (move (*d 1l 5) (/a 1a 5)))")
    T("(loop i infinity (move (*d epsilonDistance 5) (/a epsilonAngle 3)))")
    T("(loop i infinity (move (*d epsilonDistance 9) (/a epsilonAngle 2)))")
    T("(loop i infinity (move (*d epsilonLength i) (*a epsilonAngle 3)))")
    T("(loop i 9 (move (*d 1l i) (/a 1a 4)))")
    T("(move 1d 0a)")
    T("(loop i infinity (move (*d epsilonLength 6) epsilonAngle))")
    T("(loop i infinity (move (*d epsilonLength 8) epsilonAngle))")
    T("(loop k 2 (loop i infinity (move (*d epsilonLength 4) epsilonAngle)))")
    T("(loop k 2 (loop i infinity (move (*d epsilonLength 8) epsilonAngle)))")
    T("(loop s 4 (move (*d 1d 3) (/a 1a 4)))")
    T("(loop s 4 (move (*d 1d 6) (/a 1a 4)))")
    T("""
          (loop j 5
          (move 0d (/a 1a 5))
          (embed (loop i infinity
          (move (*d epsilonLength 6) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength 6) epsilonAngle))))""")
    T("""
              (loop j 5
              (embed (loop s 4 (move (*d 1d 3) (/a 1a 4))))
              (move 0d (/a 1a 5)))""")
    return demos

def rotationalSymmetryDemo():
    demos = []
    def T(source):
        demos.append(manualLogoTask(str(len(demos)), source))
            
    body = {"dashed": "(p (move 1d 0a)) (move 1d 0a) (p (move 1d 0a)) (move 1d 0a)",
            "lonely circle": "(p (move (*d 1d 2) 0a)) (loop k 2 (loop i infinity (move (*d epsilonLength 2) epsilonAngle)))",
            "square dashed": "(p (move 1d 0a)) (loop s 4 (move 1d (/a 1a 4)))",
            "square": "(loop s 4 (move (*d 1d 2) (/a 1a 4)))",
            "semicircle": "(loop i infinity (move (*d epsilonLength 4) epsilonAngle))"}
    for name in body:
        for n in [3,4,5,6,7]:
            T("""
              (loop j %d
              (embed %s)
              (move 0d (/a 1a %d)))"""%(n,body[name],n))
    return demos
              

def manualLogoTasks():
    tasks = []
    def T(name, source, needToTrain=False, supervise=False):
        tasks.append(manualLogoTask(name, source, supervise=supervise,
                                    needToTrain=needToTrain))
    if False:
        for d,a,s in [('1l','0a','(loop i infinity (move epsilonLength epsilonAngle))'),
                      ('epsilonLength','0a','(loop i infinity (move epsilonLength epsilonAngle))'),
                      ('(*d 1l 3)','0a','(move 1l 0a)'),
                      ('epsilonLength','0a','(move (*d 1l 2) 0a)'),
                      ('(*d epsilonLength 9)','0a','(move epsilonLength 0a)'),
                      ('(/d 1l 2)','0a','(move 1l 0a)')]:
            #            'epsilonLength']:
            # for a in ['epsilonAngle','0a']:
            #     for s in ['(move 1l 0a)',
            #               '(move epsilonLength 0a)',
            #               '(loop i infinity (move epsilonLength epsilonAngle))']:
            #         if d == 'epsilonLength' and s == '(move epsilonLength 0a)': continue
            T("pu: %s/%s/%s"%(d,a,s),
              """
              (pu (move %s %s) pd %s)
              """%(d,a,s))
        return tasks

    def slant(n):
        return f"(move 0d (/a 1a {n}))"

    for n,l,s in [(3,"1l",8),
                  (4,"(*d 1d 3)",None),
                  (5,"1l",None),
                  (6,"(*d 1d 2)",5),
                  (7,"1l",None),
                  (8,"(/d 1d 2)",None)]:
        T(f"{n}-gon {l}{'' if s is None else ' slanted '+str(s)}",
          f"""
          ({'' if s is None else slant(s)}
           (loop i {n}
            (move {l} (/a 1a {n}))))
          """,
          needToTrain=True)
    for n,l,s in [(3,"(*d 1l 2)",None),
                (4,"(*d 1d 4)",None),
                (5,"(*d 1d 2)",None),
                (6,"1l",None),
                (7,"(*d 1d 3)",None),
                (8,"1l",3)]:
        T(f"{n}-gon {l}{'' if s is None else ' slanted '+str(s)}",
          f"""
          ({'' if s is None else slant(s)}
           (loop i {n}
            (move {l} (/a 1a {n}))))
          """,
          needToTrain=False)

        

    T("upwards", "((move 0d (/a 1a 4)) (move 1d 0a))",
      needToTrain=True)
    T("right angle", "((move (*d 1d 2) (/a 1a 4)) (move 1d 0a))",
      needToTrain=True)
    T("right angle epsilon", "((move epsilonLength (/a 1a 4)) (move epsilonLength 0a))",
      needToTrain=True)

    T("line segment", "(move 1d 0a)",
      needToTrain=True)

    T("square slanted by 2pi/3",
      """((move 0d (/a 1a 3))
      (loop k 4 (move 1d (/a 1a 4))))""",
      needToTrain=True)
    T("semicircle slanted by 2pi/5",
      """((move 0d (/a 1a 5))
      (loop i infinity
      (move (*d epsilonLength 4) epsilonAngle)))""",
      needToTrain=True)
    T("Greek spiral slanted by 2pi/6",
      """((move 0d (/a 1a 6))
      (loop i 7 (move (*l 1l i) (/a 1a 4))))""",
      needToTrain=True)
    T("Hook slanted by 2pi/7",
      """((move 0d (/a 1a 7))
      (move 1d 0a)
      (loop i infinity
      (move (*d epsilonLength 4) epsilonAngle)))""",
      needToTrain=True)
    T("""slanted line""",
      """((move 0d (/a 1a 8))
      (move (*d 1l 3) 0a))""",
      needToTrain=True)
    

    for i in [6,7,8,9]:
        T("Greek spiral %d"%i,
          """
          (loop i %d
          (move (*l 1l i) (/a 1a 4)))
          """%i,
          needToTrain=i in [7,8])
    for i in [2,3,4,5]:
        T("smooth spiral %d"%i,
          """
          (loop i infinity 
          (move (*d epsilonLength i) (*a epsilonAngle %d)))
          """%i,
          needToTrain=i in [3,5])

    T("smooth spiral 4 slanted by 2pi/2",
      """
          ((move 0d (/a 1a 2))
      (loop i infinity 
          (move (*d epsilonLength i) (*a epsilonAngle 4))))
      """,
      needToTrain=True)

    for i in [3,5,7,9]:
        T("star %d"%i,
          """
          (loop i %d (move (*d 1d 4) (-a (/a 1a 2) (/a (/a 1a 2) %s))))
          """%(i,i),
          needToTrain=i in [5,9])

    T("leaf iteration 1.1",
      """
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      """,
      needToTrain=True)
    T("leaf iteration 1.2",
      """
      ((move 0d (/a 1a 2))
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2))))
      """,
      needToTrain=True)
    T("leaf iteration 2.1",
      """
      (loop n 2
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      (move 0d (/a 1a 4)))
      """,
      needToTrain=True)
    T("leaf iteration 2.2",
      """
      ((move 0d (/a 1a 2))
      (loop n 2
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      (move 0d (/a 1a 4))))
      """,
      needToTrain=True)
    for n in range(3,8):
        T("flower %d"%n,
          """
          (loop j %d
          (loop n 2
          (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
          (move 0d (/a 1a 4)))
          (move 0d (/a 1a %d)))
          """%(n,n),
          needToTrain=n in range(3,5))        

    for n in [5,6]:
        T("staircase %d"%n,
          """
          (loop i %d
          (move 1d (/a 1a 4))
          (move 1d (/a 1a 4))
          (move 0d (/a 1a 2)))
          """%n,
          needToTrain=n in [5])

    for n in range(1,6):
        T("blocks zigzag %d"%n,
          """
          (loop i %d
          (move 1d (/a 1a 4)) (move 1d (/a 1a 4))
          (move 1d (+a (/a 1a 2) (/a 1a 4))) (move 1d (+a (/a 1a 2) (/a 1a 4))))
          """%n,
          needToTrain=n in [1,2,3])
    for n in [3,4]:#range(1,5):
        T("diagonal zigzag %d"%n,
          """
          ((move 0d (/a 1a 8))
          (loop i %d
          (move 1d (/a 1a 4)) 
          (move 1d (+a (/a 1a 2) (/a 1a 4)))))
          """%n,
          needToTrain=n == 4)

    

    for n in [1,2,3,4,5,6]:
        T("right semicircle of size %d"%n,
          """
          (loop i infinity
          (move (*d epsilonLength %d) (-a 0a epsilonAngle)))
          """%n,
          needToTrain=n%2 == 0)
        T("left semicircle of size %d"%n,
          f"""
          ({'' if n != 1 else slant(8)}
           (loop i infinity
            (move (*d epsilonLength {n}) epsilonAngle)))
          """,
          needToTrain=n%2 == 1)
        T("circle of size %d"%n,
              """
              ((loop i infinity
              (move (*d epsilonLength %d) epsilonAngle))
              (loop i infinity
              (move (*d epsilonLength %d) epsilonAngle)))
              """%(n,n),
          needToTrain=n in [1,4,3,5,6])

    for n in [5,6]:
        T("%d enclosed circles"%n,
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle)))"""%n,
          needToTrain=n == 5)

    for n,l in [(4,2),
                (5,3),
                (6,4),
                (3,1)]:
        T("%d-circle flower l=%d"%(n,l),
          """
          (loop j %d
          (move 0d (/a 1a %d))
          (embed (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))))"""%(n,n,l,l),
          needToTrain=(n,l) in [(6,4),(3,1)])

    for n,l in [(3,1),(2,2),(1,3),
                (2,1),(1,2),(1,1)]:
        T("%d-semicircle sequence L=%d"%(n,l),
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength %d) (-a 0a epsilonAngle))))
          """%(n,l,l),
          needToTrain=(n,l) in [(3,1),(2,2),(1,3)])

    for n,l in [(2,"1d"),
                (3,"1d")]:
        T("row of %d circles"%n,
          """
          (loop j %d
          (embed (loop k 2 (loop i infinity (move epsilonLength epsilonAngle))))
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 2)
    for n,l in [(2,"1d"),
                (3,"1d")]:
        T("row of %d lines"%n,
          """
          (loop j %d
          (move 1d 0a)
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 2)
    T("line next to semicircle",
      """
      ((move 1d 0a) (p (move 1d 0a)) (loop i infinity (move epsilonLength epsilonAngle)))
      """,
      needToTrain=True)
    for n,l in [(3,"(/d 1d 2)"),
                (4,"(/d 1d 3)")]:
        T("%d dashed lines of size %s"%(n,l),
          """(loop i %d (p (move 1d 0a)) (move %s 0a))"""%(n,l),
          needToTrain=n == 3)
    T("broken circle",
      """
      ((loop i infinity (move epsilonLength epsilonAngle)) (p (move 1d 0a)) (loop i infinity (move epsilonLength epsilonAngle)))
      """,
      needToTrain=True)
    T("circle next to semicircle",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (loop i infinity (move epsilonLength epsilonAngle)))
      """,
      needToTrain=True)
    T("semicircle next to square",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (loop i infinity (move 1d (/a 1a 4))))
      """,
      needToTrain=False)
    T("circle next to square",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (loop i infinity (move 1d (/a 1a 4))))
      """,
      needToTrain=False)
    T("circle next to line",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (move 1d 0a))
      """,
      needToTrain=True)
    T("line next to circle",
      """
      ((move 1d 0a)
      (p (move 1d 0a))
      (loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))      
      (move 1d 0a))
      """,
      needToTrain=True)
    for n,l in [(4,"1d"),
                (5,"1d")]:
        T("row of %d dashes"%n,
          """
          (loop j %d
          (embed (move 0d (/a 1a 4)) (move 1d 0a))
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 4)        
    for n,l in [(5,"1d"),(6,"1d")]:
        T("row of %d semicircles"%n,
          """
          (loop j %d
          (embed (loop i infinity (move epsilonLength epsilonAngle)))
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 5)

    with random_seed(42): # carefully selected for maximum entropy
        for n in [3,4,5,6,7]:
            body = {"empty": "(move 1d 0a)",
                    "spiral": "(loop i infinity (move (*d epsilonLength i) (*a epsilonAngle 2)))",
                    "dashed": "(p (move 1d 0a)) (move 1d 0a)",
                    "circle": "(move 1d 0a) (loop k 2 (loop i infinity (move epsilonLength epsilonAngle)))",
                    "lonely circle": "(p (move 1d 0a)) (loop k 2 (loop i infinity (move epsilonLength epsilonAngle)))",
                    "square dashed": "(p (move 1d 0a)) (loop s 4 (move 1d (/a 1a 4)))",
                    "square": "(move 1d 0a) (loop s 4 (move 1d (/a 1a 4)))",
                    "close large semicircle": "(loop i infinity (move (*d epsilonLength 2) epsilonAngle))",
                    "close semicircle": "(loop i infinity (move epsilonLength epsilonAngle))",
                    "semicircle": "(move 1d 0a) (loop i infinity (move epsilonLength epsilonAngle))",
                    "double dashed": "(p (move 1d 0a)) (move 1d 0a) (p (move 1d 0a)) (move 1d 0a)",
                    "Greek": "(loop i 3 (move (*l 1l i) (/a 1a 4)))"}
            for name in body:
                if name == "spiral" and n not in [3,5]: continue
                if name == "square" and n not in [5,3,6,7]: continue
                if name == "semicircle" and n not in [5,3,4,6]: continue
                if name == "Greek" and n not in [3,5]: continue
                if name == "double dashed" and n not in [6,4,3]: continue
                
                mustTrain = False

                mustTrain = mustTrain or (n == 3 and name == "Greek")
                mustTrain = mustTrain or (n == 7 and name == "empty")
                mustTrain = mustTrain or (n == 5 and name == "dashed")
                mustTrain = mustTrain or (n == 7 and name == "circle")
                mustTrain = mustTrain or (n == 6 and name == "circle")
                mustTrain = mustTrain or (n == 6 and name == "lonely circle")
                mustTrain = mustTrain or (n == 5 and name == "square")
                mustTrain = mustTrain or (n == 7 and name == "square")
                mustTrain = mustTrain or (n == 5 and name == "semicircle")
                mustTrain = mustTrain or (n == 3 and name == "square dashed")
                mustTrain = mustTrain or (n == 6 and name == "close semicircle")
                mustTrain = mustTrain or (n == 5 and name == "close large semicircle")
                mustTrain = mustTrain or (n == 3 and name == "spiral")
                mustTrain = mustTrain or (n == 6 and name == "double dashed")
                mustTrain = mustTrain or (n == 3 and name == "double dashed")
                #mustTrain = mustTrain or (n == 6 and name == "empty")

                #mustTrain = mustTrain or (random.random() < 0.07) # calibrated to give 70 training tasks
                

                # # cap number of super easy snowflakes
                # if name == "empty" and n not in [7]: mustTrain = False
                # if name == "dashed" and n not in [4]: mustTrain = False
                

                T("%d-%s snowflake"%(n,name),
                  """
                  (loop j %d
                  (embed %s)
                  (move 0d (/a 1a %d)))"""%(n,body[name],n),
                  needToTrain=mustTrain)

    for n in [3,4]:#2,3,4]:
        T("%d-row of squares"%n,
          """
          (loop i %d
          (embed (loop k 4 (move 1d (/a 1a 4))))
          (move 1d 0a))
          """%n,
          needToTrain=n == 4)
    T("2x2 grid",
    """
    (for x 2 (embed (for y 2
       (embed (loop k 4 (move 1d (/a 1a 4))))
       (move 1d 0a)))
       (move 0d (/a 1a 4)) (move 1d (-a 0a (/a 1a 4))))
    """)
    T("slanted squares",
      """
      ((embed (loop k 4 (move 1d (/a 1a 4))))
      (move 0d (/a 1a 8))
      (loop k 4 (move 1d (/a 1a 4))))
      """)
    for l in range(1,6):
        T("square of size %d"%l,
          """
          (for i 4
          (move (*d 1d %d) (/a 1a 4)))
          """%l,
          needToTrain=l in range(4))
    for n in [5,7]:
        T("%d-concentric squares"%n,
          """
          (for i %d
          (embed (loop j 4 (move (*d 1d i) (/a 1a 4)))))
          """%n,
          needToTrain=n == 5)
    return tasks

def montageTasks(tasks, prefix="", columns=None, testTrain=False):
    import numpy as np
    
    w = 128
    arrays = [t.highresolution for t in tasks]
    for a in arrays:
        assert len(a) == w*w

    if testTrain:
        arrays = [a for a,t in zip(arrays, tasks) if t.mustTrain ] + [a for a,t in zip(arrays, tasks) if not t.mustTrain ]
        
    arrays = [np.array([a[i:i + w]
                        for i in range(0, len(a), w) ])
              for a in arrays]
    i = montage(arrays, columns=columns)

    import scipy.misc        
    scipy.misc.imsave('/tmp/%smontage.png'%prefix, i)
    if testTrain:
        trainingTasks = arrays[:sum(t.mustTrain for t in tasks)]
        testingTasks = arrays[sum(t.mustTrain for t in tasks):]
        random.shuffle(trainingTasks)
        random.shuffle(testingTasks)
        arrays = trainingTasks + testingTasks
    else:
        random.shuffle(arrays)
    scipy.misc.imsave('/tmp/%srandomMontage.png'%prefix, montage(arrays, columns=columns))

def demoLogoTasks():
    import scipy.misc
    import numpy as np

    g0 = Grammar.uniform(primitives, continuationType=turtle)
    eprint("dreaming into /tmp/dreams_0...")
    N = 1000
    programs = [ p
                     for _ in range(N)
                     for p in [g0.sample(arrow(turtle,turtle),
                                         maximumDepth=20)]
                     if p is not None]
    os.system("mkdir  -p /tmp/dreams_0")
    for n,p in enumerate(programs):
        with open(f"/tmp/dreams_0/{n}.dream","w") as handle:
            handle.write(str(p))
    drawLogo(*programs, pretty=True, smoothPretty=False,
             resolution=512,
             filenames=[f"/tmp/dreams_0/{n}_pretty.png"
                        for n in range(len(programs)) ],
             timeout=1)
    
    if len(sys.argv) > 1:
        tasks = makeTasks(sys.argv[1:],proto=False)
    else:
        tasks = makeTasks(['all'],proto=False)
    montageTasks(tasks,columns=16,testTrain=True)
    for n,t in enumerate(tasks):
        a = t.highresolution
        w = int(len(a)**0.5)
        scipy.misc.imsave('/tmp/logo%d.png'%n, np.array([a[i:i+w]
                                                         for i in range(0,len(a),w) ]))
        logo_safe_name = t.name.replace("=","_").replace(' ','_').replace('/','_').replace("-","_") + ".png"
        #os.system(f"convert /tmp/logo{n}.png -morphology Dilate Octagon /tmp/{logo_safe_name}")
        os.system(f"convert /tmp/logo{n}.png -channel RGB -negate /tmp/{logo_safe_name}")
    eprint(len(tasks),"tasks")
    eprint(sum(t.mustTrain for t in tasks),"need to be trained on")

    for t in dSLDemo():
        a = t.highresolution
        w = int(len(a)**0.5)
        scipy.misc.imsave('/tmp/logoDemo%s.png'%t.name, np.array([a[i:i+w]
                                                                  for i in range(0,len(a),w) ]))
        os.system(f"convert /tmp/logoDemo{t.name}.png -morphology Dilate Octagon /tmp/logoDemo{t.name}_dilated.png")

    tasks = [t for t in tasks if t.mustTrain ]
    random.shuffle(tasks)
    montageTasks(tasks[:16*3],"subset",columns=16)

    montageTasks(rotationalSymmetryDemo(),"rotational")

    
