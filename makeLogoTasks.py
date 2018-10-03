# coding: utf8

from task import Task
from type import arrow
from logoPrimitives import turtle
import png
import os
import sys
from program import *
from utilities import *

rootdir = "./data/logo/"


def fileToArray(fname):
    r = png.Reader(filename=fname)
    array = [[y for y in x[3::4]] for x in r.read()[2]]
    flatten = [item for sublist in array for item in sublist]
    return flatten


def pretty_string(shape, size):
    out = ""
    nl = "\n"
    out += "╭"
    out += "─" * (size * 2)
    out += "╮"
    out += nl
    for j in range(size):
        out += "│"
        for i in range(size):
            if int(shape[j * size + (i % size)]) < 51:
                out += "  "
            elif int(shape[j * size + (i % size)]) < 102:
                out += "░░"
            elif int(shape[j * size + (i % size)]) < 153.6:
                out += "▒▒"
            elif int(shape[j * size + (i % size)]) < 204.8:
                out += "▓▓"
            else:
                out += "██"
        out += "│"
        out += nl
    out += "╰"
    out += "─" * (size * 2)
    out += "╯"
    out += nl
    return out


def pretty_print(shape, size):
    print((pretty_string(shape, size)))


def allTasks():
    return next(os.walk(rootdir))[1]


def makeTasks(subfolders, proto):
    problems = []

    if subfolders == ['all']:
        subfolders = allTasks()

    def problem(n, examples, highresolution, needToTrain=False):
        outputType = arrow(turtle, turtle)
        task = Task(n,
                    outputType,
                    [([0], y) for _, y in examples])
        task.mustTrain = needToTrain
        task.proto = proto
        task.specialTask = ("LOGO", {"proto": proto})
        task.highresolution = highresolution
        problems.append(task)

    for subfolder in subfolders:
        for _, subf, _ in os.walk(rootdir + subfolder):
            for subfl in subf:
                for _, _, files in os.walk(rootdir + subfolder + "/" + subfl):
                    for f in files:
                        if f.endswith("_l.png"):
                            fullPath = rootdir + subfolder + "/" + subfl + '/' + f
                            eprint(fullPath)
                            img1 = fileToArray(fullPath)
                            highresolution = fileToArray(fullPath.replace("_l.png", "_h.png"))
                            try:
                                problem(subfolder+"/"+subfl,
                                        [([], img1)],
                                        highresolution,
                                        needToTrain=True)
                            except FileNotFoundError:
                                problem(subfolder+"_"+f,
                                        [([], img1)],
                                        highresolution,
                                        needToTrain=True)
    return manualLogoTasks() + problems

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
    _al = Program.parse("logo_ADDL")
    _sl = Program.parse("logo_SUBL")

    _pu = Program.parse("logo_PU")
    _pd = Program.parse("logo_PD")
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
        if k == Symbol("pu"): return Application(_pu, continuation)
        if k == Symbol("pd"): return Application(_pd, continuation)
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


def manualLogoTask(name, expression, proto=False, needToTrain=False):
    p = parseLogo(expression)
    from logoPrimitives import primitives
    from grammar import Grammar
    g = Grammar.uniform(primitives)
    try: g.logLikelihood(arrow(turtle,turtle),p)
    except: eprint("WARNING: could not calculate likelihood of manual logo",p)

    [output, highresolution] = \
            [subprocess.check_output(['./logoDrawString',
                                      '0',
                                      "none",
                                      str(resolution),
                                      str(p)],
                                     timeout=1).decode("utf8")
             for resolution in [28,128]]
    shape = list(map(int, output.split(',')))
    highresolution = list(map(float, highresolution.split(',')))
    t = Task(name, arrow(turtle,turtle),
             [(([0]), shape)])
    t.mustTrain = needToTrain
    t.proto = proto
    t.specialTask = ("LOGO", {"proto": proto})

    t.highresolution = highresolution

    return t

def manualLogoTasks():
    tasks = []
    def T(*a): tasks.append(manualLogoTask(*a))
    T("pu/pd",
      """((move 1l 0a) pu (move 1l epsilonAngle) pd (move 1l 0a))""")
    T("pd",
      """(pu (move 1l (/a 1a 2)) (move 1l epsilonAngle) pd (move (*d 1d 2) 0a))""")
    T("pd square",
      """(loop i 4 
      pu (move 1l 0a) pd (move 1l (/a 1a 4)))""")
    T("pd dashed",
      """
      (loop i infinity
      pu (move epsilonLength 0a) pd (move epsilonLength 0a))
      """)

    for n,l in [(3,"1l"),
                (5,"1l"),
                (6,"(*d 1d 2)"),
                (7,"1l"),
                (8,"(/d 1d 2)")]:
        T("%d-gon %s"%(n,l),
          """
          (loop i %d
          (move %s (/a 1a %d)))
          """%(n,l,n))

    T("upwards", "((move 0d (/a 1a 4)) (move 1d 0a))")
    T("right angle", "((move (*d 1d 2) (/a 1a 4)) (move 1d 0a))")
    T("right angle epsilon", "((move epsilonLength (/a 1a 4)) (move epsilonLength 0a))")
    

    for i in range(3,7):
        T("Greek spiral %d"%i,
          """
          (loop i %d
          (move (*l 1l i) (/a 1a 4)))
          """%i)
        
        T("smooth spiral %d"%i,
          """
          (loop i infinity 
          (move (*d epsilonLength i) (*a epsilonAngle %d)))
          """%i)

    for i in [3,5,7,9]:
        T("star %d"%i,
          """
          (loop i %d (move (*d 1d 3) (-a (/a 1a 2) (/a (/a 1a 2) %s))))
          """%(i,i))

    T("leaf iteration 1.1",
      """
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      """)
    T("leaf iteration 1.2",
      """
      ((move 0d (/a 1a 2))
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2))))
      """)
    T("leaf iteration 2.1",
      """
      (loop n 2
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      (move 0d (/a 1a 4)))
      """)
    T("leaf iteration 2.2",
      """
      ((move 0d (/a 1a 2))
      (loop n 2
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      (move 0d (/a 1a 4))))
      """)
    for n in [5,7]:
        T("flower %d"%n,
          """
          (loop j %d
          (loop n 2
          (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
          (move 0d (/a 1a 4)))
          (move 0d (/a 1a %d)))
          """%(n,n))
        

    for n in [3,5]:
        T("staircase %d"%n,
          """
          (loop i %d
          (move 1d (/a 1a 4))
          (move 1d (/a 1a 4))
          (move 0d (/a 1a 2)))
          """%n)

    for n in range(1,4):
        T("blocks zigzag %d"%n,
          """
          (loop i %d
          (move 1d (/a 1a 4)) (move 1d (/a 1a 4))
          (move 1d (+a (/a 1a 2) (/a 1a 4))) (move 1d (+a (/a 1a 2) (/a 1a 4))))
          """%n)
    for n in range(1,5):
        T("diagonal zigzag %d"%n,
          """
          ((move 0d (/a 1a 8))
          (loop i %d
          (move 1d (/a 1a 4)) 
          (move 1d (+a (/a 1a 2) (/a 1a 4)))))
          """%n)

    

    for n in range(1,7):
        if n%2 == 0:
            T("semicircle of size %d"%n,
              """
              (loop i infinity
              (move (*d epsilonLength %d) (-a 0a epsilonAngle)))
              """%n)
        else:
            T("semicircle of size %d"%n,
              """
              (loop i infinity
              (move (*d epsilonLength %d) epsilonAngle))
              """%n)
        T("circle of size %d"%n,
          """
          ((loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle)))
          """%(n,n))

    for n in [5]:
        T("%d enclosed circles"%n,
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle)))"""%n)

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
          (move (*d epsilonLength %d) epsilonAngle))))"""%(n,n,l,l))

    for n,l in [(3,1),(2,2),(1,3)]:
        T("%d-semicircle sequence L=%d"%(n,l),
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength %d) (-a 0a epsilonAngle))))
          """%(n,l,l))

    for n,l in [(3,"1d")]:
        T("row of %d circles"%n,
          """
          (loop j %d
          (embed (loop k 2 (loop i infinity (move epsilonLength epsilonAngle))))
          pu (move %s 0a) pd)"""%(n,l))
    for n,l in [(4,"1d")]:
        T("row of %d dashes"%n,
          """
          (loop j %d
          (embed (move 0d (/a 1a 4)) (move 1d 0a))
          pu (move %s 0a) pd)"""%(n,l))        
    for n,l in [(5,"1d")]:
        T("row of %d semicircles"%n,
          """
          (loop j %d
          (embed (loop i infinity (move epsilonLength epsilonAngle)))
          pu (move %s 0a) pd)"""%(n,l))

    for n in [3,5,6]:
        body = {"empty": "(move 1d 0a)",
                "dashed": "pu (move 1d 0a) pd (move 1d 0a)",
                "circle": "(move 1d 0a) (loop k 2 (loop i infinity (move epsilonLength epsilonAngle)))",
                "square": "(loop s 4 (move 1d (/a 1a 4)))",
                "semicircle": "(move 1d 0a) (loop i infinity (move epsilonLength epsilonAngle))"}
        for name in body:
            T("%d-%s snowflake"%(n,name),
              """
              (loop j %d
              (embed %s)
              (move 0d (/a 1a %d)))"""%(n,body[name],n))

    for n in [2,3,4]:
        T("%d-row of squares"%n,
          """
          (loop i %d
          (embed (loop k 4 (move 1d (/a 1a 4))))
          (move 1d 0a))
          """%n)
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
    for l in range(1,4):
        T("square of size %d"%l,
          """
          (for i 4
          (move (*d 1d %d) (/a 1a 4)))
          """%l)
    for n in [5]:
        T("%d-concentric squares"%n,
          """
          (for i %d
          (embed (loop j 4 (move (*d 1d i) (/a 1a 4)))))
          """%n)
    return tasks

def montageTasks(tasks):
    import numpy as np
    
    w = 128
    arrays = [t.highresolution for t in tasks]
    for a in arrays:
        assert len(a) == w*w

    arrays = [np.array([a[i:i + w]
                        for i in range(0, len(a), w) ])
              for a in arrays]
    i = montage(arrays)

    import scipy.misc
    scipy.misc.imsave('/tmp/montage.png', i)
    
    
    

if __name__ == "__main__":
    allTasks()
    if len(sys.argv) > 1:
        tasks = makeTasks(sys.argv[1:],proto=False)
    else:
        tasks = makeTasks(['all'],proto=False)
    montageTasks(tasks)
    eprint(len(tasks),"tasks")
