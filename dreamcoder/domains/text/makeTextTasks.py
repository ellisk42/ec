from dreamcoder.task import *
from dreamcoder.type import *
from dreamcoder.utilities import *

import random


def lcs(u, v):
    # t[(n,m)] = length of longest common string ending at first
    # n elements of u & first m elements of v
    t = {}

    for n in range(len(u) + 1):
        for m in range(len(v) + 1):
            if m == 0 or n == 0:
                t[(n, m)] = 0
                continue

            if u[n - 1] == v[m - 1]:
                t[(n, m)] = 1 + t[(n - 1, m - 1)]
            else:
                t[(n, m)] = 0
    l, n, m = max((l, n, m) for (n, m), l in t.items())
    return u[n - l:n]


delimiters = ['.', ',', ' ', '(', ')', '-']
characters = [chr(ord('a') + j)
              for j in range(26)] + \
             [chr(ord('A') + j)
              for j in range(26)] + \
    [str(j) for j in range(10)] + \
    ['+']

WORDS = None


def randomDelimiter():
    return random.choice(delimiters)


def randomCharacter():
    return random.choice(characters)


def randomWord(minimum=1, predicate=None):
    global WORDS
    if WORDS is None:
        tasks, cheating = loadPBETasks()
        observations = {''.join(z)
                        for t in tasks
                        for xs, y in t.examples
                        for z in list(xs) + [y]}

        def splitMany(s, ds):
            if len(ds) == 0:
                return [s]
            d = ds[0]
            ds = ds[1:]
            s = [w
                 for z in s.split(d)
                 for w in splitMany(z, ds)
                 if len(w) > 0]
            return s

        WORDS = {w
                 for o in observations
                 for w in splitMany(o, delimiters)}
        WORDS = list(sorted(list(WORDS)))

    # a disproportionately large fraction of the words have length three
    # the purpose of this is to decrease the number of 3-length words we have
    while True:
        if random.random() > 0.7:
            candidate = random.choice([w for w in WORDS if len(w) >= minimum])
        else:
            candidate = random.choice(
                [w for w in WORDS if len(w) >= minimum and len(w) != 3])
        if predicate is None or predicate(candidate):
            return candidate


def randomWords(ds, minimum=1, lb=2, ub=4):
    words = [randomWord(minimum=minimum)
             for _ in range(random.choice(range(lb, ub+1)))]
    s = ""
    for j,w in enumerate(words):
        if j > 0:
            s += random.choice(ds)
        s += w
    return s


def makeTasks():
    import random
    random.seed(9)

    NUMBEROFEXAMPLES = 4

    problems = []

    def toList(s): return [c for c in s]
    # Converts strings into a list of characters depending on the type

    def preprocess(x):
        if isinstance(x, tuple):
            return tuple(preprocess(z) for z in x)
        if isinstance(x, list):
            return [preprocess(z) for z in x]
        if isinstance(x, str):
            return [c for c in x]
        if isinstance(x, bool):
            return x
        assert False

    def problem(n, examples, needToTrain=False):
        task = Task(n, guess_arrow_type(examples),
                    [(preprocess(x),
                      preprocess(y))
                     for x, y in examples])
        task.mustTrain = True
        problems.append(task)

    for d1, d2 in randomPermutation(crossProduct(delimiters, delimiters))[
            :len(delimiters) * 2]:
        if d1 != d2:
            problem("Replace '%s' w/ '%s'" % (d1, d2),
                    [((x,), x.replace(d1, d2))
                     for _ in range(NUMBEROFEXAMPLES)
                     for x in [randomWords(d1)]],
                    needToTrain=False)
    for d in delimiters:
        problem("drop first word delimited by '%s'" % d,
                [((x,), d.join(x.split(d)[1:]))
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWords(d)]],
                needToTrain=True)
        for n in [0, 1, -1]:
            problem("nth (n=%d) word delimited by '%s'" % (n, d),
                    [((x,), x.split(d)[n])
                     for _ in range(NUMBEROFEXAMPLES)
                     for x in [randomWords(d)]],
                    needToTrain=True)
    for d1 in delimiters:
        problem("Append two words delimited by '%s'" % (d1),
                [((x, y), x + d1 + y)
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord()]
                 for y in [randomWord()]],
                needToTrain=True)
    for d1, d2 in randomPermutation(
        crossProduct(
            delimiters, delimiters))[
            :len(delimiters)]:
        problem("Append two words delimited by '%s%s'" % (d1, d2),
                [((x, y), x + d1 + d2 + y)
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord()]
                 for y in [randomWord()]],
                needToTrain=True)
    for n in range(1, 6):
        problem("Drop last %d characters" % n,
                [((x,), x[:-n])
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord(minimum=n)]],
                needToTrain=True)
        if n > 1:
            problem("Take first %d characters" % n,
                    [((x,), x[:n])
                     for _ in range(NUMBEROFEXAMPLES)
                     for x in [randomWord(minimum=n)]],
                    needToTrain=True)
    for d1, d2 in randomPermutation(
        crossProduct(
            delimiters, delimiters))[
            :len(delimiters)]:
        problem("Extract word delimited by '%s' - '%s'" % (d1, d2),
                [((a + d1 + b + d2 + c + d + e,), b)
                 for _ in range(int(NUMBEROFEXAMPLES / 2))
                 for d in [d1, d2]
                 for a in [randomWord()]
                 for b in [randomWord()]
                 for c in [randomWord()]
                 for e in [randomWord()]],
                needToTrain=True)

    for n in range(len(delimiters)):
        problem("First letters of words (%s)" % ("I" * (1 + n)),
                [((x,), "".join(map(lambda z: z[0], x.split(' '))))
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWords(' ')]
                 ],
                needToTrain=True)
        
    for d in delimiters:
        problem("Take first character and append '%s'" % d,
                [((x,), x[0] + d)
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord()]],
                needToTrain=True)

    for n in range(len(delimiters)):
        problem("Abbreviate separate words (%s)" % ("I" * (n + 1)),
                [((x, y), "%s.%s." % (x[0], y[0]))
                 for _ in range(NUMBEROFEXAMPLES)
                 for y in [randomWord()]
                 for x in [randomWord()]])
        d = delimiters[n]
        problem("Abbreviate words separated by '%s'" % d,
                [((x + d + y,), "%s.%s." % (x[0], y[0]))
                 for _ in range(NUMBEROFEXAMPLES)
                 for y in [randomWord()]
                 for x in [randomWord()]])

    for n in range(len(delimiters)):
        problem("Append 2 strings (%s)" % ('I' * (n + 1)),
                [((x, y), x + y)
                 for _ in range(NUMBEROFEXAMPLES)
                 for y in [randomWord()]
                 for x in [randomWord()]],
                needToTrain=True)

    for n in range(len(delimiters)):
        w = randomWord(minimum=3)
        problem("Prepend '%s'" % w,
                [((x,), w + x)
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord()]])
        w = randomWord(minimum=3)
        problem("Append '%s'" % w,
                [((x,), x + w)
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord()]])
        w = randomWord(minimum=3)
        problem("Prepend '%s' to first word" % w,
                [((x + ' ' + y,), w + x)
                 for _ in range(NUMBEROFEXAMPLES)
                 for x in [randomWord()]
                 for y in [randomWord()]])

    for n in range(1,6):
        problem("parentheses around a single word (%s)"%('I'*n),
                [((w,),"(%s)"%w)
                 for _ in range(NUMBEROFEXAMPLES)
                 for w in [randomWord()] ])
    problem("parentheses around first word",
            [((w + " " + s,),"(%s)"%w)
             for _ in range(NUMBEROFEXAMPLES)
             for w in [randomWord()]
             for s in [randomWords(" ")] ])
    problem("parentheses around second word",
            [((s,), "(%s)"%(s.split(" ")[1]))
             for _ in range(NUMBEROFEXAMPLES)
             for s in [randomWords(" ")] ])

    allowed = [d for d in delimiters if d not in "()"]
    for d1,d2 in randomPermutation(crossProduct(allowed, allowed))[:len(delimiters)]:
        problem("parentheses around word delimited by '%s' & '%s'"%(d1,d2),
                [((prefix + d1 + word + d2 + suffix,),
                  prefix + d1 + '(' + word + ')' + d2 + suffix)
                 for _ in range(NUMBEROFEXAMPLES)
                 for prefix in [randomWords("", lb=0, ub=1)]
                 for suffix in [randomWords(allowed, ub=2, lb=1)]
                 for word in [randomWord()] ])

    for n in range(7):
        w = randomWord(minimum=3)
        problem("ensure suffix `%s`"%w,
                [ ((s + (w if f else ""),), s + w)
                  for _ in range(NUMBEROFEXAMPLES)
                  for s in [randomWords(" ")]
                  for f in [random.choice([True,False])] ])
        eprint(problems[-1])
            

    for p in problems:
        guessConstantStrings(p)

    return problems


def loadPBETasks(directory="PBE_Strings_Track"):
    """
    Processes sygus benchmarks into task objects
    For these benchmarks, all of the constant strings are given to us.
    In a sense this is cheating (nb: the production release of flashfill does something equivalent to this "cheating")
    Returns (tasksWithoutCheating, tasksWithCheating).
    NB: Results in paper are done without "cheating"
    """
    import os
    from sexpdata import loads, Symbol

    def findStrings(s):
        if isinstance(s, list):
            return [y
                    for x in s
                    for y in findStrings(x)]
        if isinstance(s, str):
            return [s]
        return []

    def explode(s):
        return [c for c in s]

    tasks = []
    cheatingTasks = []
    for f in os.listdir(directory):
        if not f.endswith('.sl'):
            continue
        with open(directory + "/" + f, "r") as handle:
            message = "(%s)" % (handle.read())

        expression = loads(message)

        constants = []
        name = f
        examples = []
        declarative = False
        for e in expression:
            if len(e) == 0:
                continue
            if e[0] == Symbol('constraint'):
                e = e[1]
                assert e[0] == Symbol('=')
                inputs = e[1]
                assert inputs[0] == Symbol('f')
                inputs = inputs[1:]
                output = e[2]
                examples.append((inputs, output))
            elif e[0] == Symbol('synth-fun'):
                if e[1] == Symbol('f'):
                    constants += findStrings(e)
                else:
                    declarative = True
                    break
        if declarative: continue
        
        examples = list({(tuple(xs), y) for xs, y in examples})

        task = Task(name, arrow(*[tstr] * (len(examples[0][0]) + 1)),
                    [(tuple(map(explode, xs)), explode(y))
                     for xs, y in examples])
        cheat = task

        tasks.append(task)
        cheatingTasks.append(cheat)

    for p in tasks:
        guessConstantStrings(p)
    return tasks, cheatingTasks


def guessConstantStrings(task):
    if task.request.returns() == tlist(tcharacter):
        examples = task.examples
        guesses = {}
        N = 10
        T = 2
        for n in range(min(N, len(examples))):
            for m in range(n + 1, min(N, len(examples))):
                y1 = examples[n][1]
                y2 = examples[m][1]
                l = ''.join(lcs(y1, y2))
                if len(l) > 2:
                    guesses[l] = guesses.get(l, 0) + 1

        task.stringConstants = [g for g, f in guesses.items()
                                if f >= T]
    else:
        task.stringConstants = []
                    

    task.BIC = 1.
    task.maxParameters = 1

    task.specialTask = ("stringConstant",
                        {"maxParameters": task.maxParameters,
                         "stringConstants": task.stringConstants})


if __name__ == "__main__":
    challenge, _ = loadPBETasks("data/sygus")

    tasks = makeTasks()
    print(len(tasks), "synthetic tasks")
    tasks = []
    for t in tasks + challenge:
        print(t.name)
        for xs, y in t.examples:
            xs = ['"' + "".join(x) + '"' for x in xs]
            y = "".join(y)
            print('f(%s) = "%s"' % (", ".join(xs), y))
        print("\t{%s}" % (t.stringConstants))
        print()
    assert False
    # def maximumLength(x):
    #     if isinstance(x,list):
    #         return max([len(x)] + map(maximumLength,x))
    #     return 1

    # print max(maximumLength(z) for t in tasks
    #     for (x,),y in t.examples
    #     for z in [x,y] )

    if len(sys.argv) > 1 and "json" in sys.argv[1]:
        import json
        tasks = makeTasks()
        obj = [t.as_json_dict() for t in tasks]
        json.dump(obj, sys.stdout)
    else:
        as_tex = len(sys.argv) > 1 and "tex" in sys.argv[1]
        for t in tasks:
            print(t.name)
            print(t.request)
            if as_tex:
                print("""\\begin{tabular}{ll}
                \\toprule Input&Output\\\\\\midrule
        %s
        \\\\\\bottomrule
        \\end{tabular}""" % (" \\\\\n ".join(x[0] + " & " + y for x, y in t.examples)))
            else:
                for x, y in t.examples:
                    print(x[0], '\t', y)
            print()
        print(len(tasks), "tasks")
