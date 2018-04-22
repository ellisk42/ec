from task import *
from type import *

import random


delimiters = ['.',',',' ','(',')','-']

def randomDelimiter():
    return random.choice(delimiters)

def randomCharacter():
    return chr(ord(random.choice(['a','A'])) + random.choice(range(26)))
def randomWord():
    return "".join([randomCharacter() for _ in range(random.choice(range(3,6))) ])
def randomWhiteWord():
    # Word with white space interspersed
    w = "".join([randomCharacter() for _ in range(random.choice(range(4,7))) ])

    # Put up to 2 random spaces into the word
    numberOfSpaces = random.choice(range(3))
    for _ in range(numberOfSpaces):
        j = random.choice(range(1,len(w)))
        w = w[:j] + " " + w[j:]

    # Put up to 2 spaces onto the start and end
    while True:
        starting = random.choice(range(0,3))
        ending = random.choice(range(0,3))
        if starting > 0 or ending > 0:
            return " "*starting + w + " "*ending
def randomWhiteWords(d):
    assert d != " "
    return d.join(randomWhiteWord() for _ in range(random.choice(range(2,5))) )
def randomWords(d):
    return d.join([randomWord() for _ in range(random.choice(range(2,5))) ])

singleWordOperations = {"lowercase": lambda x: x.lower(),
                        "uppercase": lambda x: x.upper(),
                        "capitalize": lambda x: x.capitalize(),
                        "double": lambda x: x + x,
                        #"strip": lambda x: x.strip(),
                        "first character": lambda x: x[0],
                        #"first 2 characters": lambda x: x[:2],
                        "drop first character": lambda x: x[1:],
                        #"last character": lambda x: x[-1],
                        #"last two characters": lambda x: x[-2:]}
                        }
compatibleCompositions = {(case, character)
                          for case in ["lowercase","uppercase","double"]
                          for character in ["first character","first 2 characters",
                                            "drop first character","last character",
                                            "last two characters"] } | \
 {("capitalize", character)
  for character in ["first 2 characters","last two characters","double"]} | \
 {(character,"double")
  for character in ["drop first character","capitalize"] } | \
 {("double","capitalize"),
  ("first character", "drop first character"),
  ("first character", "last two characters"),
  ("first 2 characters", "drop first character"),
  ("drop first character", "first 2 characters"),
  ("drop first character","drop first character")
  }

def makeOldTasks():
    NUMBEROFEXAMPLES = 4
    problems = []
    def toList(s): return [c for c in s]
    # Converts strings into a list of characters depending on the type
    def preprocess(x,t):
        if t == tstr: return toList(x)
        if t.name == "list":
            return [preprocess(z, t.arguments[0]) for z in x]
        return x
        
    def problem(n, examples, needToTrain = False):
        inputType = guess_type([ x for x,y in examples ])
        outputType = guess_type([ y for x,y in examples])
        task = Task(n, arrow(inputType, outputType),
                    [((preprocess(x,inputType),),
                      preprocess(y,outputType))
                     for x,y in examples ])
        if needToTrain: task.mustTrain = True
        problems.append(task)
    problem("Map strip",
                [ (x, map(lambda z: z.strip(), x))
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [[randomWhiteWord() for _ in range(random.choice(range(1,5)))]]
                ], needToTrain = True)
    for d in delimiters:
        problem("Map "+"strip"+"after splitting on "+d,
            [ (x, map(lambda z: z.strip(),x.split(d)))
              for _ in range(NUMBEROFEXAMPLES)
              for x in [randomWords(d)]
            ])
        problem("Map "+"strip"+" and then join with "+d,
            [ (x, d.join(map(lambda z: z.strip(),x)))
              for _ in range(NUMBEROFEXAMPLES)
              for x in [[randomWord() for _ in range(random.choice(range(1,5)))]]
            ])

    for n,f in singleWordOperations.iteritems():
        problem("Map "+n,
                [ (x, map(f,x))
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [[randomWord() for _ in range(random.choice(range(1,5)))]]
                ], needToTrain = True)
        for d in delimiters:
            problem("Map "+n+"after splitting on "+d,
                [ (x, map(f,x.split(d)))
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [randomWords(d)]
                ])
            problem("Map "+n+" and then join with "+d,
                [ (x, d.join(map(f,x)))
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [[randomWord() for _ in range(random.choice(range(1,5)))]]
                ])
    
    for n,f in singleWordOperations.iteritems():
        importantOperations = {"double","capitalize","first character","drop first character"}
        for j in range(2 if n in importantOperations else 1):
            np = n
            if j > 0:
                np = n + " (%s)"%('I'*j)
            problem(np, [(x,f(x)) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()] ], needToTrain = True)
     
    problem("strip", [(x, x.strip())
                      for _ in range(NUMBEROFEXAMPLES)
                      for x in [randomWhiteWord()] ])
    for n,f in singleWordOperations.iteritems():
        problem(n+".strip", [(x,f(x.strip()))
                             for _ in range(NUMBEROFEXAMPLES)
                             for x in [randomWhiteWord()] ])
    [problem(n1 + "." + n2, 
             [(x,f1(f2(x))) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()] ],
             needToTrain = True)
     for n1,f1 in singleWordOperations.iteritems()
     for n2,f2 in singleWordOperations.iteritems()
     if (n1,n2) in compatibleCompositions
    ]
    [problem("Replace delimiter '%s' w/ '%s'"%(d1,d2),
             [(x,x.replace(d1,d2))
              for x in [randomWords(d1)] ])
     for d1 in delimiters
     for d2 in delimiters
     if d1 != d2]
    [problem("Delete delimiter '%s'"%d,
                 [(x,x.replace(d,""))
              for x in [randomWords(d)] ])
     for d in delimiters]
    [problem("Apply %s delimited by '%s' to input delimited by '%s'"%(n,d1,d2),
             [(x, d2.join(map(f,x.split(d1))))
              for _ in range(NUMBEROFEXAMPLES)
              for x in [randomWords(d1)] ])
     for n,f in singleWordOperations.iteritems()
     for d1 in delimiters
     for d2 in delimiters
     if d1 != d2 and \
     n not in ['lowercase','uppercase']]
    for d1 in delimiters:
        if d1 == ' ': continue
        for d2 in delimiters:
            problem("Apply strip delimited by '%s' to input delimited by '%s'"%(d1,d2),
                    [(x,d2.join(map(lambda z: z.strip(),x.split(d1))))
                     for _ in range(NUMBEROFEXAMPLES)
                     for x in [randomWhiteWords(d1)] ])
    [problem("Apply %s to input delimited by '%s'"%(n,d),
             [(x, "".join(map(f,x.split(d))))
              for _ in range(NUMBEROFEXAMPLES)
              for x in [randomWords(d)] ])
     for n,f in singleWordOperations.iteritems()
     for d in delimiters
     if n not in ['lowercase','uppercase']
     ]
    [problem("Extract prefix up to '%s' (exclusive)"%d,
                                     [(x,y)
                                      for _ in range(NUMBEROFEXAMPLES)
                                      for y in [randomWord()]
                                      for x in [y + d + randomWord()]
                                     ])
     for d in delimiters ]
    [problem("Extract prefix up to '%s' (inclusive)"%d,
                                     [(x,y)
                                      for _ in range(NUMBEROFEXAMPLES)
                                      for y in [randomWord() + d]
                                      for x in [y + d + randomWord()]
                                     ])
                      for d in delimiters ]
    [problem("Extract suffix up to '%s' (exclusive)"%d,
                                     [(x,y)
                                      for _ in range(NUMBEROFEXAMPLES)
                                      for y in [randomWord()]
                                      for x in [randomWord() + d + y]
                                     ])
                      for d in delimiters ]
    [problem("Extract suffix up to '%s' (inclusive)"%d,
                                     [(x,y)
                                      for _ in range(NUMBEROFEXAMPLES)
                                      for y in [randomWord() + d]
                                      for x in [randomWord() + d + y]
                                     ])
                      for d in delimiters ]
    [problem("Extract string delimited by '%s','%s'"%(d1,d2),
                                        [(x,y)
                                 for _ in range(NUMBEROFEXAMPLES)
                                 for y in [randomWord()]
                                 for x in [randomWord() + d1 + y + d2 + randomWord()]])
                        for d1 in delimiters
                        for d2 in delimiters]
    [problem("Extract string delimited by '%s' (inclusive),'%s'"%(d1,d2),
                                [(x,y)
                                 for _ in range(NUMBEROFEXAMPLES)
                                 for y in [d1 + randomWord() + d2]
                                 for x in [randomWord() + y + randomWord()]])
                        for d1 in delimiters
                        for d2 in delimiters]
    [problem("Extract string delimited by '%s' (inclusive),'%s' (inclusive)"%(d1,d2),
                                [(x,y)
                                 for _ in range(NUMBEROFEXAMPLES)
                                 for y in [d1 + randomWord()]
                                 for x in [randomWord() + y + d2 + randomWord()]])
                        for d1 in delimiters
                        for d2 in delimiters] 
    [problem("Apply %s to string delimited by '%s','%s'"%(n,d1,d2),
                                [(x,f(y))
                                 for _ in range(NUMBEROFEXAMPLES)
                                 for y in [randomWord()]
                                 for x in [randomWord() + d1 + y + d2 + randomWord()]])
                      for n,f in singleWordOperations.iteritems()
                        for d1 in delimiters
                        for d2 in delimiters]
    for d1 in delimiters:
        for d2 in delimiters:
            if d1 == ' ' or d2 == ' ': continue
            problem("Apply strip to string delimited by '%s','%s'"%(d1,d2),
                    [(x, y.strip())
                     for _ in range(NUMBEROFEXAMPLES)
                     for y in [randomWhiteWord()]
                     for x in [randomWord() + d1 + y + d2 + randomWord()] ])
    return problems

def makeTasks():
    NUMBEROFEXAMPLES = 4
    
    problems = []
    def toList(s): return [c for c in s]
    # Converts strings into a list of characters depending on the type
    def preprocess(x):
        if isinstance(x,tuple): return tuple( preprocess(z) for z in x)
        if isinstance(x,list): return [ preprocess(z) for z in x ]
        if isinstance(x,str): return [ c for c in x ]
        assert False

    def problem(n, examples, needToTrain = False):
        task = Task(n, guess_arrow_type(examples),
                    [(preprocess(x),
                      preprocess(y))
                     for x,y in examples ])
        if needToTrain: task.mustTrain = True
        problems.append(task)

    for d1 in delimiters:
        for d2 in delimiters:
            if d1 != d2:
                problem("Replace '%s' w/ '%s'"%(d1,d2),
                        [ ((x,), x.replace(d1,d2))
                          for _ in range(NUMBEROFEXAMPLES) 
                          for x in [randomWords(d1)] ],
                        needToTrain=False)
    for d in delimiters:
        problem("drop first were delimited by '%s'"%d,
                [ ((x,), d.join(x.split(d)[1:]))
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [randomWords(d)] ])
        for n in [0,1,-1]:
            problem("nth (n=%d) word delimited by '%s'"%(n,d),
                    [ ((x,), x.split(d)[n])
                      for _ in range(NUMBEROFEXAMPLES)
                      for x in [randomWords(d)] ],
                    needToTrain=True)
    for d1 in delimiters:
        problem("Append two words delimited by '%s'"%(d1),
                    [ ((x,y), x + d1 + y)
                      for _ in range(NUMBEROFEXAMPLES)
                      for x in [randomWord()]
                      for y in [randomWord()] ])
        for d2 in delimiters:
            problem("Append two words delimited by '%s%s'"%(d1,d2),
                    [ ((x,y), x + d1 + d2 + y)
                      for _ in range(NUMBEROFEXAMPLES)
                      for x in [randomWord()]
                      for y in [randomWord()] ])
    for n in xrange(1,4):
        problem("Drop last %d characters"%n,
                [ ((x,), x[:-n])
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [randomWord() + randomWord()] ])
    for d in delimiters:
        problem("Take first character and append '%s'"%d,
                [ ((x,), x[0] + d)
                  for _ in range(NUMBEROFEXAMPLES)
                  for x in [randomWord()] ])
    return problems



def loadPBETasks(directory="PBE_Strings_Track"):
    """
    Processes sygus benchmarks into task objects
    For these benchmarks, all of the constant strings are given to us.
    In a sense this is cheating (nb: the production release of flashfill does something equivalent to this "cheating")
    Returns (tasksWithoutCheating, tasksWithCheating)
    """
    import os
    from sexpdata import loads, Symbol

    def findStrings(s):
        if isinstance(s,list):
            return [y
                for x in s
                for y in findStrings(x) ]
        if isinstance(s,str):
            return [s]
        return []
    
    tasks = []
    cheatingTasks = []
    for f in os.listdir(directory):
        if not f.endswith('.sl'): continue
        with open(directory + "/" + f,"r") as handle: message = "(%s)"%(handle.read())
        expression = loads(message)

        constants = []
        name = f
        examples = []
        for e in expression:
            if len(e) == 0: continue
            if e[0] == Symbol('constraint'):
                e = e[1]
                assert e[0] == Symbol('=')
                inputs = e[1]
                assert inputs[0] == Symbol('f')
                inputs = inputs[1:]
                output = e[2]
                examples.append((inputs, output))
            elif e[0] == Symbol('synth-fun'):
                assert e[1] == Symbol('f')
                constants += findStrings(e)

        task = Task(name, arrow(*[tstr]*(len(examples[0][0])+1)),
                    [(tuple(xs),y)
                     for xs,y in examples ])
        cheat = Task(name + "_cheating", arrow(*[tstr]*(len(examples[0][0])+1+len(constants))),
                     [(tuple(constants + xs),y)
                     for xs,y in examples ])
        tasks.append(task)
        print name
        print "\n".join(map(str,examples[:3]))
        cheatingTasks.append(cheat)

    return tasks, cheatingTasks
    

if __name__ == "__main__":
    import sys
    loadPBETasks()

    tasks = makeTasks()
    for t in tasks: print t.describe()
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
            print t.name
            print t.request
            if as_tex:
                print """\\begin{tabular}{ll}
                \\toprule Input&Output\\\\\\midrule
        %s
        \\\\\\bottomrule
        \\end{tabular}"""%(" \\\\\n ".join( x[0] + " & " + y for x,y in t.examples ))
            else:
                for x,y in t.examples:
                    print x[0],'\t',y
            print
        print len(tasks),"tasks"
