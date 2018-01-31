from task import *
from type import *


import random



delimiters = ['.',',',' ','<','>']

def randomDelimiter():
    return random.choice(delimiters)

def randomCharacter():
    return chr(ord(random.choice(['a','A'])) + random.choice(range(26)))
def randomWord():
    return "".join([randomCharacter() for _ in range(random.choice(range(3,6))) ])
def randomWords(d):
    return d.join([randomWord() for _ in range(random.choice(range(2,5))) ])

singleWordOperations = {"lowercase": lambda x: x.lower(),
                        "uppercase": lambda x: x.upper(),
                        "capitalize": lambda x: x.capitalize(),
                        "double": lambda x: x + x,
                        "first character": lambda x: x[0],
                        "first 2 characters": lambda x: x[:2],
                        "drop first character": lambda x: x[1:],
                        "last character": lambda x: x[-1],
                        "last two characters": lambda x: x[-2:]}
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
# for x,y in compatibleCompositions:
#     assert x in singleWordOperations
#     assert y in singleWordOperations




def makeTasks():
    NUMBEROFEXAMPLES = 4
    problems = []
    def problem(n, examples):
        problems.append(RegressionTask(n, arrow(tstr,tstr),
                                       [((x,),y) for x,y in examples ]))
    
    [problem(n, [(x,f(x)) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()] ])
     for n,f in singleWordOperations.iteritems() ]
    [problem(n1 + "." + n2, 
             [(x,f1(f2(x))) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()] ])
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
    return problems


if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        print t.name
        for x,y in t.examples:
            print x[0],'\t',y
        print
    print len(tasks),"tasks"
