from task import *
from type import *


import random



delimiters = ['@','.','<','>',',',' ']

def randomDelimiter():
    return random.choice(delimiters)

def randomCharacter():
    return chr(ord(random.choice(['a','A'])) + random.choice(range(26)))
def randomWord():
    return "".join([randomCharacter() for _ in range(random.choice(range(6)) + 4) ])
def randomWords(d):
    return d.join([randomWord() for _ in range(random.choice(range(6)) + 3) ])

singleWordOperations = {"lowercase": lambda x: x.lower(),
                        "uppercase": lambda x: x.upper(),
                        "capitalize": lambda x: x.capitalize(),
                        "first character": lambda x: x[0],
                        "first 2 characters": lambda x: x[:2],
                        "drop first character": lambda x: x[1:],
                        "last character": lambda x: x[-1],
                        "last two characters": lambda x: x[-2:]}

def makeTasks():
    singleWordProblems = [RegressionTask(n, arrow(tstring,tstring),
                                         [((x,),f(x)) for _ in range(5) for x in [randomWord()] ])
                          for n,f in singleWordOperations.iteritems() ]
    doubleWordProblems = [RegressionTask(n1 + "." + n2, arrow(tstring,tstring),
                                         [((x,),f1(f2(x))) for _ in range(5) for x in [randomWord()] ])
                          for n1,f1 in singleWordOperations.iteritems()
                          for n2,f2 in singleWordOperations.iteritems()
                          if ("character" in n1) != ("character" in n2) and n1 < n2]
    mapSingleProblems = [RegressionTask("Apply %s delimited by '%s' to input delimited by '%s'"%(n,d1,d2),
                                        arrow(tstring,tstring),
                                 [((x,),d2.join(map(f,x.split(d1))))
                                  for _ in range(5)
                                  for x in [randomWords(d1)] ])
                         for n,f in singleWordOperations.iteritems()
                         for d1 in delimiters
                         for d2 in delimiters
                         if d1 != d2]
    mapDoubleProblems = [RegressionTask("Apply %s.%s delimited by '%s' to input delimited by '%s'"%(n1,n2,d1,d2),
                                        arrow(tstring,tstring),
                                 [((x,),d2.join(map(f2,map(f1,x.split(d1)))))
                                  for _ in range(5)
                                  for x in [randomWords(d1)] ])
                         for n1,f1 in singleWordOperations.iteritems()
                         for n2,f2 in singleWordOperations.iteritems()
                         if ("character" in n1) != ("character" in n2) and n1 < n2
                         for d1 in delimiters
                         for d2 in delimiters ]
    extractPrefix1 = [RegressionTask("Extract prefix up to '%s' (exclusive)"%d,
                                     arrow(tstring,tstring),
                                     [((x,),y)
                                      for _ in range(5)
                                      for y in [randomWord()]
                                      for x in [y + d + randomWord()]
                                     ])
                      for d in delimiters ]
    extractPrefix2 = [RegressionTask("Extract prefix up to '%s' (inclusive)"%d,
                                     arrow(tstring,tstring),
                                     [((x,),y)
                                      for _ in range(5)
                                      for y in [randomWord() + d]
                                      for x in [y + d + randomWord()]
                                     ])
                      for d in delimiters ]
    extractSuffix1 = [RegressionTask("Extract suffix up to '%s' (exclusive)"%d,
                                     arrow(tstring,tstring),
                                     [((x,),y)
                                      for _ in range(5)
                                      for y in [randomWord()]
                                      for x in [randomWord() + d + y]
                                     ])
                      for d in delimiters ]
    extractSuffix2 = [RegressionTask("Extract suffix up to '%s' (inclusive)"%d,
                                     arrow(tstring,tstring),
                                     [((x,),y)
                                      for _ in range(5)
                                      for y in [randomWord() + d]
                                      for x in [randomWord() + d + y]
                                     ])
                      for d in delimiters ]
    extractDelimited1 = [RegressionTask("Extract string delimited by '%s','%s'"%(d1,d2),
                                        arrow(tstring,tstring),
                                        [((x,),y)
                                 for _ in range(5)
                                 for y in [randomWord()]
                                 for x in [randomWord() + d1 + y + d2 + randomWord()]])
                        for d1 in delimiters
                        for d2 in delimiters]
    extractDelimited2 = [RegressionTask("Extract string delimited by '%s' (inclusive),'%s'"%(d1,d2),
                                        arrow(tstring,tstring),
                                [((x,),y)
                                 for _ in range(5)
                                 for y in [d1 + randomWord() + d2]
                                 for x in [randomWord() + y + randomWord()]])
                        for d1 in delimiters
                        for d2 in delimiters]
    extractDelimited3 = [RegressionTask("Extract string delimited by '%s' (inclusive),'%s' (inclusive)"%(d1,d2),
                                        arrow(tstring,tstring),
                                [((x,),y)
                                 for _ in range(5)
                                 for y in [d1 + randomWord()]
                                 for x in [randomWord() + y + d2 + randomWord()]])
                        for d1 in delimiters
                        for d2 in delimiters] 
    applyDelimited = [RegressionTask("Apply %s to string delimited by '%s','%s'"%(n,d1,d2),
                                     arrow(tstring,tstring),
                                [((x,),f(y))
                                 for _ in range(5)
                                 for y in [randomWord()]
                                 for x in [randomWord() + d1 + y + d2 + randomWord()]])
                      for n,f in singleWordOperations.iteritems()
                        for d1 in delimiters
                        for d2 in delimiters]

    return singleWordProblems + doubleWordProblems + extractPrefix1 + extractPrefix2 + extractSuffix1 + extractSuffix2 + mapSingleProblems + mapDoubleProblems + extractDelimited1 + extractDelimited2 + applyDelimited


