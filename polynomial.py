from ec import *


addition = Primitive("+",
                     arrow(tint,arrow(tint,tint)),
                     lambda x: lambda y: x + y)
multiplication = Primitive("*",
                           arrow(tint,arrow(tint,tint)),
                           lambda x: lambda y: x * y)
square = Primitive("square",
                   arrow(tint,tint),
                   lambda x: x*x)
k1 = Primitive("1",tint,1)
k0 = Primitive("0",tint,0)





g0 = Grammar.uniform([addition, multiplication, k0,k1])

            

tasks = [ RegressionTask("%dx^2 + %dx + %d"%(a,b,c),
                         arrow(tint,tint),
                         [(x,a*x*x + b*x + c) for x in range(6) ])
          for a in range(10)
          for b in range(10)
          for c in range(10) ]

frontiers = enumerateFrontiers(g0, 10**3, tasks)
frontiers = [ frontier.keepTopK(1) for frontier in frontiers ]

FragmentGrammar.induceFromFrontiers(g0, frontiers)

numberOfHitTasks = 0
for frontier in frontiers:
    if frontier.empty():
        #print "MISS",frontier.task.name
        pass
    else:
        print "HIT",frontier.task.name,"with",frontier.bestPosterior().program
        numberOfHitTasks += 1
print "Hit %d/%d tasks"%(numberOfHitTasks,len(tasks))


