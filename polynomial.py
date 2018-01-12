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





MAXIMUMCOEFFICIENT = 5
tasks = [ RegressionTask("%dx^2 + %dx + %d"%(a,b,c),
                         arrow(tint,tint),
                         [(x,a*x*x + b*x + c) for x in range(6) ],
                         features = [float(a*x*x + b*x + c) for x in range(6) ])
          for a in range(MAXIMUMCOEFFICIENT)
          for b in range(MAXIMUMCOEFFICIENT)
          for c in range(MAXIMUMCOEFFICIENT) ]

if __name__ == "__main__":
    explorationCompression([addition, multiplication, k0,k1], tasks,
                           frontierSize = 10**4,
                           iterations = 5,
                           pseudoCounts = 10.0)
