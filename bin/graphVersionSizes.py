import matplotlib.pyplot as plot
import math
import sys
import re

plot.figure()
A = len(sys.argv) - 1
assert A == 3
for a,fn in enumerate(sys.argv[1:]):
    print(fn)
    ss = []
    hs = []
    cs = []
    es = []
    with open(fn,"r") as handle:
        for l in handle:
            if not l.startswith("DATA"):
                continue
            try:
                size = re.search("size=([^\s]+)\s",l)[1]
                height = re.search("height=([^\s]+)\s",l)[1]
                compressed = re.search("\|vs\|=([^\s]+)\s",l)[1]
                expanded = re.search("\|\[vs\]\|=([^\s]+)",l)[1]
                print(l)
                print(size, height, compressed, expanded)
                ss.append(int(size))
                hs.append(int(height))
                cs.append(int(compressed))
                es.append(math.e**float(expanded))
            except:
                print("ERROR:")
                print(l)
                sys.exit(0)





    plot.subplot(A,3,1 + a*3)
    plot.title(" ")
    plot.scatter(ss,cs)
    #plot.gca().set_yscale('log')
    plot.xlabel("expression size")
    plot.ylabel("version space size")

    if False:
        plot.subplot(2,2,2)
        plot.title(" ")
        plot.scatter(hs,cs)
        #plot.gca().set_yscale('log')
        plot.xlabel("expression height")
        plot.ylabel("version space size")


    plot.subplot(A,3,2 + a*3)
    plot.scatter(cs,es)
    plot.gca().set_yscale('log')
    #plot.gca().set_xscale('log')
    plot.title(f"{a+1} refactoring steps")
    plot.xlabel("version space size")
    plot.ylabel("# refactorings")


    plot.subplot(A,3,3 + a*3)
    plot.title(" ")
    plot.scatter(ss,es)
    plot.gca().set_yscale('log')
    plot.xlabel("expression size")
    plot.ylabel("# refactorings")


plot.show()
