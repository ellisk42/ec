# from utilities import *


# from time import sleep


# def f(_):
#     sleep(2)

# parallelMap(4,f,[None]*4)
# print "Done with parallel map now I'm going to sleep for ten seconds"
# sleep(10)


def f(): return 1,2

class k():
    def __init__(self):
        self.j = 42

    def g(self):
        self.j,_ = f()

x = k()
print x.j
x.g()
print x.j
