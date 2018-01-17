from utilities import *


from time import sleep


def f(_):
    sleep(2)

parallelMap(4,f,[None]*4)
print "Done with parallel map now I'm going to sleep for ten seconds"
sleep(10)
