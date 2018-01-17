import time
import traceback
import cPickle as pickle
import os
import subprocess
import sys

def flushEverything():
    sys.stdout.flush()
    sys.stdin.flush()

if __name__ == "__main__":
    sys.setrecursionlimit(1000)
    
    [ra,wr] = sys.argv[1:]
    r = os.fdopen(int(ra),'rb')

    start = time.time()
    #usingDill(True)
    message = pickle.loads(r.read())
    #usingDill(False)
    print "Compiled driver unpacked the message in time",time.time() - start
    
    #module = __import__(message["module"])
    #module.__dict__[message["functionName"]]
    function = message["function"]
    try:
        returnValue = function(*message["arguments"],
                               **message["keywordArguments"])
        returnValue = (True,returnValue)
    except:
        returnValue = (False,traceback.format_exc())
    start = time.time()
    returnValue = pickle.dumps(returnValue)
    print "Packed return value in time",time.time() - start
    w = os.fdopen(int(wr),'wb')
    start = time.time()
    w.write(returnValue)
    w.close()
    print "Sent return value in time",time.time() - start
    flushEverything()
    
