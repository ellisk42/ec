import traceback
import dill
import os
import subprocess
import sys

from utilities import usingDill

if __name__ == "__main__":
    sys.setrecursionlimit(1000)
    
    [ra,wr] = sys.argv[1:]
    r = os.fdopen(int(ra),'rb')
    
    usingDill(True)
    message = dill.loads(r.read())
    usingDill(False)
    
    #module = __import__(message["module"])
    #module.__dict__[message["functionName"]]
    function = message["function"]
    try:
        returnValue = function(*message["arguments"],
                               **message["keywordArguments"])
        returnValue = (True,returnValue)
    except:
        returnValue = (False,traceback.format_exc())
    w = os.fdopen(int(wr),'wb')
    w.write(dill.dumps(returnValue))
    w.close()
