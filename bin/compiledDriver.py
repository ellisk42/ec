import sys
import time
import traceback
import pickle as pickle

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.utilities import eprint

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    
    start = time.time()
    request = pickle.load(sys.stdin.buffer)
    dt = time.time() - start
    if dt > 1:
        eprint(
            "(compiled driver warning: SLOW) Compiled driver unpacked the message in time",
            dt)

    response = (False, None)
    try:
        start = time.time()
        f = request["function"]
        result = f(*request["arguments"],
                   **request["keywordArguments"])
        response = (True, result)
    except Exception as e:
        eprint("Exception thrown in pypy process for %s:" % f.__name__)
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
    finally:
        start = time.time()
        pickle.dump(response, sys.stdout.buffer)
        dt = time.time() - start
        if dt > 1:
            eprint(
                "(compiled driver warning: SLOW) Compiled driver packed the message in time",
                dt)
