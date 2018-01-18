import sys
import time
import traceback
import cPickle as pickle

from utilities import eprint


if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    start = time.time()
    request = pickle.load(sys.stdin)
    eprint("Compiled driver unpacked the message in time", time.time() - start)

    response = (False, None)
    try:
        start = time.time()
        f = request["function"]
        result = f(*request["arguments"],
                   **request["keywordArguments"])
        response = (True, result)
        eprint("Compiled driver executed response in time", time.time() - start)
    except Exception as e:
        eprint("Exception thrown in pypy process for %s:" % f.__name__)
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
    finally:
        start = time.time()
        pickle.dump(response, sys.stdout)
        eprint("Packed and sent return value in time", time.time() - start)
