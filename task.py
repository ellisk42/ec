from utilities import *

import signal

class EvaluationTimeout(Exception): pass

EVALUATIONTABLE = {}


class RegressionTask(object):
    def __init__(self, name, request, examples, features = None, cache = True):
        self.cache = cache
        self.features = features
        self.request = request
        self.name = name
        self.examples = examples
    def __str__(self): return self.name
    def __eq__(self,o): return self.name == o.name
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.name)
    def check(self,e,timeout = None):
        if not (timeout is None):
            def timeoutCallBack(_1,_2): raise EvaluationTimeout()
            signal.signal(signal.SIGALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_PROF, timeout)
            
        try:
            f = e.evaluate([])
            
            for x,y in self.examples:
                if self.cache and (x,e) in EVALUATIONTABLE: p = EVALUATIONTABLE[(x,e)]
                else:
                    try: p = f(x)                    
                    except: p = None
                    if self.cache: EVALUATIONTABLE[(x,e)] = p
                if p != y:
                    if not (timeout is None): signal.setitimer(signal.ITIMER_PROF, 0)
                    return False

            if not (timeout is None): signal.setitimer(signal.ITIMER_PROF, 0)
            return True
        except EvaluationTimeout: return False
        
    def logLikelihood(self,e):
        if self.check(e): return 0.0
        else: return NEGATIVEINFINITY

class DifferentiableTask(RegressionTask):
    def __init__(self, name, request, examples, features = None):
        super(DifferentiableTask,self).__init__(name, request, examples, features, cache = cache)

    def logLikelihood(self,e):
        


