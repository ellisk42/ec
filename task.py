from utilities import *
from differentiation import *

import signal

class EvaluationTimeout(Exception): pass

EVALUATIONTABLE = {}


class RegressionTask(object):
    def __init__(self, name, request, examples, features = None, cache = True):
        '''request: the type of this task
        examples: list of tuples of (input, output). input should be a tuple, with one entry for each argument
        cache: should program evaluations be cached?
        features: list of floats.'''
        self.cache = cache
        self.features = features
        self.request = request
        self.name = name
        self.examples = examples
    def __str__(self): return self.name
    def __eq__(self,o): return self.name == o.name
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.name)
    def predict(self, f, x):
        for a in x: f = f(a)
        return f
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
                    try: p = self.predict(f,x)
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
    def __init__(self, name, request, examples, _ = None,
                 features = None, BIC = 1., loss = None, likelihoodThreshold = None):
        self.likelihoodThreshold = likelihoodThreshold
        assert loss != None
        self.loss = loss
        
        super(DifferentiableTask,self).__init__(name, request, examples, features, cache = False)
        self.BIC = BIC
    

    def logLikelihood(self,e):
        e, parameters = e.replacePlaceholders()
        f = e.evaluate([])
        
        loss = sum( self.loss(Placeholder.maybe(self.predict(f, [ Placeholder.named("X_",float(a))
                                                                  for a in x ])),
                              Placeholder.named("Y_",float(y)))
                    for x,y in self.examples )

        l = loss.resilientBackPropagation(parameters, lr = 0.5, steps = 100)
            
        # BIC penalty
        penalty = self.BIC*len(parameters)*math.log(len(self.examples))

        if self.likelihoodThreshold != None:
            if l > -self.likelihoodThreshold: return NEGATIVEINFINITY
            else: return 0.
        else:
            return -l - penalty
        

        
        


