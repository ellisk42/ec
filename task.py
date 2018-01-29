from program import *
from utilities import *
from differentiation import *

import random
import signal

class EvaluationTimeout(Exception): pass

EVALUATIONTABLE = {}


class RegressionTask(object):
    def __init__(self, name, request, examples, features = None, cache = False):
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
    def __repr__(self):
        return "RegressionTask(name={self.name}, request={self.request}, examples={self.examples}"\
            .format(self=self)
    def __eq__(self,o): return self.name == o.name
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.name)
    def predict(self, f, x):
        for a in x: f = f(a)
        return f
    def check(self,e,timeout = None):
        if timeout is not None:
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
                    if timeout is not None: signal.setitimer(signal.ITIMER_PROF, 0)
                    return False

            if timeout is not None: signal.setitimer(signal.ITIMER_PROF, 0)
            return True
        except EvaluationTimeout: return False
        
    def logLikelihood(self,e):
        if self.check(e): return 0.0
        else: return NEGATIVEINFINITY

    @staticmethod
    def standardizeTasks(tasks):
        """
        Mutates the features of the tasks so that they have mean zero and standard deviation 1
        Returns (means, standard deviations)
        """
        averages, deviations = RegressionTask.featureMeanAndStandardDeviation(tasks)
        for t in tasks:
            t.features = RegressionTask.standardizeFeatures(averages, deviations, t.features)
        return averages, deviations
    @staticmethod
    def featureMeanAndStandardDeviation(tasks):
        dimension = len(tasks[0].features)
        averages = [ sum(t.features[j] for t in tasks)/float(len(tasks))
                     for j in range(dimension) ]
        variances = [ sum( (t.features[j] - averages[j])**2 for t in tasks )/float(len(tasks))
                      for j in range(dimension) ]
        standardDeviations = [ v**0.5 for v in variances ]
        for j,s in enumerate(standardDeviations):
            if s == 0.:
                eprint("WARNING: Feature %d is always %f"%(j+1, averages[j]))
        return averages, standardDeviations
    @staticmethod
    def standardizeFeatures(averages, standardDeviations, features):
        return [ (f - a)/s if s > 0 else (f - a) for a,s,f in zip(averages, standardDeviations, features) ]

        

class DifferentiableTask(RegressionTask):
    def __init__(self, name, request, examples, _ = None,
                 features = None, BIC = 1., loss = None, likelihoodThreshold = None):
        assert loss is not None
        self.loss = loss
        self.BIC = BIC
        self.likelihoodThreshold = likelihoodThreshold
        
        super(DifferentiableTask,self).__init__(name, request, examples, features, cache = False)
        
    def logLikelihood(self,e):
        e, parameters = PlaceholderVisitor.execute(e)
        f = e.evaluate([])

        loss = sum( self.loss(self.predict(f, map(float,x)), float(y))
                    for x,y in self.examples ) / float(len(self.examples))
        if isinstance(loss, DN):
            loss = loss.resilientBackPropagation(parameters, lr = 0.05, steps = 50)
            
        # BIC penalty
        penalty = self.BIC*len(parameters)*math.log(len(self.examples))

        if self.likelihoodThreshold is not None:
            if loss > -self.likelihoodThreshold: return NEGATIVEINFINITY
            else: return -penalty
        else:
            return -loss - penalty
        
def squaredErrorLoss(prediction, target):
    d = prediction - target
    return d*d
def l1loss(prediction, target):
    return abs(prediction - target)

class PlaceholderVisitor(object):
    def __init__(self): self.parameters = []
    def primitive(self, e):
        if e.name == 'REAL':
            placeholder = Placeholder.named("REAL_", 0.)
            self.parameters.append(placeholder)
            return Primitive(e.name, e.tp, placeholder)
        return e
    def invented(self,e): return e.body.visit(self)
    def abstraction(self,e): return Abstraction(e.body.visit(self))
    def application(self,e):
        return Application(e.f.visit(self),e.x.visit(self))
    def index(self,e): return e

    @staticmethod
    def execute(e):
        v = PlaceholderVisitor()
        e = e.visit(v)
        return e, v.parameters
