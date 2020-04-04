from dreamcoder.program import *
from dreamcoder.differentiation import *

import signal


class EvaluationTimeout(Exception):
    pass


EVALUATIONTABLE = {}


class Task(object):
    def __init__(self, name, request, examples, features=None, cache=False):
        '''request: the type of this task
        examples: list of tuples of (input, output). input should be a tuple, with one entry for each argument
        cache: should program evaluations be cached?
        features: list of floats.'''
        self.cache = cache
        self.features = features
        self.request = request
        self.name = name
        self.examples = examples
        if len(self.examples) > 0:
            assert all(len(xs) == len(examples[0][0])
                       for xs, _ in examples), \
                "(for task %s) FATAL: Number of arguments varies." % name

    def __str__(self):
        if self.supervision is None:
            return self.name
        else:
            return self.name + " (%s)"%self.supervision

    def __repr__(self):
        return "Task(name={self.name}, request={self.request}, examples={self.examples}"\
            .format(self=self)

    def __eq__(self, o): return self.name == o.name

    def __ne__(self, o): return not (self == o)

    def __hash__(self): return hash(self.name)

    def describe(self):
        description = ["%s : %s" % (self.name, self.request)]
        for xs, y in self.examples:
            if len(xs) == 1:
                description.append("f(%s) = %s" % (xs[0], y))
            else:
                description.append("f%s = %s" % (xs, y))
        return "\n".join(description)

    def predict(self, f, x):
        for a in x:
            f = f(a)
        return f

    @property
    def supervision(self):
        if not hasattr(self, 'supervisedSolution'): return None
        return self.supervisedSolution

    def check(self, e, timeout=None):
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        try:
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

            try:
                f = e.evaluate([])
            except IndexError:
                # free variable
                return False
            except Exception as e:
                eprint("Exception during evaluation:", e)
                return False

            for x, y in self.examples:
                if self.cache and (x, e) in EVALUATIONTABLE:
                    p = EVALUATIONTABLE[(x, e)]
                else:
                    try:
                        p = self.predict(f, x)
                    except BaseException:
                        p = None
                    if self.cache:
                        EVALUATIONTABLE[(x, e)] = p
                if p != y:
                    if timeout is not None:
                        signal.signal(signal.SIGVTALRM, lambda *_: None)
                        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
                    return False

            return True
        # except e:
            # eprint(e)
            # assert(False)
        except EvaluationTimeout:
            eprint("Timed out while evaluating", e)
            return False
        finally:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)

    def logLikelihood(self, e, timeout=None):
        if self.check(e, timeout):
            return 0.0
        else:
            return NEGATIVEINFINITY

    @staticmethod
    def featureMeanAndStandardDeviation(tasks):
        dimension = len(tasks[0].features)
        averages = [sum(t.features[j] for t in tasks) / float(len(tasks))
                    for j in range(dimension)]
        variances = [sum((t.features[j] -
                          averages[j])**2 for t in tasks) /
                     float(len(tasks)) for j in range(dimension)]
        standardDeviations = [v**0.5 for v in variances]
        for j, s in enumerate(standardDeviations):
            if s == 0.:
                eprint(
                    "WARNING: Feature %d is always %f" %
                    (j + 1, averages[j]))
        return averages, standardDeviations

    def as_json_dict(self):
        return {
            "name": self.name,
            "request": str(self.request),
            "examples": [{"inputs": x, "output": y} for x, y in self.examples]
        }


class DifferentiableTask(Task):

    def __init__(self, name, request, examples, _=None,
                 features=None, BIC=1., loss=None, likelihoodThreshold=None,
                 steps=50, restarts=300, lr=0.5, decay=0.5, grow=1.2, actualParameters=None,
                 temperature=1., maxParameters=None, clipLoss=None, clipOutput=None):
        assert loss is not None
        self.temperature = temperature
        self.actualParameters = actualParameters
        self.maxParameters = maxParameters
        self.loss = loss
        self.BIC = BIC
        self.likelihoodThreshold = likelihoodThreshold

        arguments = {"parameterPenalty": BIC * math.log(len(examples)),
                     "temperature": temperature,
                     "steps": steps, "restarts": restarts, "lr": lr, "decay": decay, "grow": grow,
                     "maxParameters": maxParameters,
                     "lossThreshold": -likelihoodThreshold}
        if clipLoss is not None: arguments['clipLoss'] = float(clipLoss)
        if clipOutput is not None: arguments['clipOutput'] = float(clipOutput)
        if actualParameters is not None: arguments['actualParameters'] = int(actualParameters)
        
        self.specialTask = ("differentiable",
                            arguments)

        super(
            DifferentiableTask,
            self).__init__(
            name,
            request,
            examples,
            features,
            cache=False)

    def logLikelihood(self, e, timeout=None):
        assert timeout is None, "timeout not implemented for differentiable tasks, but not for any good reason."
        e, parameters = PlaceholderVisitor.execute(e)
        if self.maxParameters is not None and len(
                parameters) > self.maxParameters:
            return NEGATIVEINFINITY
        if self.actualParameters is not None and len(
                parameters) > self.actualParameters:
            return NEGATIVEINFINITY
        f = e.evaluate([])

        loss = sum(self.loss(self.predict(f, xs), y)
                   for xs, y in self.examples) / float(len(self.examples))
        if isinstance(loss, DN):
            try:
                loss = loss.restartingOptimize(
                    parameters,
                    lr=self.specialTask[1]["lr"],
                    steps=self.specialTask[1]["steps"],
                    decay=self.specialTask[1]["decay"],
                    grow=self.specialTask[1]["grow"],
                    attempts=self.specialTask[1]["restarts"],
                    update=None)
            except InvalidLoss:
                loss = POSITIVEINFINITY

        # BIC penalty
        penalty = self.BIC * len(parameters) * math.log(len(self.examples))

        if self.likelihoodThreshold is not None:
            if loss > -self.likelihoodThreshold:
                return NEGATIVEINFINITY
            else:
                return -penalty
        else:
            return -loss / self.temperature - penalty


def squaredErrorLoss(prediction, target):
    d = prediction - target
    return d * d


def l1loss(prediction, target):
    return abs(prediction - target)


class PlaceholderVisitor(object):
    def __init__(self): self.parameters = []

    def primitive(self, e):
        if e.name == 'REAL':
            placeholder = Placeholder.named("REAL_", random.random())
            self.parameters.append(placeholder)
            return Primitive(e.name, e.tp, placeholder)
        return e

    def invented(self, e): return e.body.visit(self)

    def abstraction(self, e): return Abstraction(e.body.visit(self))

    def application(self, e):
        return Application(e.f.visit(self), e.x.visit(self))

    def index(self, e): return e

    @staticmethod
    def execute(e):
        v = PlaceholderVisitor()
        e = e.visit(v)
        return e, v.parameters
