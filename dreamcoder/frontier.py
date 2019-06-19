from dreamcoder.utilities import *
from dreamcoder.task import Task


class FrontierEntry(object):
    def __init__(
            self,
            program,
            _=None,
            logPrior=None,
            logLikelihood=None,
            logPosterior=None):
        self.logPosterior = logPrior + logLikelihood if logPosterior is None else logPosterior
        self.program = program
        self.logPrior = logPrior
        self.logLikelihood = logLikelihood

    def __repr__(self):
        return "FrontierEntry(program={self.program}, logPrior={self.logPrior}, logLikelihood={self.logLikelihood}".format(
            self=self)


class Frontier(object):
    def __init__(self, frontier, task):
        self.entries = frontier
        self.task = task

    def __repr__(
        self): return "Frontier(entries={self.entries}, task={self.task})".format(self=self)

    def __iter__(self): return iter(self.entries)

    def __len__(self): return len(self.entries)

    def json(self):
        return {"request": self.task.request.json(),
                "task": str(self.task),
                "programs": [{"program": str(e.program),
                              "logLikelihood": e.logLikelihood}
                             for e in self ]}

    DUMMYFRONTIERCOUNTER = 0

    @staticmethod
    def dummy(program, logLikelihood=0., logPrior=0., tp=None):
        """Creates a dummy frontier containing just this program"""
        if not tp:
            tp = program.infer().negateVariables()

        t = Task(
            "<dummy %d: %s>" %
            (Frontier.DUMMYFRONTIERCOUNTER,
             str(program)),
            tp,
            [])
        f = Frontier([FrontierEntry(program=program,
                                    logLikelihood=logLikelihood,
                                    logPrior=logPrior)],
                     task=t)
        Frontier.DUMMYFRONTIERCOUNTER += 1
        return f

    def marginalLikelihood(self):
        return lse([e.logPrior + e.logLikelihood for e in self])

    def temperature(self,T):
        """Divides prior by T"""
        return Frontier([ FrontierEntry(program=e.program,
                                        logPrior=e.logPrior/T,
                                        logLikelihood=e.logLikelihood)
                          for e in self],
                        task=self.task)
                                        

    def normalize(self):
        z = self.marginalLikelihood()
        newEntries = [
            FrontierEntry(
                program=e.program,
                logPrior=e.logPrior,
                logLikelihood=e.logLikelihood,
                logPosterior=e.logPrior +
                e.logLikelihood -
                z) for e in self]
        newEntries.sort(key=lambda e: e.logPosterior, reverse=True)
        return Frontier(newEntries,
                        self.task)

    def expectedProductionUses(self, g):
        """Returns a vector of the expected number of times each production was used"""
        import numpy as np

        this = g.rescoreFrontier(self).normalize()
        ps = list(sorted(g.primitives, key=str))
        features = np.zeros(len(ps))
        
        for j, p in enumerate(ps):
            for e in this:
                w = math.exp(e.logPosterior)
                features[j] += w * sum(child == p
                                       for _, child in e.program.walk() )
            if not p.isInvented: features[j] *= 0.3
        return features
            

    def removeZeroLikelihood(self):
        self.entries = [
            e for e in self.entries if e.logLikelihood != float('-inf')]
        return self

    def topK(self, k):
        if k == 0: return Frontier([], self.task)
        if k < 0: return self            
        newEntries = sorted(self.entries,
                            key=lambda e: (-e.logPosterior, str(e.program)))
        return Frontier(newEntries[:k], self.task)

    def sample(self):
        """Samples an entry from a frontier"""
        return sampleLogDistribution([(e.logLikelihood + e.logPrior, e)
                                      for e in self])

    @property
    def bestPosterior(self):
        return min(self.entries,
                   key=lambda e: (-e.logPosterior, str(e.program)))

    def replaceWithSupervised(self, g):
        assert self.task.supervision is not None
        return g.rescoreFrontier(Frontier([FrontierEntry(self.task.supervision,
                                                         logLikelihood=0., logPrior=0.)],
                                          task=self.task))

    @property
    def bestll(self):
        best = max(self.entries,
                   key=lambda e: e.logLikelihood)
        return best.logLikelihood


    @property
    def empty(self): return self.entries == []

    @staticmethod
    def makeEmpty(task):
        return Frontier([], task=task)

    def summarize(self):
        if self.empty:
            return "MISS " + self.task.name
        best = self.bestPosterior
        return "HIT %s w/ %s ; log prior = %f ; log likelihood = %f" % (
            self.task.name, best.program, best.logPrior, best.logLikelihood)

    def summarizeFull(self):
        if self.empty:
            return "MISS " + self.task.name
        return "\n".join([self.task.name] +
                         ["%f\t%s" % (e.logPosterior, e.program)
                          for e in self.normalize()])

    @staticmethod
    def describe(frontiers):
        numberOfHits = sum(not f.empty for f in frontiers)
        if numberOfHits > 0:
            averageLikelihood = sum(
                f.bestPosterior.logPrior for f in frontiers if not f.empty) / numberOfHits
        else:
            averageLikelihood = 0
        return "\n".join([f.summarize() for f in frontiers] +
                         ["Hits %d/%d tasks" % (numberOfHits, len(frontiers))] +
                         ["Average description length of a program solving a task: %f nats" % (-averageLikelihood)])

    def combine(self, other, tolerance=0.01):
        '''Takes the union of the programs in each of the frontiers'''
        assert self.task == other.task

        foundDifference = False

        x = {e.program: e for e in self}
        y = {e.program: e for e in other}
        programs = set(x.keys()) | set(y.keys())
        union = []
        for p in programs:
            if p in x:
                e1 = x[p]
                if p in y:
                    e2 = y[p]
                    if abs(e1.logPrior - e2.logPrior) > tolerance:
                        eprint(
                            "WARNING: Log priors differed during frontier combining: %f vs %f" %
                            (e1.logPrior, e2.logPrior))
                        eprint("WARNING: \tThe program is", p)
                        eprint()
                    if abs(e1.logLikelihood - e2.logLikelihood) > tolerance:
                        foundDifference = True
                        eprint(
                            "WARNING: Log likelihoods deferred for %s: %f & %f" %
                            (p, e1.logLikelihood, e2.logLikelihood))
                        if hasattr(self.task, 'BIC'):
                            eprint("\t%d examples, BIC=%f, parameterPenalty=%f, n parameters=%d, correct likelihood=%f" %
                                   (len(self.task.examples),
                                    self.task.BIC,
                                    self.task.BIC * math.log(len(self.task.examples)),
                                    substringOccurrences("REAL", str(p)),
                                    substringOccurrences("REAL", str(p)) * self.task.BIC * math.log(len(self.task.examples))))
                            e1.logLikelihood = - \
                                substringOccurrences("REAL", str(p)) * self.task.BIC * math.log(len(self.task.examples))
                            e2.logLikelihood = e1.logLikelihood

                        e1 = FrontierEntry(
                            program=e1.program,
                            logLikelihood=(
                                e1.logLikelihood +
                                e2.logLikelihood) /
                            2,
                            logPrior=e1.logPrior)
            else:
                e1 = y[p]
            union.append(e1)

        if foundDifference:
            eprint(
                "WARNING: Log likelihoods differed for the same program on the task %s.\n" %
                (self.task.name),
                "\tThis is acceptable only if the likelihood model is stochastic. Took the geometric mean of the likelihoods.")

        return Frontier(union, self.task)
