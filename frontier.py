from utilities import *

class FrontierEntry(object):
    def __init__(self, program, _ = None, logPrior = None, logLikelihood = None, logPosterior = None):
        self.logPosterior = logPrior + logLikelihood if logPosterior is None else logPosterior
        self.program = program
        self.logPrior = logPrior
        self.logLikelihood = logLikelihood
    def __repr__(self):
        return "FrontierEntry(program={self.program}, logPrior={self.logPrior}, logLikelihood={self.logLikelihood}".format(self=self)


class Frontier(object):
    def __init__(self, frontier, task):
        self.entries = frontier
        self.task = task

    def __repr__(self): return "Frontier(entries={self.entries}, task={self.task})".format(self=self)
    def __iter__(self): return iter(self.entries)
    def __len__(self): return len(self.entries)

    def normalize(self):
        z = lse([ e.logPrior + e.logLikelihood for e in self ])
        return Frontier([ FrontierEntry(program = e.program,
                                        logPrior = e.logPrior,
                                        logLikelihood = e.logLikelihood,
                                        logPosterior = e.logPrior + e.logLikelihood - z)
                          for e in self ],
                        self.task)
        
    def removeZeroLikelihood(self):
        self.entries = [ e for e in self.entries if e.logLikelihood != float('-inf') ]
        return self

    def topK(self,k):
        if k <= 0: return self
        newEntries = sorted(self.entries,
                            key = lambda e: (-e.logPosterior, str(e.program)))
        return Frontier(newEntries[:k], self.task)

    @property
    def bestPosterior(self):
        return min(self.entries,
                   key = lambda e: (-e.logPosterior, str(e.program)))

    @property
    def empty(self): return self.entries == []

    def summarize(self):
        if self.empty: return "MISS " + self.task.name
        best = self.bestPosterior
        return "HIT %s w/ %s ; log prior = %f ; log likelihood = %f"%(self.task.name, best.program, best.logPrior, best.logLikelihood)

    @staticmethod
    def describe(frontiers):
        numberOfHits = sum(not f.empty for f in frontiers)
        averageLikelihood = sum(f.bestPosterior.logPrior for f in frontiers if not f.empty) / numberOfHits
        return "\n".join([ f.summarize() for f in frontiers ] + \
                         [ "Hits %d/%d tasks"%(numberOfHits,len(frontiers))] + \
                         [ "Average description length of a program solving a task: %f nats"%(-averageLikelihood) ])

        
    def combine(self, other, tolerance = 0.001):
        '''Takes the union of the programs in each of the frontiers'''
        assert self.task == other.task
        
        x = {e.program: e for e in self }
        y = {e.program: e for e in other }
        programs = set(x.keys()) | set(y.keys())
        union = []
        for p in programs:
            if p in x:
                e1 = x[p]
                if p in y:
                    e2 = y[p]
                    assert abs(e1.logPrior - e2.logPrior) < tolerance
                    assert abs(e1.logLikelihood - e2.logLikelihood) < tolerance
            else:
                e1 = y[p]
            union.append(e1)

        return Frontier(union, self.task)
            
