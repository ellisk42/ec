
class FrontierEntry(object):
    def __init__(self, program, logPrior = None, logLikelihood = None, logPosterior = None):
        self.logPosterior = logPosterior
        self.program = program
        self.logPrior = logPrior
        self.logLikelihood = logLikelihood


class Frontier(object):
    def __init__(self, frontier, task = None):
        self.entries = frontier
        self.task = task

    def __iter__(self): return iter(self.entries)
        
    def removeZeroLikelihood(self):
        self.entries = [ e for e in self.entries if e.logLikelihood != float('-inf') ]
        return self

    def keepTopK(self,k):
        if k <= 0: return self
        self.entries.sort(key = lambda e: (e.logPrior + e.logLikelihood, str(e.program)), reverse = True)
        self.entries = self.entries[:k]
        return self

    @property
    def bestPosterior(self):
        return max(self.entries,key = lambda e: (e.logPrior + e.logLikelihood, str(e.program)))

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
                         [ "Average log prior %f"%averageLikelihood ])

        
        
