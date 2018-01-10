
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
        self.entries.sort(key = lambda e: e.logPrior + e.logLikelihood,reverse = True)
        self.entries = self.entries[:k]
        return self

    def bestPosterior(self):
        return max(self.entries,key = lambda e: e.logPrior + e.logLikelihood)

    def empty(self): return self.entries == []

        
        
