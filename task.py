
class RegressionTask():
    def __init__(self, name, request, examples):
        self.request = request
        self.name = name
        self.examples = examples
    def __str__(self): return self.name
    def check(self,e):
        e = e.evaluate([])
        for x,y in self.examples:
            if e(x) != y: return False
        return True
    def logLikelihood(self,e):
        if self.check(e): return 0.0
        else: return float('-inf')
