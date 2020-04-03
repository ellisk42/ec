#lambdaSquared.py
from valueHead import *




exampleGenRules = []

rejectionRules = []

class LambdaSquaredValueHead(BaseValueHead):
    def __init__(self):
        super(BaseValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    def computeValue(self, sketch, task):
        def recurse(sketch, exs):
    		for fn in rejectionRules:
    			if fn(sketch, exs):
    				return float('inf')

    		for fn in exampleGenRules:
    			newSketch, newExs = fn(sketch, exs)
    			if recurse(newSketch, newExs) == float('inf'):
    				return float('inf')

    		return 0.

        return recurse(sketch, task.examples)

    def valueLossFromFrontier(self, frontier, g):
        if self.use_cuda:
            return torch.tensor([0.]).cuda()
        else: 
            return torch.tensor([0.])
