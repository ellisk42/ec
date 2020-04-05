#lambdaSquared.py
from valueHead import *

exampleGenRules = []

rejectionRules = []

from Program 

def unzip(sk, n=0)
	if not sk.isAbstraction:
		return sk, n
	else:
		return unzip(sk.body, n+1)

def rezip(sk, n):
	if n==0:
		return sk
	else:
		return rezip(Application(sk), n-1)


def mapRejectionRule1(sk, exs):
	#returning true means rejection
	innerSk, n = unzip(sk)

	if innerSk.isApplication:
		f, args = innerSk.applicationParse()
		if f.isPrimitive and f.name == 'map' and not args[1].hasHoles:


			lst = rezip(args[1], n)

			for xs, y in exs:
				inp = lst.runWithArguments(xs)
				#rule detection:
				if len(y) != len(inp):
					return True

	return False


def mapNewExs(sk, exs):
	#TODO
	return newSk, newExs


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
