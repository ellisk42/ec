#policyHead

"""
if useValue is on, then require a policyHead as well, and specify which type
PolicyHead

solver needs to know about policyHead

policy needs to know how to get g ?

policy is called exclusively at (train time) or by solver, so solver can shove itself in there for base policy ... 

recognition.py:
- [ ] init: should take policyHead to construct
- [ ] train: if useValue: policyHead.computePolicyLoss always
- [ ] inference: .cpu and .cuda policyHead

dreamcoder.py:
- [ ] constructor for policyHead
- [ ] singleValround thing

Astar.py
- [ ] rewrite so it calls the policyHead

SMC
- [ ] rewrite so it calls the policyHead


What to do about grammar we infer? leave it in ...



"""


class BasePolicyHead(nn.Module):
	#this is the single step type
    def __init__(self):
        super(BasePolicyHead, self).__init__() #should have featureExtractor?

    def sampleSingleStep(self, sketch, task):
        assert False, "not implemented"

    def policyLossFromFrontier(self, frontier, g):
        assert False, "not implemented"

    def enumSingleStep(g, sketch, request, 
                        holeZipper=zipper,
                        maximumDepth=self.maxDepth):
        pass

class NoExecutionModel(nn.Module):
    def __init__(self):
        super(NoExecutionModel, self).__init__() #should have featureExtractor?

        self.model = SyntaxCheckingRobustfill or whatever

    def sampleFullProg(self, task):
        assert False, "not implemented"

  		return model.sample(task)

    def policyLossFromFrontier(self, frontier, g):

    	return 

        #assert False, "not implemented"
