#policyHead


class BasePolicyHead(nn.Module):
	#this is the single step type
    def __init__(self):
        super(BasePolicyHead, self).__init__() #should have featureExtractor?

    def sampleSingleStep(self, sketch, task):
        assert False, "not implemented"

    def policyLossFromFrontier(frontier, g):
        assert False, "not implemented"



class NoExecutionModel(nn.Module):
    def __init__(self):
        super(NoExecutionModel, self).__init__() #should have featureExtractor?

        self.model = SyntaxCheckingRobustfill or whatever

    def sampleFullProg(self, task):
        assert False, "not implemented"

  		return model.sample(task)

    def policyLossFromFrontier(frontier, g):
    	

    	return 

        #assert False, "not implemented"
