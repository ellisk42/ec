#test apis



    def computeValue(self, sketch, task):
        """
        compute value given sketch and features
        will definitely want to do good memoizing ... 
        but that's for later, just get functionality now
        [ ] compute features externally once
        """
        #modes: RNN, TREERNN, module net, graph net???

        features = self.featuresOfTask(task)
        sketchEncoding = self.sketchEncoder(sketch)
        
        if replStyle:
        	features = self.encodeOutput(task) #one for each
        	evaluationEncoding = self.abstractREPL(sketch, task)

        return self._distance(sketchEncoding, features)



    def abstractREPL(self, sketch, task):


    def trainAbstractREPL():


    def computeValueLoss(task, positive sketch, negative sketch):








class AbstractREPL(nn.Module):






"""
object for 

make a value head object
you can have different types of valueHeads
the api is:


def computeValue(self, sketch, task)

def computeLoss(self, posSketch, negSketch, task):
	


"""


