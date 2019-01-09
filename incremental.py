
class SLC():
    def __init__(self, request, lines):
        self.request = request
        self.returnType = request.returns()
        self.arguments = request.functionArguments()
        self.lines = lines

    def toProgram(self, finalLine):
        lineToExpression = []
        # push the arguments into the environment
        environment = [Index(j) for j in range(len(self.arguments))]

        for l,e in enumerate(self.lines):
            for i,v in enumerate(environment):
                e = e.substitute(Index(i), v)
            lineToExpression.append(l)
            environment.insert(0, e)

        return lineToExpression[finalLine]

    
