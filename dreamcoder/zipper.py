try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module
from dreamcoder.type import *
from dreamcoder.program import *

from dreamcoder.grammar import ContextualGrammar
from collections import namedtuple
Subtree = namedtuple("subtree", ['path', 'tp', 'env', 'context'])

class HoleFinder:
    """
    Finds holes, enumerates a big list of subtree objects. I think it's all of them or something
    Want to change to:

    [X] subtree class
    [ ] lhs thing? - might be right not to chase down there ...
    [ ] return vs yield
    [ ] don't need
    """
    def __init__(self):
        """Fn yields (expression, loglikelihood) from a type and loss.
        Therefore, loss+loglikelihood is the distance from the original program."""
        pass
    def invented(self, e, tp, env, is_lhs=False, path=[]):
        yield from ()       
    def primitive(self, e, tp, env, is_lhs=False, path=[]):
        yield from ()
    def index(self, e, tp, env, is_lhs=False, path=[]):
        yield from ()
    def application(self, e, tp, env, is_lhs=False, path=[]):

        f_tp = arrow(e.x.infer(), tp) #this might be a problem

        yield from e.f.visit(self, f_tp, env, is_lhs=True, path=path+['f'])

        try:
            x_tp = inferArg(tp, e.f.infer())
        except:
            print(tp)
            print(e)
            assert False, "had an error within inferArg"

        yield from e.x.visit(self, x_tp, env, path=path+['x'])

    def abstraction(self, e, tp, env, is_lhs=False, path=[]):
        yield from e.body.visit(self, tp.arguments[1], [tp.arguments[0]]+env, path=path+['body'])

    def hole(self, e, tp, env, is_lhs=False, path=[]):
        #for now, no context:
        context = None
        yield Subtree(path, tp, env, context) #how do i get context?

    def execute(self, e, tp):
        return e.visit(self, tp, [], path=[])


def _findHolesEnum(path, request, sk, context=Context.EMPTY, environment=[]):
    """
    calculates mdl of full program 'full' from sketch 'sk'
    """
    if sk.isHole:
        # _, summary = self.likelihoodSummary(context, environment, request, full)
        # if summary is None:
        #     eprint(
        #         "FATAL: program [ %s ] does not have a likelihood summary." %
        #         full, "r = ", request, "\n", self)
        #     assert False

        yield Subtree(path, request, environment, context), context

    elif request.isArrow():
        assert sk.isAbstraction
        #assert sk.f == full.f #is this right? or do i need to recurse?
        v = request.arguments[0]
        yield from _findHolesEnum(path+['body'], request.arguments[1], sk.body, context=context, environment=[v] + environment)

    else:
        sk_f, sk_xs = sk.applicationParse()
        #full_f, full_xs = full.applicationParse()
        if sk_f.isIndex:
            #assert sk_f == full_f, "sketch and full program don't match on an index"
            ft = environment[sk_f.i].apply(context)
        elif sk_f.isInvented or sk_f.isPrimitive:
            #assert sk_f == full_f, "sketch and full program don't match on a primitive"
            context, ft = sk_f.tp.instantiate(context)
        elif sk_f.isAbstraction:
            print(sk_f)
            assert False, "sketch is not in beta longform"
        elif sk_f.isHole:
            assert False, "hole as function not yet supported"
        elif sk_f.isApplication:
            assert False, "should never happen - bug in applicationParse"
        else: assert False

        try: context = context.unify(ft.returns(), request)                
        except UnificationFailure: assert False, "sketch is ill-typed"
        ft = ft.apply(context)
        argumentRequests = ft.functionArguments()

        assert len(argumentRequests) == len(sk_xs) #this might not be true if holes??

        yield from findHolesApplication(path, context, environment,
                                          sk_f, sk_xs, argumentRequests)

def findHolesApplication(path, context, environment,
                      sk_function, sk_arguments, argumentRequests):
    if argumentRequests == []:
            #return torch.tensor([0.]).cuda(), context #does this make sense?
            yield None, context
    else:
        argRequest = argumentRequests[0].apply(context)
        laterRequests = argumentRequests[1:]

        sk_firstSketch = sk_arguments[0]
        sk_laterSketches = sk_arguments[1:]

        newPath = path + ['f']*(len(sk_arguments) - 1) #oy vey
        for subtree, newContext in _findHolesEnum(newPath + ['x'] , argRequest, sk_firstSketch,
                                                    context=context,
                                                    environment=environment):
            if subtree is not None:
                yield subtree, newContext

        #using the new context, hope it works

        sk_newFunction = Application(sk_function, sk_firstSketch)  # is this redundant? maybe 

        yield from findHolesApplication(path, newContext, environment, 
                                        sk_newFunction, sk_laterSketches, laterRequests)


def findHolesEnum(request, sk):
    # Find holes, enum style as opposed to visitor Style
    ret = []
    for subtree, context in _findHolesEnum([], request, sk, context=Context.EMPTY, environment=[]):
        if subtree is not None:
            ret.append(subtree)

    #last context
    lastContext = context

    return [Subtree(subtree.path, subtree.tp.apply(lastContext), 
            [tp.apply(lastContext) for tp in subtree.env], 
                lastContext) 
                    for subtree in ret]


#should be unused
class HolePuncher:
    """
    as written, given a path and expression, it does something
    """
    def __init__(self, path, expr):
        """Fn yields (expression, loglikelihood) from a type and loss.
        Therefore, loss+loglikelihood is the distance from the original program.
        todo:
        [ ] stupid path reversing is confusing
        [ ] tp inference thing?
        """
        self.history = []
        self.path = list(reversed(path))
        self.expr = expr

    def enclose(self, expr):
        for h in self.history[::-1]:
            expr = h(expr)
        return expr

    def invented(self, e):
        assert self.path == []
        return self.enclose(self.expr)

    def primitive(self, e):
        assert self.path == []
        return self.enclose(self.expr)

    def index(self, e):
        assert self.path == []
        return self.enclose(self.expr)

    def application(self, e):
        if self.path==[]:
            return self.enclose(self.expr)

        step = self.path.pop()
        if step == 'f':
            self.history.append(lambda expr: Application(expr, e.x))
            return e.f.visit(self)
        else:
            assert step == 'x'
            self.history.append(lambda expr: Application(e.f, expr))
            return e.x.visit(self)

    def abstraction(self, e):
        if self.path==[]:
            return self.enclose(self.expr)
        step = self.path.pop()
        assert step == 'body'
        self.history.append(lambda expr: Abstraction(expr))
        return e.body.visit(self)

    def hole(self, e):
        assert False, "you shouldn't be here, I think"

    def execute(self, e):
        return e.visit(self) #should return actual program

class NewExprPlacer:
    """
    as written, given a path and expression, it places a new expr in a hole
    """
    def __init__(self, allowReplaceApp=False, returnInnerObj=False):
        """
        todo:
        [ ] stupid path reversing is confusing
        [ ] tp inference thing?
        """
        self.history = []
        self.allowReplaceApp = allowReplaceApp
        self.returnInnerObj = returnInnerObj
        #self.expr = expr
    def enclose(self, expr):
        for h in self.history[::-1]:
            expr = h(expr)
        return expr
    def invented(self, e):
        assert False
        #assert self.path == []
        #return self.enclose(self.expr)
    def primitive(self, e):
        assert False
        #assert self.path == []
        #return self.enclose(self.expr)
    def index(self, e):
        assert False
        #assert self.path == []
        #return self.enclose(self.expr)
    def application(self, e):
        if self.path==[]:
            if self.allowReplaceApp:
                if self.returnInnerObj: self.innerObj = e
                return self.enclose(self.expr)   
            assert False
            #return self.enclose(self.expr)

        step = self.path.pop() #why am I popping?
        if step == 'f':
            self.history.append(lambda expr: Application(expr, e.x))
            return e.f.visit(self)
        else:
            assert step == 'x'
            self.history.append(lambda expr: Application(e.f, expr))
            return e.x.visit(self)

    def abstraction(self, e):
        if self.path==[]:
            assert False
            #return self.enclose(self.expr)
        step = self.path.pop()
        assert step == 'body'
        self.history.append(lambda expr: Abstraction(expr))
        return e.body.visit(self)

    def hole(self, e):
        assert self.path == []
        if self.returnInnerObj: self.innerObj = e
        return self.enclose(self.expr)

    def execute(self, sk, path, newSubtree):
        self.innerObj = None
        self.expr = newSubtree
        self.path = list(reversed(path))
        ret = sk.visit(self) #should return actual program
        self.path = []
        self.history = []
        if self.returnInnerObj: return ret, self.innerObj
        return ret


class OneStepFollower:
    """
    as written, given a path and expression, it does something
    """
    def __init__(self):
        """
        todo:
        [ ] stupid path reversing is confusing
        [ ] tp inference thing?
        """
        #self.history = []
        #self.expr = expr
    # def enclose(self, expr):
    #     for h in self.history[::-1]:
    #         expr = h(expr)
    #     return expr

    def invented(self, e, parentInfo):
        #assert False
        assert self.path == []
        self.prod = e
        return e, parentInfo
    def primitive(self, e, parentInfo):
        assert self.path == []
        self.prod = e
        return e, parentInfo
    def index(self, e, parentInfo):
        assert self.path == []
        self.prod = e
        return e, parentInfo

    def application(self, e, parentInfo):
        if self.path==[]:
            f, xs = e.applicationParse()
            x_tps = f.tp.functionArguments()
            self.prod = f
            returnVal = f
            for i, x in enumerate(xs):
                #x_tp = x.infer() #i think this is the problem?
                x_tp = x_tps[i]
                xHole = baseHoleOfType(x_tp)
                #xHole = g.sample(x_tp, sampleHoleProb=1.0) #need grammar for bad reason
                returnVal = Application(returnVal, xHole)

            return returnVal, parentInfo

        step = self.path.pop() #why am I popping?
        if step == 'f':
            #f, xs = e.applicationParse()
            #self.history.append(lambda expr: Application(expr, e.x))
            return e.f.visit(self, parentInfo)
        else:
            assert step == 'x'

            f, xs = e.applicationParse()
            parentInfo = (f, len(xs) - 1 )
            #self.history.append(lambda expr: Application(e.f, expr))
            return e.x.visit(self, parentInfo)

    def abstraction(self, e, parentInfo):
        if self.path==[]:
            ret = Abstraction(e.body.visit(self, parentInfo)[0])
            assert e.body.visit(self, parentInfo)[1] == parentInfo
            return ret, parentInfo

        step = self.path.pop()
        assert step == 'body'
        #self.history.append(lambda expr: Abstraction(expr))
        return e.body.visit(self, parentInfo)

    def hole(self, e, parentInfo):
        assert False, "you shouldnt be here"

    def execute(self, full, path):
        self.path = list(reversed(path))
        ret, parentInfo = full.visit(self, (None, None)) #should return actual program
        self.path = []

        if self.prod.isIndex:
            excludeProd = [self.prod.i]
        else:
            excludeProd = [self.prod]

        return ret, excludeProd, parentInfo


class HoleZipper():

    def __init__(self, path, context, env):
        self.path = path
        self.context = context
        self.env = env

    def add(choice):
        pass

class HoleVisitor:
    def __init__(self):
        pass

    def application(self, e):

        fromF = e.f.visit(self)
        for zp in fromF:
            zp.addpath(path + ['f'])

        for zp in fromX:
            zp.addpath(path + ['x'])

        return fromF + fromX #is the first part necessary?

    def index(self, e):
        return []

    def abstraction(self, e):
        # do i need to add the path?
        return e.body.visit(self)

    def primitive(self, e):
        return []

    def invented(self, e):
        return []

    def hole(self, e):
        context, env
        return [HoleZipper([], context, env)]


class ParentFinder:
    """
    as written, given a path and expression, it does something
    """
    def __init__(self):
        pass

    def invented(self, e, parentInfo):
        assert False, "you shouldnt be here"
        #assert False
        #assert self.path == []
        #self.prod = e
        #return parentInfo
    def primitive(self, e, parentInfo):
        assert False, "you shouldnt be here"
        #assert self.path == []
        #self.prod = e
        #return parentInfo
    def index(self, e, parentInfo):
        assert False, "you shouldnt be here"
        #assert self.path == []
        #self.prod = e
        #return parentInfo

    def application(self, e, parentInfo):
        if self.path==[]:
            assert False, "you shouldnt be here"
        step = self.path.pop() #why am I popping?
        if step == 'f':
            return e.f.visit(self, parentInfo)
        else:
            assert step == 'x'
            f, xs = e.applicationParse()
            parentInfo = (f, len(xs) - 1 )
            #self.history.append(lambda expr: Application(e.f, expr))
            return e.x.visit(self, parentInfo)

    def abstraction(self, e, parentInfo):
        if self.path==[]:
            assert False, "you shouldnt be here"
            #return parentInfo
        step = self.path.pop()
        assert step == 'body'
        #self.history.append(lambda expr: Abstraction(expr))
        return e.body.visit(self, parentInfo)

    def hole(self, e, parentInfo):
        return parentInfo
        
    def execute(self, sketch, path):
        self.path = list(reversed(path))
        parentInfo = sketch.visit(self, (None, None)) #should return actual program
        self.path = []
        return parentInfo


def returnCandidates(zipper, sk, tp, g):

    if isinstance(g, ContextualGrammar):
        parent, parentIndex = ParentFinder().execute(sk, zipper.path)
        candidates = g._sampleOneStep(
                            parent, parentIndex,
                            zipper.context,
                            zipper.env,
                            zipper.tp,
                            mustBeLeaf=False, returnCandidates=True)

    else:
        candidates = g._sampleOneStep(zipper.tp, zipper.context, zipper.env, False, returnCandidates=True)

    return [p for l, t, p, k in candidates]

def sampleOneStepFromHole(zipper, sk, tp, g, maximumDepth, supplyDist=None):

    mustBeLeaf = len([ t for t in zipper.path if t != 'body' ] ) >= maximumDepth
    
    if isinstance(g, ContextualGrammar):
        parent, parentIndex = ParentFinder().execute(sk, zipper.path)
        newContext, newSubtree = g._sampleOneStep(
                            parent, parentIndex,
                            zipper.context,
                            zipper.env,
                            zipper.tp,
                            mustBeLeaf=mustBeLeaf,
                            supplyDist=supplyDist)

    else:
        newContext, newSubtree = g._sampleOneStep(zipper.tp, zipper.context, zipper.env, mustBeLeaf, supplyDist=supplyDist)

    newSk = NewExprPlacer().execute(sk, zipper.path, newSubtree)
    newZippers = findHoles(newSk, tp) #TODO type inference, redoing computation, can use newContext
    return newSk, newZippers

def enumSingleStep(g, sk, tp, holeZipper=None, maximumDepth=None, supplyDist=None):
    zipper = holeZipper
    mustBeLeaf = len([ t for t in zipper.path if t != 'body' ] ) >= maximumDepth

    if isinstance(g, ContextualGrammar):
        parent, parentIndex = ParentFinder().execute(sk, zipper.path)
        candidates = g._enumOneStep(
                            parent, parentIndex,
                            zipper.context,
                            zipper.env,
                            zipper.tp,
                            mustBeLeaf=mustBeLeaf,
                            supplyDist=supplyDist)

    else:
        candidates = g._enumOneStep(zipper.tp, zipper.context, zipper.env, mustBeLeaf, supplyDist=supplyDist)  

    
    for stepCost, newContext, newSubtree in candidates:
        newSk = NewExprPlacer().execute(sk, zipper.path, newSubtree)
        newZippers = findHoles(newSk, tp) #TODO type inference, redoing computation, can use newContext
        yield stepCost, newZippers, newSk


def followPathOneStep(zipper, last, full, tp):
    #implement with visitor
    nextNode, excludeProd, parentInfo = OneStepFollower().execute(full, zipper.path) #TODO
    newSk = NewExprPlacer().execute(last, zipper.path, nextNode)

    return newSk, excludeProd, parentInfo, nextNode

def sampleWrongOneStep(zipper, last, full, tp, g, excludeProd=[], parentInfo=None):
    """
    follow the full program down the zipper, and then sample a token that isn't the one in full
    excludePrim will have prims and have integers for indices
    """

    # if this is a zipper into a lambda then use the lambdas grammar
    if len(zipper.env) > 1:
        assert zipper.path[0] == 'body'
        assert zipper.path[1] != 'body'
        if g.g_lambdas is None: # backwards compatability. Careful it doesnt carry the max depth thru tho
            g.g_lambdas = Grammar.uniform(get_lambdas())
        g = g.g_lambdas

    if isinstance(g, ContextualGrammar):
        parent, parentIndex = parentInfo
        _, wrongSubtree = g._sampleOneStep(
                            parent, parentIndex,
                            zipper.context,
                            zipper.env,
                            zipper.tp,
                            excludeProd=excludeProd)

    else:
        _, wrongSubtree = g._sampleOneStep(zipper.tp,
                                    zipper.context,
                                    zipper.env, 
                                    excludeProd=excludeProd)

    negEx = NewExprPlacer().execute(last, zipper.path, wrongSubtree)
    return negEx

def baseHoleOfType(tp):
    if tp.isArrow():
        expr = baseHoleOfType(tp.arguments[1])
        return Abstraction(expr)
    return Hole(tp=tp)

######
#TOP LEVEL - do in terms of args 

def findHolesVisitor(sk, tp):
    """depricated"""
    zippers = list(HoleFinder().execute(sk, tp))
    return zippers

def findHoles(sk, tp):
    return findHolesEnum(tp, sk)

#will be a method of grammar
def sampleSingleStep(g, sk, tp, holeZippers=None, maximumDepth=4, ordering='random'):
    #if ordering != 'last':
    #    print(f"warning, ordering is {ordering} in sampleSingleStep")
    #choose hole to expandz
    if holeZippers is None: holeZippers = findHoles(sk, tp)

    if ordering == 'first':
        zipper = holeZippers[0]
    elif ordering == 'last':
        zipper = holeZippers[-1]
    elif ordering == 'random':
        zipper = random.choice(holeZippers)
    else:
        raise ValueError

    #some sort of sample visitor, walks down to the hole with a visitor (like sketchSample), then calls sample
    newSk, newZippers = sampleOneStepFromHole(zipper, sk, tp, g, maximumDepth)
    return newSk, newZippers

def getTracesFromProg(full, tp, g, onlyPos=False, returnNextNode=False, ordering='random'):
    #if ordering != 'last':
    #    print(f"warning, ordering is {ordering} in getTracesFromProg")

    last = baseHoleOfType(tp) #this sets it up with first hole

    trace = []
    negTrace = []
    holesToExpand = []
    zippers = findHoles(last, tp) #TODO
    targetNodes = []
    while zippers:
        if ordering == 'first':
            zipper = zippers[0]
        elif ordering == 'last':
            zipper = zippers[-1]
        elif ordering == 'random':
            zipper = random.choice(zippers)
        else:
            raise ValueError


        holesToExpand.append(zipper)

        newLast, excludeProd, parentInfo, nextNode = followPathOneStep(zipper, last, full, tp) #TODO
        trace.append(newLast)
        targetNodes.append(nextNode)  

        if not onlyPos:
            negLast = sampleWrongOneStep(zipper, last, full, tp, g, excludeProd=excludeProd, parentInfo=parentInfo) #TODO
            negTrace.append(negLast)

        last = newLast
        zippers = findHoles(last, tp)
    if returnNextNode:
        return [baseHoleOfType(tp)] + trace[:-1], negTrace, targetNodes, holesToExpand
    return trace, negTrace
