try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *

from dreamcoder.domains.tower.towerPrimitives import *
import time
import itertools

bootstrapTarget()
g = Grammar.uniform([k0,k1,addition, subtraction, Program.parse('cons'), 
    Program.parse('car'), Program.parse('cdr'), Program.parse('empty'),
    Program.parse('fold'), Program.parse('empty?')])

#g = g.randomWeights(lambda *a: random.random())
#p = Program.parse("(lambda (+ 1 $0))")
request = arrow(tint,tint)

def testEnumWithHoles():
    i = 0
    for ll,_,p in g.enumeration(Context.EMPTY,[],request,
                               6.,
                               enumerateHoles=True):

        i+=1
        if i==4:
            print("ref sketch", p)
            break

    for ll,_,pp in g.sketchEnumeration(Context.EMPTY,[],request, p,
                               6.,
                               enumerateHoles=True):
        print(pp)

        #ll_ = g.logLikelihood(request,p)
        #print(ll,p,ll_)
        #d = abs(ll - ll_)
        #assert d < 0.0001

def testSampleWithHoles():

    for _ in range(1000):
        p = g.sample(request, maximumDepth=6, maxAttempts=None, sampleHoleProb=.2)
        print(p, p.hasHoles)

        print("a sample from the sketch")
        print("\t", g.sampleFromSketch( request, p, sampleHoleProb=.2))

def findError():
    p = Program.parse('(lambda (fold <HOLE> 0 (lambda (lambda (+ $1 $0)) )))')
    for i in range(100):
        print("\t", g.sampleFromSketch( request, p, sampleHoleProb=.2))

def test_training_stuff():
    p = g.sample(request)
    print(p)

    for x in sketchesFromProgram(p, request, g):
        print(x)

def test_getTrace():
    full = g.sample(request)
    print("full:")
    print('\t', full)
    trace, negTrace= getTracesFromProg(full, request, g, onlyPos=False)
    print("trace:", *trace, sep='\n\t')
    #assert [full == p.betaNormalForm() trace]
    print("negTrace", *negTrace, sep='\n\t')

    assert trace[-1] == full
    return full


def test_sampleOneStep():
    for i in range(100):
        subtree = g._sampleOneStep(request, Context.EMPTY, [])
        print(subtree)

def test_holeFinder():

    #p = Program.parse('(lambda (fold <HOLE> 0 (lambda (lambda (+ $1 $0)) )))')
    for i in range(100):
        expr = g.sample( request, sampleHoleProb=.2)
        


    expr = Program.parse('(empty? (cons <HOLE> <HOLE>))')
    print("\t", expr )

    print([(subtree.path, subtree.tp, subtree.env) for subtree in findHolesEnum(tbool, expr)])
    #print([(subtree.path, subtree.env) for subtree in findHoles(expr, request)])

def test_sampleSingleStep():

    nxt = baseHoleOfType(request)
    zippers = findHoles(nxt, request)
    print(nxt)

    while zippers:
        nxt, zippers = sampleSingleStep(g, nxt, request, holeZippers=zippers)
        print(nxt)

def test_SingleStep1():

    #request = tbool
    #expr = Program.parse('(empty? (cons <HOLE> <HOLE>))')
    expr = Program.parse('(lambda (fold <HOLE> 0 (lambda (lambda (+ $1 $0)) )))')

    nxt, zippers = sampleSingleStep(g, expr, request)

    print(nxt)
    #print([(subtree.path, subtree.tp, subtree.env) for subtree in zippers])
    print(zippers)



# request = arrow(tint, tint)
# i = 0
# for ll,_,p in g.enumeration(Context.EMPTY,[],request,
#                            12.,
#                            enumerateHoles=False):


#     print(i, p)
#     #if i==7: break
#     i+=1

def test_abstractHoles():

    #g = Grammar.uniform([k0,k1,addition, subtraction])
    bootstrapTarget_extra()
    expr = g.sample( request, sampleHoleProb=.2)
    expr = Program.parse('(lambda (map (lambda <HOLE>) $0))')
    #expr = Program.parse('(lambda (map (lambda (+ $0 17111)) (map (lambda <HOLE>) $0)))')
    #expr = Program.parse('(lambda (map (lambda (+ $0 1)) (map (lambda <HOLE>) $0)))')
    expr = Program.parse('(lambda (map (lambda (is-square $0)) $0))')

    #expr = Program.parse('(lambda (map (lambda 3) $0))')
    print("expr:", expr)

    p = expr.evaluate([])
    print("eval:", p([4,3,1]))


def test_abstractREPL():
    from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks
    from dreamcoder.domains.list.main import LearnedFeatureExtractor
    tasks = make_list_bootstrap_tasks()
    cuda = True

    featureExtractor = LearnedFeatureExtractor(tasks, testingTasks=tasks[-3:], cuda=cuda)
    valueHead = AbstractREPLValueHead(g, featureExtractor, H=64)

    sketch = Program.parse('(lambda (map (lambda <HOLE>) $0))')
    #print(tasks[2])
    task = tasks[2]
    task.examples = task.examples[:1] 
    print(task.examples)
    valueHead.computeValue(sketch, task)

def test_trainListREPL(T=2, repeat=True, test_every=3000, num_tasks=None, value=False,policy=False, share_extractor=True):
    """
    share_extractor means all value/policy heads will share the same LearnedFeatureExtractor which is often what you want unless
    comparing multiple value functions side by side.
    """
    assert value or policy
    from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloader
    from dreamcoder.domains.list.main import LearnedFeatureExtractor
    from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives
    from dreamcoder.valueHead import SimpleRNNValueHead
    from dreamcoder.policyHead import RNNPolicyHead
    from dreamcoder.Astar import Astar
    from likelihoodModel import AllOrNothingLikelihoodModel
    #import mlb
    taskloader = DeepcoderTaskloader(
        f'dreamcoder/domains/list/DeepCoder_data/T{T}_A2_V512_L10_train_perm.txt',
        allowed_requests=[arrow(tlist(tint),tlist(tint))],
        repeat=True,
        micro=num_tasks, # load only 3 tasks
        )
    testloader = DeepcoderTaskloader(
        f'dreamcoder/domains/list/DeepCoder_data/T{T}_A2_V512_L10_test_perm.txt',
        allowed_requests=[arrow(tlist(tint),tlist(tint))],
        repeat=False,
        micro=None, # load only 3 tasks
        )
    test_tasks = [testloader.getTask()[1] for _ in range(10)]
    
    class FakeRecognitionModel:
        # pretends to be whatever Astar wants from its RecognitionModel. Which isn't much lol
        def __init__(self,valueHead,policyHead):
            self.policyHead = policyHead
            self.valueHead = valueHead
    class FakeFrontier:
        # pretends to be whatever valueLossFromFrontier wants for simplicity
        def __init__(self,program,task):
            self.task = task # satisfies frontier.task call
            self._fullProg = program
            self.program = self # trick for frontier.sample().program._fullProg
        def sample(self):
            return self


    g = Grammar.uniform(deepcoderPrimitives())

    new_extractor = lambda: LearnedFeatureExtractor([], testingTasks=[], cuda=True) # test on same tasks you train on
    def extractor():
        if share_extractor:
            if not hasattr(extractor,'_extractor'):
                extractor._extractor = new_extractor()
            return extractor._extractor
        return new_extractor()

    # if you want to add more heads to compare them side by side just add them to the valueHeads or policyHeads list!
    policyHeads = [RNNPolicyHead(g, extractor(), H=512) if policy else None]
    #valueHeads = [SimpleRNNValueHead(g, extractor(), H=64) if value else None]
    valueHeads = [None]
    valueHeads = list(filter(None,valueHeads))
    policyHeads = list(filter(None,policyHeads))
    params = itertools.chain.from_iterable([list(head.parameters()) for head in valueHeads+policyHeads])
    optimizer = torch.optim.Adam(params, lr=0.001, eps=1e-3, amsgrad=True)

    j=0
    tstart = time()
    while True:
        prgms_and_tasks = [taskloader.getTask() for _ in range(1000)] # a full size batch but it loops every 3 tasks
        tasks = [task for program,task in prgms_and_tasks]
        frontiers = [FakeFrontier(program,task) for program,task in prgms_and_tasks]
        for f in frontiers: # list of 3 tasks that is looped 500 times
            for i,head in enumerate(valueHeads+policyHeads):
                head.zero_grad()
                if hasattr(head,'valueLossFromFrontier'):
                    loss = head.valueLossFromFrontier(f, g) # value loss
                else:
                    loss = head.policyLossFromFrontier(f, g) # policy loss
                loss.backward()
                optimizer.step()
                if j % 200 == 0:
                    print(f"[{j}] [{i}] {head.__class__.__name__} {loss.item()}")
            if j % test_every == 0:
                if j != 0:
                    print(f"{test_every} steps in {time()-tstart:.1f}s")
                print("Testing...")
                likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
                solver = Astar(FakeRecognitionModel(policyHeads[0].RNNHead,policyHeads[0]))
                for task in test_tasks:
                    fs, times, num_progs, solns = solver.infer(
                            g, 
                            [task],
                            likelihoodModel, 
                            timeout=3,
                            elapsedTime=0,
                            evaluationTimeout=0.01,
                            maximumFrontiers={task: 2},
                            CPUs=1,
                        ) 
                    solns = solns[task]
                    times = times[task]
                    if len(solns) > 0:
                        #mlb.green(f"solved {task.name} with {len(solns)} solns in {times:.2f}s (searched {num_progs} programs)")
                        print(f"solved {task.name} with {len(solns)} solns in {times:.2f}s (searched {num_progs} programs)")
                    else:
                        #mlb.red(f"failed to solve {task.name} (searched {num_progs} programs)")
                        print(f"failed to solve {task.name} (searched {num_progs} programs)")
                tstart = time()


            j += 1
    
            #loss = sum([valueHead.valueLossFromFrontier(frontiers[i], g) for i in range(3)])/3
            # For NUM_TASKS=1 it should converge to like .0001 or less or whatever roughly
            # For NUM_TASKS=3 it should print roughly this (tho sometimes it doesnt get quite that low):
            # 0 4.185908317565918
            # 200 2.2845935821533203
            # 400 0.09386047720909119
            # 600 0.1874145269393921
            # 800 0.25692281126976013
            # 1000 0.19681498408317566
            # 1200 0.04110711067914963
            # 1400 0.11137421429157257
            # 1600 2.1041994094848633
            # 1800 0.04916973039507866
            # 2000 0.03187175095081329
            # 2200 0.0027384415734559298
            # 2400 0.12562792003154755
            # 2600 0.07696416229009628
            # 2800 0.03026503510773182
            # 3000 0.003952879458665848
            # 3200 0.0077062807977199554
            # 3400 0.002124919556081295
            # 3600 0.006129148416221142
            # 3800 0.0011640205048024654
            # 4000 0.0007198153762146831
            # 4200 0.0002680253819562495
            # 4400 0.0006631703581660986
            # 4600 0.0005345264216884971
            # 4800 0.001332330284640193


def test_trainAbstractREPL():
    from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks
    from dreamcoder.domains.list.main import LearnedFeatureExtractor
    tasks = make_list_bootstrap_tasks()
    cuda = True

    TASKID = 3

    import dill
    with open('testFrontier.pickle', 'rb') as h:
        lst = dill.load(h)

    # for i, (frontier, g) in enumerate(lst):
    #     if i == TASKID:
    #         print(i)
    #         print(frontier.task.examples)
    #         print(frontier.sample().program._fullProg)

    frontier, g = lst[TASKID]

    featureExtractor = LearnedFeatureExtractor(tasks, testingTasks=tasks[-3:], cuda=cuda)
    valueHead = AbstractREPLValueHead(g, featureExtractor, H=64)
    #valueHead = SimpleRNNValueHead(g, featureExtractor, H=64)

    print(frontier.task.examples)
    print(frontier.sample().program._fullProg)
    optimizer = torch.optim.Adam(valueHead.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    for i in range(400):
        valueHead.zero_grad()

        
        losses = [valueHead.valueLossFromFrontier(frontier, g) for frontier, g in lst]
        loss = sum(losses)
        print(loss.data.item())
        loss.backward()
        optimizer.step()

def test_abstractHolesTower():


    def _empty_tower(h): return (h,[])
    expr = Program.parse('(lambda (1x3 (moveHand <HOLE> (reverseHand <TowerHOLE>))) )') 
    expr = Program.parse('(lambda (1x3 (moveHand <HOLE> (reverseHand <TowerHOLE>))) )') 
    #expr = Program.parse('(lambda (<TowerHOLE>) )') 
    expr = Program.parse('(lambda (tower_loopM <HOLE> (lambda (lambda <TowerHOLE>)) <TowerHOLE>))')
    #animateTower('test', expr)

    x = expr.evaluateHolesDebug([])(_empty_tower)(TowerState(history=[])) #can initialize tower state with 
    print(x)
    

def test_abstractHolesTowerValue():
    from dreamcoder.domains.tower.towerPrimitives import primitives, new_primitives, animateTower
    from dreamcoder.domains.tower.main import TowerCNN
    from dreamcoder.domains.tower.makeTowerTasks import makeSupervisedTasks
    from dreamcoder.valueHead import TowerREPLValueHead

    g = Grammar.uniform(new_primitives,
                         continuationType=ttower)
    tasks = makeSupervisedTasks()

    def _empty_tower(h): return (h,[])

    exprs = []
    exprs.append (Program.parse('(lambda (tower_loopM <HOLE> (lambda (lambda (moveHand 3 (3x1 $0)))) <TowerHOLE>)) ') )
    exprs.append (Program.parse('(lambda (1x3 (moveHand <HOLE> (reverseHand <TowerHOLE>))) )') )
    #expr = Program.parse('(lambda (<TowerHOLE>) )') 
    exprs.append (Program.parse('(lambda (tower_loopM <HOLE> (lambda (lambda <TowerHOLE>)) <TowerHOLE>))'))
    #animateTower('test', expr)
    #expr = Program.parse('(lambda (<TowerHOLE>) )') 
    exprs.append (Program.parse('(lambda (3x1 (1x3 <TowerHOLE>) ))') )
    exprs.append (Program.parse('(lambda (reverseHand (1x3 <TowerHOLE>) ))') )
    exprs.append (Program.parse('(lambda (tower_loopM <HOLE> (lambda (lambda <TowerHOLE>)) <TowerHOLE>))'))
    exprs.append (Program.parse('(lambda (tower_loopM 1 (lambda (lambda <TowerHOLE>)) <TowerHOLE>))'))
    exprs.append (Program.parse('(lambda (tower_embed (lambda (moveHand 1 (1x3 $0))) $0 ) )'))
    #expr = Program.parse('(lambda (1x3 (tower_embed (lambda (1x3 $0 )) <TowerHOLE> )) )')
    #expr = Program.parse('(lambda (tower_loopM <HOLE> (lambda (lambda <TowerHOLE>)) <TowerHOLE>))')
    #print(executeTower(expr))
    exprs.append (Program.parse('(lambda (tower_loopM <HOLE> (lambda (lambda (1x3 $0))) <TowerHOLE>))'))
    #print(expr.evaluateHolesDebug([])(_empty_tower)(TowerState(history=[])))
    #expr = Program.parse('(lambda (tower_loopM 5 (lambda (lambda <TowerHOLE>)) (3x1 <TowerHOLE>)))')

    featureExtractor = TowerCNN(tasks, testingTasks=tasks[-3:], cuda=True)
    valueHead = TowerREPLValueHead(g, featureExtractor, H=1024)

    #for expr in exprs:

    expr = "(lambda (1x3 (tower_loopM 8 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM 8 (lambda (lambda (moveHand 6 (3x1 $0)))) $0))))) (1x3 (1x3 (tower_embed (lambda <TowerHOLE>)((lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 <TowerHOLE>))))) <HOLE> (lambda <TowerHOLE>))))))))"
    expr = "(lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) <TowerHOLE> 1 (lambda <TowerHOLE>)))"
    
    expr = "(lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) $0 1 (lambda (tower_embed (lambda $0) (moveHand 3 (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) (1x3 <TowerHOLE>) 5 (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) <TowerHOLE> <HOLE> (lambda <TowerHOLE>)))))))))"
    
    expr = "(lambda (reverseHand (#(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0)))))))) 2 8  (#(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0)))))))) 1 2 (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) <TowerHOLE> 8 (lambda <TowerHOLE>))))))"
    expr=Program.parse(expr)
    print()
    print()
    print()
    print("old:", expr)
    print()
    expr = expr.betaNormalForm()
    print("new:", expr)
    print()
    print()
    print()

    x = valueHead.computeValue(expr, tasks[1])
    # x = expr.evaluateHolesDebug([])(_empty_tower)(TowerState(history=[])) #can initialize tower state with 
    print(x)

def test_TowerREPLValueConvergence():

    from dreamcoder.domains.tower.towerPrimitives import primitives, new_primitives, animateTower
    from dreamcoder.domains.tower.main import TowerCNN
    from dreamcoder.domains.tower.makeTowerTasks import makeSupervisedTasks
    from dreamcoder.valueHead import TowerREPLValueHead

    g = Grammar.uniform(new_primitives,
                         continuationType=ttower)
    tasks = makeSupervisedTasks()

    def _empty_tower(h): return (h,[])

    import dill
    with open('testTowerFrontiers.pickle', 'rb') as h:
        lst = dill.load(h)

    #import pdb; pdb.set_trace()

    # for i, frontier in enumerate(lst):
    #     animateTower(f'front{i}', frontier.entries[0].program._fullProg)

    def saveState(path, prog):
        import scipy.misc
        state, actions = prog.evaluate([])(_empty_tower)(TowerState(history=[]))
        plan = [tup for tup in state.history if isinstance(tup, tuple)]
        hand = state.hand
        image = renderPlan(plan, drawHand=hand, pretty=False, drawHandOrientation=state.orientation)
        scipy.misc.imsave(path, image)

    saveState("test1.png", Program.parse('(lambda (reverseHand (1x3 (1x3 $0))))') )
    saveState("test2.png", Program.parse('(lambda  (1x3 (1x3 $0)) )'))


    featureExtractor = TowerCNN(tasks, testingTasks=tasks[-3:], cuda=True)
    valueHead = TowerREPLValueHead(g, featureExtractor, H=1024)
    optimizer = torch.optim.Adam(valueHead.parameters(), lr=0.001, eps=1e-3, amsgrad=True)


    for i in range(5000):
        valueHead.zero_grad()
        
        losses = [valueHead.valueLossFromFrontier(frontier, g) for frontier in lst ]#+ lst[2:]]
        #looks like 
        loss = sum(losses)
        print(loss.data.item())
        if not loss.requires_grad:
            print("error on loss")
            continue
        loss.backward()
        optimizer.step()

def test_TowerRNNPolicySearch():

    from dreamcoder.domains.tower.towerPrimitives import primitives, new_primitives, animateTower
    from dreamcoder.domains.tower.main import TowerCNN
    from dreamcoder.domains.tower.makeTowerTasks import makeSupervisedTasks
    from dreamcoder.valueHead import TowerREPLValueHead
    from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead

    g = Grammar.uniform(new_primitives,
                         continuationType=ttower)
    tasks = makeSupervisedTasks()

    def _empty_tower(h): return (h,[])

    import dill
    with open('testTowerFrontiers.pickle', 'rb') as h:
        lst = dill.load(h)

    #import pdb; pdb.set_trace()

    # for i, frontier in enumerate(lst):
    #     animateTower(f'front{i}', frontier.entries[0].program._fullProg)

    def saveState(path, prog):
        import scipy.misc
        state, actions = prog.evaluate([])(_empty_tower)(TowerState(history=[]))
        plan = [tup for tup in state.history if isinstance(tup, tuple)]
        hand = state.hand
        image = renderPlan(plan, drawHand=hand, pretty=False, drawHandOrientation=state.orientation)
        scipy.misc.imsave(path, image)

    saveState("test1.png", Program.parse('(lambda (reverseHand (1x3 (1x3 $0))))') )
    saveState("test2.png", Program.parse('(lambda  (1x3 (1x3 $0)) )'))


    featureExtractor = TowerCNN(tasks, testingTasks=tasks[-3:], cuda=True)
    #valueHead = TowerREPLValueHead(g, featureExtractor, H=1024)
    policyHead = RNNPolicyHead(g, featureExtractor, H=1024)
    optimizer = torch.optim.Adam(policyHead.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    for frontier in lst:
        print(frontier.entries[0].program._fullProg)

    for i in range(10):
        policyHead.zero_grad()
        
        losses = [policyHead.policyLossFromFrontier(frontier, g) for frontier in lst ]#+ lst[2:]]
        #looks like 
        loss = sum(losses)
        print(loss.data.item())
        if not loss.requires_grad:
            print("error on loss")
            continue
        loss.backward()
        optimizer.step()

    policyHead.cpu()
    policyHead.use_cuda = False
    torch.set_num_threads(1)

    from likelihoodModel import AllOrNothingLikelihoodModel

    graph = ""
    ID = 'towers' + str(20)
    runType ="PolicyNoBigramLoss" #"Policy"
    path = f'experimentOutputs/{ID}{runType}Sample_SRE=True{graph}.pickle'
    print(path)
    with open(path, 'rb') as h:
        r = dill.load(h)
    
    if not hasattr(r.recognitionModel, "policyHead"):
        r.recognitionModel.policyHead = BasePolicyHead()
    g = r.grammars[-1]
    print(r.recognitionModel.gradientStepsTaken)
    solver = r.recognitionModel.solver
    times = []
    ttasks = r.getTestingTasks()
    for i in range(30):
        tasks = [ttasks[i]]
        likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        #tasks = [frontier.task]
        fs, searchTimes, totalNumberOfPrograms, reportedSolutions = solver.infer(g, tasks, likelihoodModel, 
                                            timeout=30,
                                            elapsedTime=0,
                                            evaluationTimeout=0.01,
                                            maximumFrontiers={tasks[0]: 2},
                                            CPUs=1,
                                            ) 
        print("done")
        print("total prog", totalNumberOfPrograms)  
        print("searchTimes", searchTimes)

def test_TowerREPLPolicyConvergence():

    from dreamcoder.domains.tower.towerPrimitives import primitives, new_primitives, animateTower
    from dreamcoder.domains.tower.main import TowerCNN
    from dreamcoder.domains.tower.makeTowerTasks import makeSupervisedTasks
    from dreamcoder.valueHead import TowerREPLValueHead
    from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead, REPLPolicyHead

    g = Grammar.uniform(new_primitives,
                         continuationType=ttower)
    tasks = makeSupervisedTasks()

    def _empty_tower(h): return (h,[])

    import dill
    with open('testTowerFrontiers.pickle', 'rb') as h:
        lst = dill.load(h)

    featureExtractor = TowerCNN(tasks, testingTasks=tasks[-3:], cuda=True)
    #valueHead = TowerREPLValueHead(g, featureExtractor, H=1024)
    policyHead = REPLPolicyHead(g, featureExtractor, H=1024, encodeTargetHole=False, canonicalOrdering=True)
    optimizer = torch.optim.Adam(policyHead.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    for frontier in lst:
        print(frontier.entries[0].program._fullProg)

    for i in range(10000):
        policyHead.zero_grad()
        losses = [policyHead.policyLossFromFrontier(frontier, g) for frontier in lst ]#+ lst[2:]]
        #looks like 
        loss = sum(losses)
        print(i, loss.data.item())
        if not loss.requires_grad:
            print("error on loss")
            continue
        loss.backward()
        optimizer.step()

def finetuneAndTestPolicy():
    from dreamcoder.domains.tower.towerPrimitives import primitives, new_primitives, animateTower
    from dreamcoder.domains.tower.main import TowerCNN
    from dreamcoder.domains.tower.makeTowerTasks import makeSupervisedTasks
    from dreamcoder.valueHead import TowerREPLValueHead
    from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead, REPLPolicyHead

    graph = ""
    ID = 'towers' + str(20)
    runType ="" #"Policy"
    model = "REPL" #"RNN"
    path = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'
    print(path)
    import dill
    with open(path, 'rb') as h:
        r = dill.load(h)

    g = r.grammars[-1]

    featureExtractor = r.recognitionModel.valueHead.featureExtractor
    
    if model == "REPL":
        policyHead = REPLPolicyHead(g, featureExtractor, 
                                                H=1024,
                                                encodeTargetHole=False,
                                                canonicalOrdering=True)
        policyHead.REPLHead = r.recognitionModel.valueHead
    elif model =="RNN":
        policyHead = RNNPolicyHead(g, featureExtractor, 
                                                H=512,
                                                encodeTargetHole=False,
                                                canonicalOrdering=True)
        policyHead.RNNHead = r.recognitionModel.valueHead

    #policyHead.featureExtractor = policyHead.valueHead.featureExtractor

    r.recognitionModel.cpu()
    del r.recognitionModel
    policyHead.cuda()

    optimizer = torch.optim.Adam(policyHead.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    testingTasks = r.getTestingTasks()
    trainingTasks = r.taskSolutions.keys()

    trainFrontiers = [r.allFrontiers.get(t, None) for t in trainingTasks] 
    trainFrontiers = [t for t in trainFrontiers if (t and t.entries)] #if not empty
    print("num train frontiers", len(trainFrontiers))

    testFrontiers = [r.allFrontiers.get(t, None) for t in testingTasks]
    testFrontiers = [r.recognitionTaskMetrics.get(t, {'frontier': None} ).get('frontier', None) for t in testingTasks]
    testFrontiers = [t for t in testFrontiers if (t and t.entries)] #if not empty
    print("num test frontiers", len(testFrontiers))

    #import pdb; pdb.set_trace()

    for i in range(75):
        policyHead.zero_grad()
        losses = [policyHead.policyLossFromFrontier(frontier, g) for frontier in trainFrontiers ]#+ lst[2:]]
        loss = sum(losses)
        print(f"epoch {i}: avg loss {loss.data.item()/len(losses)}")

        if not loss.requires_grad:
            print("error on loss")
            continue
        loss.backward()
        optimizer.step()

        testLoss = [policyHead.policyLossFromFrontier(frontier, g) for frontier in testFrontiers ]
        avgTestLoss = sum(testLoss)/len(testLoss)
        print(f"\t {avgTestLoss.data.item()}")

    testLosses = []
    for i in range(10):
        policyHead.eval()    
        losses = [policyHead.policyLossFromFrontier(frontier, g) for frontier in testFrontiers ]#+ lst[2:]]
        loss = sum(losses)/len(losses)
        #print(i, loss.data.item())
        testLosses.append(loss.data.item())

    print(sum(testLosses) / len(testLosses))

from dreamcoder.symbolicAbstractTowers import *
def test_abstraction_bug():

    def showState(sk):
        absSketch = ConvertSketchToAbstract().execute(sk.betaNormalForm())
        absState = executeAbstractSketch(absSketch) #gets he history
        print(absState)
        print(absState.history)
    
    sk130 = Program.parse("(lambda (tower_loopM 5 (lambda (lambda (tower_embed (lambda (#(lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) (moveHand $2 $0) $4 (lambda (reverseHand $0))))))))) $0 $1 4))) $2 2 (reverseHand (#(lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) $1 $0 (lambda (1x3 (reverseHand $0)))))) <TowerHOLE> <intHOLE>)))) $0))) $0))")
    sk131 = Program.parse("(lambda (tower_loopM 5 (lambda (lambda (tower_embed (lambda (#(lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) (moveHand $2 $0) $4 (lambda (reverseHand $0))))))))) $0 $1 4))) $2 2 (reverseHand (#(lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) $1 $0 (lambda (1x3 (reverseHand $0)))))) <TowerHOLE> 1)))) $0))) $0))")

    print(130)
    showState(sk130)

    print(131)
    showState(sk131)

import dill
def test_policyTiming():
    from dreamcoder.Astar import Astar
    from likelihoodModel import AllOrNothingLikelihoodModel
    from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead, REPLPolicyHead
    from dreamcoder.domains.tower.makeTowerTasks import makeNewMaxTasks

    sys.setrecursionlimit(50000)
    graph = ""
    ID = 'towers' + str(3)
    runType = "" #"PolicyOnly" #"Policy"
    #runType =""
    model = "Sample"
    useREPLnet = False
    path = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'
    print(path)
    with open(path, 'rb') as h:
        r = dill.load(h)
    
    # useREPLnet = True
    # model = "REPL"

    if not hasattr(r.recognitionModel, "policyHead"):
        r.recognitionModel.policyHead = BasePolicyHead()

    if useREPLnet:
        path=f"experimentOutputs/{ID}PolicyOnlyREPL.pickle_RecModelOnly"
        with open(path, 'rb') as h:
            repl = torch.load(h)
        #import pdb; pdb.set_trace()

        r.recognitionModel = repl
        #print(r.recognitionModel.policyHead)

    g = r.grammars[-1]
    print(r.recognitionModel.gradientStepsTaken)
    solver = r.recognitionModel.solver

    solver = Astar(r.recognitionModel)

    times = []
    ttasks = r.getTestingTasks()


    # testFrontiers = [r.recognitionTaskMetrics.get(t, {'frontier': None} ).get('frontier', None) for t in ttasks]
    # testFrontiers = [t for t in testFrontiers if (t and t.entries)] #if not empty

    # maxSize = 0
    # for f in testFrontiers:
    #     for entry in f.entries:
    #         p = entry.program
    #         maxSize = max(maxSize, p.size())
    # print("max size", maxSize)

    # print("num test frontiers", len(testFrontiers))
    # policyHead = r.recognitionModel.policyHead
    # testLosses = []
    # for i in range(10):
    #     policyHead.eval()
    #     losses = [policyHead.policyLossFromFrontier(frontier, g) for frontier in testFrontiers ]#+ lst[2:]]
    #     loss = sum(losses)/len(losses)
    #     policyHead.zero_grad()
    #     #print(i, loss.data.item())
    #     testLosses.append(loss.data.item())
    # print("average loss on test frontiers:")
    # print(sum(testLosses) / len(testLosses))

    ttasks = makeNewMaxTasks() + r.getTestingTasks()

    #import pdb; pdb.set_trace()
    print("using max tasks")

    #r.recognitionModel.policyHead.cpu()

    nhit = 0
    stats = {}
    nums = {}
    for t in ttasks:
        tasks = [t]
        #tasks = []
        print(tasks)
        likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        #tasks = [frontier.task]

        g = r.recognitionModel.grammarOfTask(tasks[0]).untorch()
        fs, searchTimes, totalNumberOfPrograms, reportedSolutions = solver.infer(g, tasks, likelihoodModel, 
                                            timeout=30,
                                            elapsedTime=0,
                                            evaluationTimeout=0.01,
                                            maximumFrontiers={tasks[0]: 2},
                                            CPUs=1,
                                            ) 
        print("done")
        print("total prog", totalNumberOfPrograms)  
        print("searchTimes", searchTimes)
        if list(searchTimes.values())[0]:
            nhit += 1

        for k, v in reportedSolutions.items():
            stats[k] = v
    
        nums[tasks[0]] = totalNumberOfPrograms

    print("n hit:", nhit)

    class PseudoResult:
        pass 

    pseudoResult = PseudoResult()
    pseudoResult.testingSearchStats = [stats]
    pseudoResult.testingNumOfProg = [nums]

    savePath = f'experimentOutputs/{ID}{runType}PseudoResult{model}_SRE=True.pickle'
    with open(savePath, 'wb') as h:
        dill.dump(pseudoResult, h)
    print("saved at", savePath)


if __name__=='__main__':
    #findError()
    #testSampleWithHoles()
    #test_training_stuff()
    #test_holeFinder()
    #full = test_getTrace()
    #test_sampleOneStep()
    #test_sampleSingleStep()
    #test_SingleStep1()
    #test_abstractHoles()
    #test_abstractREPL()
    # test_trainAbstractREPL()
    # bootstrapTarget_extra()
    # from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks
    # from dreamcoder.domains.list.main import LearnedFeatureExtractor
    # tasks = make_list_bootstrap_tasks()
    # expr = Program.parse('(lambda (map (lambda (is-square $0)) $0))')
    # test_abstractHolesTower()
    # test_abstractHolesTowerValue()
    #test_policyTiming()
    # test_TowerREPLValueConvergence()
    # test_TowerRNNPolicyConvergence()
    # test_TowerREPLPolicyConvergence()
    # finetuneAndTestPolicy()
    # test_abstraction_bug()

    with torch.cuda.device(6):
        test_trainListREPL(value=True,policy=True)
