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
import itertools

from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloader
from dreamcoder.domains.list.main import LearnedFeatureExtractor
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives
from dreamcoder.valueHead import SimpleRNNValueHead, ListREPLValueHead, BaseValueHead
from dreamcoder.policyHead import RNNPolicyHead,BasePolicyHead,ListREPLPolicyHead, NeuralPolicyHead
from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import mlb
import time

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
def extractor(group,H):
    """
    Returns an extractor object. Reuses the same one it gave previously if shared=True
    """
    new_extractor = lambda: LearnedFeatureExtractor([], testingTasks=[], H=H, cuda=True)
    if not group in extractor._groups:
        extractor._groups[group] = new_extractor()
    return extractor._groups[group]
extractor._groups = {}

def test_trainListREPL(T=1, repeat=True, freeze_examples=False, print_every=200, H=64, test_every=1000, batch_size=1000, num_tasks=None, value='tied',policy=True):
    """
    share_extractor means all value/policy heads will share the same LearnedFeatureExtractor which is often what you want unless
    comparing multiple value functions side by side.

    Test 1: a single super simple program and the training examples are all the same
        T=1
        freeze_examples=True
        batch_size=1 # this is important
        num_tasks=1
        repeat=True
        test_every=None
    Test 2: same as test 1 but training examples change for that one program
        T=1
        freeze_examples=False
        # (batch_size no longer matters)
        num_tasks=1
        repeat=True
        test_every=None
    Test 3:
        num_tasks = 3
    Test 4:
        T=1
        num_tasks = None
        test_every = 1000


    """
    assert value or policy

    print("intializing tensorboard")
    w = SummaryWriter(
        log_dir='runs/test',
        max_queue=10,
    )
    print("done")
    #w.add_image('colorful boi', torch.rand(3, 20, 20), dataformats='CHW')

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
    num_test = 5 if T==1 else 10
    test_tasks = [testloader.getTask()[1] for _ in range(num_test)] if test_every is not None else None
    g = Grammar.uniform(deepcoderPrimitives())

    rnn_ph = RNNPolicyHead(g, extractor(0,H), H=H) if policy else None
    #pHead = BasePolicyHead()
    if value == 'tied':
        rnn_vh = rnn_ph.RNNHead
    else:
        rnn_vh = SimpleRNNValueHead(g, extractor(0,H), H=H) if value else None

    repl_ph = ListREPLPolicyHead(g, extractor(1,H), H=H, encodeTargetHole=True)
    if value == 'tied':
        repl_vh = repl_ph.RNNHead
    else:
        repl_vh = ListREPLValueHead(g, extractor(1,H), H=H) if value else None
    
    rnn_vh = rnn_ph = None
    #repl_vh = repl_ph = None

    
    heads = list(filter(None,[repl_ph, repl_vh, rnn_ph, rnn_vh]))
    vheads = list(filter(lambda h: isinstance(h,BaseValueHead),heads))
    pheads = list(filter(lambda h: isinstance(h,NeuralPolicyHead),heads))

    params = itertools.chain.from_iterable([head.parameters() for head in heads])
    optimizer = torch.optim.Adam(params, lr=0.001, eps=1e-3, amsgrad=True)

    rnn_astar = {'value':rnn_vh, 'policy':rnn_ph}
    repl_astar = {'value':repl_vh, 'policy':repl_ph}
    #astars = [rnn_astar]
    astars = [repl_astar]

    j=0
    tstart = time.time()
    frontiers = None
    while True:
        # TODO you should really rename getTask to getProgramAndTask or something
        if frontiers is None or not freeze_examples:
            prgms_and_tasks = [taskloader.getTask() for _ in range(batch_size)]
            tasks = [task for program,task in prgms_and_tasks]
            frontiers = [FakeFrontier(program,task) for program,task in prgms_and_tasks]
        for f in frontiers: # work thru batch of `batch_size` examples
            #mlb.green(f._fullProg)
            for head in heads:
                head.zero_grad()
            # TODO TEMP
            losses = []
            for head in vheads:
                loss = head.valueLossFromFrontier(f, g)
                losses.append(loss)
            for head in pheads:
                loss = head.policyLossFromFrontier(f, g)
                losses.append(loss)
            
            sum(losses).backward()
            optimizer.step()

            # printing and logging
            if j % print_every == 0:
                for head,loss in zip(vheads+pheads,losses): # important that the right things zip together (both lists ordered same way)
                    print(f"[{j}] {head.__class__.__name__} {loss.item()}")
                    w.add_scalar(head.__class__.__name__, loss.item(), j)
                print()
                w.flush()

            # testing
            if test_every is not None and j % test_every == 0:
                if j != 0:
                    elapsed = time.time()-tstart
                    print(f"{test_every} steps in {elapsed:.1f}s ({test_every/elapsed:.1f} steps/sec)")
                for astar in astars:
                    print(f"Testing: {astar['policy'].__class__.__name__} and {astar['value'].__class__.__name__}")

                    likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
                    solver = Astar(FakeRecognitionModel(astar['value'],astar['policy']))
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
                            mlb.green(f"solved {task.name} with {len(solns)} solns in {times:.2f}s (searched {num_progs} programs)")
                        else:
                            mlb.red(f"failed to solve {task.name} (searched {num_progs} programs)")
                    print()
                tstart = time.time()
            j += 1
    

if __name__ == '__main__':
    with torch.cuda.device(6):
        test_trainListREPL()