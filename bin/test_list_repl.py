try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import shutil
import sys
sys.path.append('/afs/csail.mit.edu/u/m/mlbowers/clones') # needed for tmux sudo workaround

import hydra
from hydra import utils
from omegaconf import DictConfig,OmegaConf,open_dict


import argparse
from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *
from dreamcoder.SMC import SearchResult

from dreamcoder.domains.tower.towerPrimitives import *
import itertools
import torch
import numpy as np

from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloader
from dreamcoder.domains.list.main import ListFeatureExtractor
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives
from dreamcoder.valueHead import SimpleRNNValueHead, ListREPLValueHead, BaseValueHead, SampleDummyValueHead
from dreamcoder.policyHead import RNNPolicyHead,BasePolicyHead,ListREPLPolicyHead, NeuralPolicyHead
from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from torch.utils.tensorboard import SummaryWriter
import mlb
import time
import matplotlib.pyplot as plot

class FakeRecognitionModel(nn.Module):
    # pretends to be whatever Astar wants from its RecognitionModel. Which isn't much lol
    def __init__(self,valueHead,policyHead):
        super().__init__()
        self.policyHead = policyHead
        self.valueHead = valueHead
    # def save(self, path):
    #     torch.save(self.state_dict(),path)
    # @staticmethod
    # def load(path):
    #     return torch.load

class Poisoned: pass
class State:
    def __init__(self):
        self.no_pickle = []
    # these @properties are kinda important. Putting them in __init sometimes gives weird behavior after loading
    @property
    def state(self):
        return self
    @property
    def as_kwargs(self):
        kwargs = {'state': self.state}
        kwargs.update(self.__dict__)
        return kwargs
    def new(self,cfg):
        self.cwd = os.getcwd()
        allowed_requests = None if cfg.data.allow_complex_requests else [arrow(tlist(tint),tlist(tint))]

        taskloader = DeepcoderTaskloader(
            utils.to_absolute_path(f'dreamcoder/domains/list/DeepCoder_data/T{cfg.data.T}_A2_V512_L10_train_perm.txt'),
            allowed_requests=allowed_requests,
            repeat=cfg.data.repeat,
            num_tasks=cfg.data.num_tasks,
            )
        testloader = DeepcoderTaskloader(
            utils.to_absolute_path(f'dreamcoder/domains/list/DeepCoder_data/T{cfg.data.T}_A2_V512_L10_test_perm.txt'),
            allowed_requests=allowed_requests,
            repeat=False,
            num_tasks=None,
            )

        extractor = ExtractorGenerator(cfg=cfg, maximumLength = taskloader.L+2)

        test_tasks = testloader.getTasks(cfg.data.num_tests, ignore_eof=True)
        print(f'Got {len(test_tasks)} testing tasks')
        num_valid = int(cfg.data.valid_frac*len(test_tasks))
        validation_tasks = test_tasks[:num_valid]
        test_tasks = test_tasks[num_valid:]
        print(f'Split into {len(test_tasks)} testing tasks and {len(validation_tasks)} validation tasks')
        validation_frontiers = [FakeFrontier(program, task) for program, task in validation_tasks]

        g = Grammar.uniform(deepcoderPrimitives())



        if cfg.model.policy:
            phead = {
                'rnn': RNNPolicyHead,
                'repl': ListREPLPolicyHead,
            }[cfg.model.type](cfg=cfg, extractor=extractor(0), g=g)
        else:
            phead = BasePolicyHead()
        if cfg.model.value:
            if cfg.model.tied:
                assert cfg.model.policy
                vhead = phead.vhead
            else:
                vhead = {
                    'rnn': SimpleRNNValueHead,
                    'repl': ListREPLValueHead,
                }[cfg.model.type](cfg=cfg, extractor=extractor(0), g=g)
        else:
            vhead = SampleDummyValueHead()

        heads = [vhead,phead]

        if cfg.cuda:
            heads = [h.cuda() for h in heads]


        max_depth = 10

        params = itertools.chain.from_iterable([head.parameters() for head in heads])
        optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, eps=1e-3, amsgrad=True)

        astar = Astar(FakeRecognitionModel(vhead, phead), maxDepth=max_depth)
        j=0
        frontiers = None
        self.update(locals()) # do self.* = * for everything
        self.post_load()
    
    def save(self, locs, name):
        """
        use like state.save(locals(),"name_of_save")
        """
        self.update(locs)
        temp = {}
        for key in self.no_pickle:
            temp[key] = self[key]
            self[key] = Poisoned
        if not os.path.isdir('saves'):
            os.mkdir('saves')
        path = f'saves/{name}'
        print(f"saving state to {path}...")
        torch.save(self, f'{path}.tmp')
        print('critical step, do not interrupt...')
        shutil.move(f'{path}.tmp',f'{path}')
        print("done")
        for key in self.no_pickle:
            self[key] = temp[key]
    def load(self, path):
        path = utils.to_absolute_path(path)
        state = torch.load(path)
        self.update(state.__dict__)
        self.post_load()
    def post_load(self):
        print(f"chdir to {self.cwd}")
        os.chdir(self.cwd)
        print("intializing tensorboard")
        w = SummaryWriter(
            log_dir='tb',
            max_queue=10,
        )
        print("done")
        self.no_pickle.append('w')
        self.update(locals())
        

    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)
    def __repr__(self):
        body = []
        for k,v in self.__dict__.items():
            body.append(f'{k}: {repr(v)}')
        body = '\n\t'.join(body)
        return f"State(\n\t{body}\n)"
    def update(self,dict):
        for k,v in dict.items():
            if hasattr(type(self), k) and isinstance(getattr(type(self),k), property):
                continue # dont overwrite properties (throws error)
            self[k] = v

def train_model(
    state,
    cfg,
    taskloader,
    vhead,
    phead,
    heads,
    w,
    g,
    optimizer,
    astar,
    test_tasks,
    validation_frontiers,
    frontiers=None,
    best_validation_loss=np.inf,
    j=0,
    **kwargs,
        ):
    print(f"j:{j}")
    tstart = None
    phead.featureExtractor.run_tests()
    while True:

        if cfg.loop.max_steps and j > cfg.loop.max_steps:
            break

        # TODO you should really rename getTask to getProgramAndTask or something
        if frontiers is None or not cfg.data.freeze_examples:
            prgms_and_tasks = taskloader.getTasks(1000)
            tasks = [task for program,task in prgms_and_tasks]
            frontiers = [FakeFrontier(program,task) for program,task in prgms_and_tasks]
        for f in frontiers: # work thru batch of `batch_size` examples
            for head in heads:
                head.zero_grad()
            vloss = vhead.valueLossFromFrontier(f, g)
            ploss = phead.policyLossFromFrontier(f, g)
            loss = vloss + ploss
            loss.backward()
            optimizer.step()

            mlb.freezer('pause')

            if mlb.predicate('return'):
                return
            if mlb.predicate('which'):
                print(OmegaConf.to_yaml(cfg))
                print(os.getcwd())

            # printing and logging
            if j % cfg.loop.print_every == 0:
                for head,loss in zip([vhead,phead],[vloss,ploss]): # important that the right things zip together (both lists ordered same way)
                    print(f"[{j}] {head.__class__.__name__} {loss.item()}")
                    w.add_scalar(head.__class__.__name__, loss.item(), j)
                print()
                w.flush()

            # validation
            if cfg.loop.test_every is not None and j % cfg.loop.test_every == 0:
                # timer
                if tstart is not None:
                    elapsed = time.time()-tstart
                    print(f"{cfg.loop.test_every} steps in {elapsed:.1f}s ({cfg.loop.test_every/elapsed:.1f} steps/sec)")
                # get valid loss
                with torch.no_grad():
                    vloss = ploss = 0
                    for f in validation_frontiers:
                        vloss += vhead.valueLossFromFrontier(f, g)
                        ploss += phead.policyLossFromFrontier(f, g)
                    vloss /= len(validation_frontiers)
                    ploss /= len(validation_frontiers)
                # print valid loss
                for head, loss in zip([vhead,phead],[vloss,ploss]):
                    mlb.blue(f"Validation Loss [{j}] {head.__class__.__name__} {loss.item()}")
                    w.add_scalar(head.__class__.__name__+' Validation Loss', loss.item(), j)

                # test on valid set
                model_results = test_models([astar], validation_frontiers, timeout=cfg.loop.timeout, verbose=True)
                accuracy = len(model_results[0].search_results) / len(validation_frontiers) * 100
                w.add_scalar(head.__class__.__name__+' Validation Accuracy', accuracy, j)
                plot_model_results(model_results, file='validation', salt=j)

                # save model if new record for lowest validation loss
                val_loss = (vloss+ploss).item()
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    state.save(locals(),'best_validation')
                    mlb.green('new lowest validation loss!')
                    w.add_scalar(head.__class__.__name__+' Validation Loss (best)', loss, j)
                    w.add_scalar(head.__class__.__name__+' Validation Accuracy (best)', accuracy, j)

                tstart = time.time()
            if mlb.predicate('test'):
                model_results = test_models([astar],test_tasks,timeout=cfg.loop.timeout, verbose=True)
                plot_model_results(model_results, file='test', salt=j)

            j += 1 # increment before saving so we resume on the next iteration
            if cfg.loop.save_every is not None and (j-1) % cfg.loop.save_every == 0: # the j-1 is important for not accidentally repeating a step
                state.save(locals(),'autosave')
            

    #def __getstate__(self):
        #Classes can further influence how their instances are pickled; if the class defines the method __getstate__(), it is called and the returned object is pickled as the contents for the instance, instead of the contents of the instance’s dictionary. If the __getstate__() method is absent, the instance’s __dict__ is pickled as usual.
    #def __setstate__(self,state):
        #Upon unpickling, if the class defines __setstate__(), it is called with the unpickled state. In that case, there is no requirement for the state object to be a dictionary. Otherwise, the pickled state must be a dictionary and its items are assigned to the new instance’s dictionary.
        #Note If __getstate__() returns a false value, the __setstate__() method will not be called upon unpickling.

class FakeFrontier:
    # pretends to be whatever valueLossFromFrontier wants for simplicity
    def __init__(self,program,task):
        self.task = task # satisfies frontier.task call
        self._fullProg = program
        self.program = self # trick for frontier.sample().program._fullProg
    def sample(self):
        return self

class ExtractorGenerator:
    def __init__(self,cfg,maximumLength):
        self.cfg = cfg
        self.maximumLength = maximumLength
        self._groups = {}
    def __call__(self, group):
        """
        Returns an extractor object. If called twice with the same group (an int or string or anything) the same object will be returned (ie share weights)
        """
        if group not in self._groups:
            self._groups[group] = ListFeatureExtractor(maximumLength=self.maximumLength, cfg=self.cfg)
        return self._groups[group]


def test_models(astars, test_tasks, timeout, verbose=True):
    if len(test_tasks) > 0 and isinstance(test_tasks[0], FakeFrontier):
        test_tasks = [f.task for f in test_tasks]
    if len(test_tasks) > 0 and isinstance(test_tasks[0], (tuple, list)):
        test_tasks = [task for program,task in test_tasks]
    model_results = []
    for astar in astars:
        name = f"{astar.owner.policyHead.__class__.__name__}_&&_{astar.owner.valueHead.__class__.__name__}"
        print(f"Testing: {name}")
        search_results = []
        likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        for task in test_tasks:
            g = Grammar.uniform(deepcoderPrimitives())
            fs, times, num_progs, solns = astar.infer(
                    g, 
                    [task],
                    likelihoodModel, 
                    timeout=timeout,
                    elapsedTime=0,
                    evaluationTimeout=0.01,
                    maximumFrontiers={task: 2},
                    CPUs=1,
                ) 
            solns = solns[task]
            times = times[task]
            if len(solns) > 0:
                assert len(solns) == 1 # i think this is true, I want it to be true lol
                soln = solns[0]
                search_results.append(soln)
                if verbose: mlb.green(f"solved {task.name} with {len(solns)} solns in {times:.2f}s (searched {num_progs} programs)")
            else:
                if verbose: mlb.red(f"failed to solve {task.name} (searched {num_progs} programs)")
        model_results.append(ModelResult(name, search_results, len(test_tasks)))
        if verbose: mlb.blue(f'solved {len(search_results)}/{len(test_tasks)} tasks ({len(search_results)/len(test_tasks)*100:.1f}%)\n')
    return model_results

class ModelResult:
    def __init__(self, name, search_results, num_tests):
        self.empty = (len(search_results) == 0)
        if len(search_results) > 0:
            assert isinstance(search_results[0], SearchResult)
        self.search_results = search_results
        self.num_tests = num_tests
        self.name = name
        if not self.empty:
            self.max_time = max([r.time for r in search_results])
            self.max_evals = max([r.evaluations for r in search_results])
        else:
            self.max_time = 0
            self.max_evals = 0
    def fraction_hit(self, predicate):
        valid = [r for r in self.search_results if predicate(r)]
        return len(valid)/self.num_tests*100

def plot_model_results(model_results, file=None, salt=''):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir('model_results'):
        os.mkdir('model_results')
    assert isinstance(model_results, list)
    assert isinstance(model_results[0], ModelResult)

    # plot vs time
    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('percent correct')
    plot.ylim(bottom=0., top=100.)
    x_max = min([model_result.max_time for model_result in model_results])
    for model_result in model_results:
        xs = list(np.arange(0,x_max,0.1)) # start,stop,step
        plot.plot(xs,
                [model_result.fraction_hit(lambda r: r.time < x) for x in xs],
                label=model_result.name,
                linewidth=4)
    plot.legend()

    if file:
        plot.savefig(f"plots/{file}_time.png")
        mlb.yellow(f"saved plot to plots/{file}_time.png")
    else:
        plot.show()

    # plot vs evaluations
    plot.figure()
    plot.xlabel('Evaluations')
    plot.ylabel('percent correct')
    plot.ylim(bottom=0., top=100.)
    x_max = min([model_result.max_evals for model_result in model_results])
    for model_result in model_results:
        xs = list(range(x_max))
        plot.plot(xs,
                [model_result.fraction_hit(lambda r: r.evaluations <= x) for x in xs],
                label=model_result.name,
                linewidth=4)
    plot.legend()

    if file:
        plot.savefig(f"plots/{file}_evals@{salt}.png")
        mlb.yellow(f"saved plot to experimentOutputs/{file}_evals@{salt}.png\n")
    else:
        plot.show()

    print(f"saving model_results used in plotting to model_results/{file}_model_results@{salt}")
    torch.save(model_results,f"model_results/{file}_model_results@{salt}")

def t4(state):
    """
    The model in `state` should be tested on t4 data.
    Uses `cfg.data.num_tests`, `cfg.loop.timeout`
    """
    cfg = state.cfg
    assert cfg.load is not None, "youre running T4 without loading a model to test it on"
    timeout = cfg.loop.timeout # should get overridden if fed by commandline
    print(f"Running T4 tests on {cfg.load} with timeout={timeout}")

    t4_loader = DeepcoderTaskloader(
        utils.to_absolute_path(f'dreamcoder/domains/list/DeepCoder_data/T4_A2_V512_L10_train_perm.txt'),
        allowed_requests=state.allowed_requests,
        repeat=False,
        num_tasks=None,
        )
    t4_tasks = t4_loader.getTasks(cfg.data.num_tests)
    model_results = test_models([state.astar],t4_tasks,timeout=timeout, verbose=True)
    plot_model_results(model_results, file='t4', salt=state.j)






@hydra.main(config_path="conf", config_name='config')
def hydra_main(cfg):
    if cfg.verbose:
        mlb.set_verbose()
    with torch.cuda.device(cfg.device):
        state = State()
        print_overrides = []
        if cfg.load is None:
            print("no file to load from, creating new state...")
            state.new(cfg=cfg)
        elif cfg.mode != 'plot':
            #HydraConfig.instance().set_config(cfg)
            print(f"loading from outputs/{cfg.load}...")
            state.load(
                'outputs/'+cfg.load # 2020-09-06/13-49-11/saves/autosave'
                )
            print("loaded")
            assert all(['=' in arg for arg in sys.argv[1:]])
            overrides = [arg.split('=')[0] for arg in sys.argv[1:]]
            for override in overrides:
                # eg override = 'data.T'
                dotpath = override.split('.')
                target = state.cfg # the old cfg
                source = cfg # the cfg that contains the overrides
                for attr in dotpath[:-1]: # all but the last one (which we'll use setattr on)
                    target = target[attr]
                    source = source[attr]
                overrided_val = source[dotpath[-1]]
                print_overrides.append(f'overriding {override} to {overrided_val}')
                with open_dict(target): # disable strict mode
                    target[dotpath[-1]] = overrided_val
                    
        print()
        print(OmegaConf.to_yaml(cfg))
        print(os.getcwd())
        for string in print_overrides: # just want this to print after the big wall of yaml
            mlb.purple(string)

        # big switch statement over cfg.mode
        if cfg.mode == 'resume':
            print("Entering training loop...")
            train_model(**state.as_kwargs)
        elif cfg.mode == 'test':
            model_results = test_models([state.astar],state.test_tasks,timeout=state.cfg.loop.timeout, verbose=True)
            plot_model_results(model_results, file='test', salt=state.j)
        elif cfg.mode == 'plot':
            assert isinstance(cfg.load,list)
            model_results = []
            for file in cfg.load:
                model_results.append(torch.load('outputs/'+file))
            plot_model_results(model_results)
        elif cfg.mode == 'T4':
            t4(state)
        elif cfg.mode == 'profile':
            mlb.purple('[profiling]')
            import cProfile,pstats
            from pstats import SortKey as sort
            cProfile.runctx('train_model(**state.as_kwargs)',globals(),locals(),'profiled')
            p = pstats.Stats('profiled')
            p.strip_dirs()
            p.sort_stats(sort.CUMULATIVE)
            p.reverse_order()
            p.print_stats()
            print('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
            print('tottime: doesnt include subfunctions')
            print('percall: previous column divided by num calls')
            
            raise Exception("Take a look around!")
        elif cfg.mode == 'inspect':
            print()
            print("=== Inspecting State ===")
            print(OmegaConf.to_yaml(state.cfg))
            print(os.getcwd())
            print(state)
            raise Exception("Take a look around!") # intentional Exception so you can look at `state` and debug it.
        # not really sure if this is needed
        #hydra.core.hydra_config.HydraConfig.set_config(cfg)

if __name__ == '__main__':
    hydra_main()
