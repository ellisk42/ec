try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import pathlib
import contextlib
import multiprocessing as mp
import shutil
import sys,os
import glob
import signal

import hydra
from hydra import utils
from omegaconf import DictConfig,OmegaConf,open_dict
import omegaconf
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import dreamcoder
import dreamcoder.domains
import dreamcoder.domains.list
import dreamcoder.domains.list.makeDeepcoderData
from dreamcoder.domains.list.makeDeepcoderData import *
from datetime import datetime

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
torch.set_num_threads(1) # or else it gets unnecessarily crazy
import numpy as np
import random

from dreamcoder.domains.list.makeDeepcoderData import *
from dreamcoder.domains.list.main import ListFeatureExtractor
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives,deepcoderPrimitivesPlusPlus
from dreamcoder.valueHead import SimpleRNNValueHead, ListREPLValueHead, BaseValueHead, SampleDummyValueHead
from dreamcoder.policyHead import RNNPolicyHead,BasePolicyHead,ListREPLPolicyHead, NeuralPolicyHead
from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from torch.utils.tensorboard import SummaryWriter
import mlb
import time
import matplotlib.pyplot as plot




# doesnt actually do anything i think
def tmux_closed():
    sys.exit(1)
signal.signal(signal.SIGHUP,tmux_closed)


class Poisoned: pass

def window_avg(window):
    window = list(filter(lambda x: x is not None, window))
    return sum(window)/len(window)

class State:
    def __init__(self):
        self.no_pickle = []
        self.cfg = None
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
        self.name = cfg.name
        if cfg.prefix is not None:
            self.name = cfg.prefix + '.' + self.name

        taskloader = DeepcoderTaskloader(
            cfg=cfg,
            mode='train'
            )
        testloader = DeepcoderTaskloader(
            cfg=cfg,
            mode='test'
            )

        extractor = ExtractorGenerator(cfg=cfg, maximumLength = max([cfg.data.train.L,cfg.data.test.L])+2)

        #taskloader.check()
        test_frontiers = testloader.getTasks()
        #taskloader.check()
        #del testloader # I dont wanna deal w saving it
        print(f'Got {len(test_frontiers)} testing tasks')
        num_valid = int(cfg.data.test.valid_frac*len(test_frontiers))
        validation_frontiers = test_frontiers[:num_valid]
        test_frontiers = test_frontiers[num_valid:]
        print(f'Split into {len(test_frontiers)} testing tasks and {len(validation_frontiers)} validation tasks')

        if cfg.data.train.expressive_lambdas:
            assert cfg.data.test.expressive_lambdas
            prims = deepcoderPrimitivesPlusPlus()
        else:
            prims = deepcoderPrimitives()

        
        g = Grammar.uniform(prims, g_lambdas = taskloader.g_lambdas)

        #taskloader.check()

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

        # TEMP
        # prog = Program.parse('(lambda (MAP (lambda (+ 1 $0)) $0))')
        # f = FakeFrontier(prog, test_tasks[0][1])
        # phead.policyLossFromFrontier(f,g)
        # for f in test_tasks:
        #     phead.policyLossFromFrontier(f,g)
        #     print(f"ran on {f._fullProg}")
        #     print()

        max_depth = 10

        params = itertools.chain.from_iterable([head.parameters() for head in heads])
        optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, eps=1e-3, amsgrad=True)

        astar = make_astar(vhead,phead,max_depth)
        j=0
        frontiers = None

        loss_window = 1
        plosses = [None]*loss_window
        vlosses = [None]*loss_window

        #taskloader.check()

        self.update(locals()) # do self.* = * for everything
        self.post_load()
    
    @contextlib.contextmanager
    def saveable(self):
        temp = {}
        for key in self.no_pickle:
            temp[key] = self[key]
            self[key] = Poisoned
        
        saveables = []
        for k,v in self.__dict__.items():
            if isinstance(v,State):
                continue # dont recurse on self
            if hasattr(v,'saveable'):
                saveables.append(v)
        try:
            with contextlib.ExitStack() as stack:
                for saveable in saveables:
                    stack.enter_context(saveable.saveable()) # these contexts stay open until ExitStack context ends
                yield None
        finally:
            for key in self.no_pickle:
                self[key] = temp[key]

    def save(self, locs, name):
        """
        use like state.save(locals(),"name_of_save")
        """
        self.update(locs)

        if not os.path.isdir('saves'):
            os.mkdir('saves')
        path = f'saves/{name}'
        print(f"saving state to {path}...")

        with self.saveable():
            torch.save(self, f'{path}.tmp')

        print('critical step, do not interrupt...')
        shutil.move(f'{path}.tmp',f'{path}')
        print("done")
        # self.taskloader.buf = q
        # self.taskloader.lock = l
        # if self.taskloader.cfg.threaded:
        #     self.taskloader.lock.release()
    def load(self, path):
        path = utils.to_absolute_path(path)
        print("torch.load")
        state = torch.load(path)
        print("self.update")
        self.update(state.__dict__)
        print("self.post_load")
        self.post_load()
        print("loaded")
    def post_load(self):
        print(f"chdir to {self.cwd}")
        os.chdir(self.cwd)
        self.init_tensorboard()
        for k,v in self.__dict__.items():
            if isinstance(v,State):
                continue # dont recurse on self
            if hasattr(v,'post_load'):
                v.post_load()
    def init_tensorboard(self):
        print("intializing tensorboard")
        self.w = SummaryWriter(
            log_dir=self.name,
            max_queue=1,
        )
        print("writer for",self.name)
        print("done")
        self.no_pickle.append('w')
    def rename(self,name):
        if self.name == name:
            return
        old_name = self.name
        self.name = name

        # shut down tensorboard since it was using the old name
        if hasattr(self,w) and self.w is not Poisoned:
            self.w.flush()
            self.w.close()
            del self.w

        # move old tensorboard files
        os.rename(old_name,self.name)
        
        self.init_tensorboard() # reboot tensorboard
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
    test_frontiers,
    validation_frontiers,
    loss_window,
    vlosses,
    plosses,
    frontiers=None,
    best_validation_loss=np.inf,
    j=0,
    **kwargs,
        ):
    print(f"j:{j}")
    tstart = None
    phead.featureExtractor.run_tests()

    if frontiers is None:
        frontiers = []

    while True:
        print(f"{len(frontiers)=}")
        if len(frontiers) == 0:
            mlb.red("reloading frontiers")
            frontiers = taskloader.getTasks()
            assert len(frontiers) > 0
        while len(frontiers) > 0: # work thru batch of `batch_size` examples
            f = frontiers.pop(0)

            if cfg.data.train.freeze:
                frontiers.append(f) # put back at the end

            # abort if reached end
            if cfg.loop.max_steps and j > cfg.loop.max_steps:
                mlb.purple(f'Exiting because reached maximum step for the above run (step: {j}, max: {cfg.loop.max_steps})')
                return
            
            # train the model
            for head in heads:
                head.train()
                head.zero_grad()
            try:
                vloss = vhead.valueLossFromFrontier(f, g)
                ploss = phead.policyLossFromFrontier(f, g)
                print(f'loss {ploss.item():.2f} on {f.p}')
            except InvalidSketchError:
                print(f"Ignoring training program {f._fullProg} because of out of range intermediate computation")
                continue
            loss = vloss + ploss
            loss.backward()

            # for printing later
            plosses[j % loss_window] = ploss.item()
            vlosses[j % loss_window] = vloss.item()
            optimizer.step()

            mlb.freezer('pause')

            if mlb.predicate('return'):
                return
            if mlb.predicate('which'):
                which(cfg)
            if mlb.predicate('rename'):
                name = input('Enter new name:')
                state.rename(name)
                # VERY important to do this:
                w = state.w
                name = state.name

            # printing and logging
            if j % cfg.loop.print_every == 0:
                if tstart is not None:
                    elapsed = time.time()-tstart
                    time_str = f" ({cfg.loop.print_every/elapsed:.1f} steps/sec)"
                else:
                    time_str = ""
                tstart = time.time()
                vloss_avg = window_avg(vlosses)
                ploss_avg = window_avg(plosses)
                for head,loss in zip([vhead,phead],[vloss_avg,ploss_avg]): # important that the right things zip together (both lists ordered same way)
                    print(f"[{j}]{time_str} {head.__class__.__name__} {loss}")
                    w.add_scalar('TrainLoss/'+head.__class__.__name__, loss, j)
                print()
                w.flush()

            # validation loss
            if cfg.loop.valid_every is not None and j % cfg.loop.valid_every == 0:
                # get valid loss
                for head in heads:
                    head.eval()
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
                    w.add_scalar('ValidationLoss/'+head.__class__.__name__, loss.item(), j)
                # save model if new record for lowest validation loss
                val_loss = (vloss+ploss).item()
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    state.save(locals(),'best_validation')
                    mlb.green('new lowest validation loss!')
                    w.add_scalar('ValidationLossBest/'+head.__class__.__name__, loss, j)

            # search on validation set
            if cfg.loop.search_valid_every is not None and j % cfg.loop.search_valid_every == 0:
                model_results = test_models([astar],
                                            validation_frontiers[: cfg.loop.search_valid_num_tasks],
                                            g,
                                            timeout=cfg.loop.search_valid_timeout,
                                            verbose=True)
                accuracy = len(model_results[0].search_results) / len(validation_frontiers[:cfg.loop.search_valid_num_tasks]) * 100
                w.add_scalar('ValidationAccuracy/'+head.__class__.__name__, accuracy, j)
                plot_model_results(model_results, file='validation', salt=j, w=w, j=j, tb_name=f'ValdiationAccuracy')

            # if mlb.predicate('test'): # NOT REALLY USED
            #     model_results = test_models([astar], test_tasks, g, timeout=cfg.loop.search_valid_timeout, verbose=True)
            #     plot_model_results(model_results, file='test', salt=j)

            j += 1 # increment before saving so we resume on the next iteration
            if cfg.loop.save_every is not None and (j-1) % cfg.loop.save_every == 0: # the j-1 is important for not accidentally repeating a step
                state.save(locals(),'autosave')
            
    #def __getstate__(self):
        #Classes can further influence how their instances are pickled; if the class defines the method __getstate__(), it is called and the returned object is pickled as the contents for the instance, instead of the contents of the instance’s dictionary. If the __getstate__() method is absent, the instance’s __dict__ is pickled as usual.
    #def __setstate__(self,state):
        #Upon unpickling, if the class defines __setstate__(), it is called with the unpickled state. In that case, there is no requirement for the state object to be a dictionary. Otherwise, the pickled state must be a dictionary and its items are assigned to the new instance’s dictionary.
        #Note If __getstate__() returns a false value, the __setstate__() method will not be called upon unpickling.


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


def test_models(astars, test_tasks, g, timeout, verbose=True):
    """
    `astars`: a list of one or more Astar objects
        These can be easily made with makeDeepcoderData.make_astar(vhead,phead,maxDepth)
    `test_tasks`: a list of Tasks or FakeFrontiers to run search on
    `g`: Grammar passed to Astar.infer()
    `timeout`: the search timeout
    """
    if len(test_tasks) > 0 and isinstance(test_tasks[0], FakeFrontier):
        test_tasks = [f.task for f in test_tasks]
    model_results = []
    for astar in astars:
        astar.owner.policyHead.eval()
        astar.owner.valueHead.eval()
        name = f"{astar.owner.policyHead.__class__.__name__}_&&_{astar.owner.valueHead.__class__.__name__}"
        print(f"Testing: {name}")
        search_results = []
        likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        for task in test_tasks:
            with torch.no_grad():
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
        model_results.append(ModelResult(name, search_results, len(test_tasks), timeout))
        if verbose: mlb.blue(f'solved {len(search_results)}/{len(test_tasks)} tasks ({len(search_results)/len(test_tasks)*100:.1f}%)\n')
    return model_results

class ModelResult:
    def __init__(self, name, search_results, num_tests, timeout):
        self.empty = (len(search_results) == 0)
        if len(search_results) > 0:
            assert isinstance(search_results[0], SearchResult)
        self.timeout = timeout
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

def plot_model_results(model_results, file, title=None, salt='', save_model_results=True, w=None, j=None, tb_name=None):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir('model_results'):
        os.mkdir('model_results')
    assert isinstance(model_results, list)
    assert isinstance(model_results[0], ModelResult)


    if title is None:
        title = file

    print(f'Plotting {len(model_results)} model results')

    # plot vs time
    plot.figure()
    plot.title(title)
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

    plot.savefig(f"plots/{file}_time.png")
    mlb.yellow(f"saved plot to plots/{file}_time.png")

    # plot vs evaluations
    plot.figure()
    plot.title(title)
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
    plot.savefig(f"plots/{file}_evals@{salt}.png")
    mlb.yellow(f"saved plot to plots/{file}_evals@{salt}.png\n")

    if w is not None:
        print("Adding figure to Tensorboard")
        assert j is not None
        assert tb_name is not None

        fig = plot.gcf() # get current figure
        w.add_figure(tb_name,fig,j)
        print("Added figure")

    if save_model_results:
        print(f"saving model_results used in plotting to model_results/{file}_{salt}")
        torch.save(model_results,f"model_results/{file}_{salt}")
    print(os.getcwd())

def which(cfg):
    if cfg is None:
        return
    print(OmegaConf.to_yaml(cfg))
    print(os.getcwd())
    timestamp = os.path.basename(os.path.dirname(os.getcwd())) + '%2F' + os.path.basename(os.getcwd())
    print(f'http://localhost:6696/#scalars&regexInput={timestamp}')
    print("curr time:",datetime.now())

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
        expressive_lambdas=cfg.data.expressive_lambdas,
        lambda_depth=cfg.data.lambda_depth,
        )
    t4_tasks = t4_loader.getTasks(cfg.data.num_tests)
    model_results = test_models([state.astar],t4_tasks, state.g, timeout=timeout, verbose=True)
    plot_model_results(model_results, file=f't4_{timeout}s', salt=state.j)

class Tests:
    def __init__(self):
        self.tests = {}
        self.tests_dir = pathlib.Path(utils.to_absolute_path('list_tests/'))
    def test(self,fn):
        self.tests[fn.__name__] = fn
tests = Tests()



@tests.test
def deepcoder(cfg):
    test_cfg = cfg.data.test

    # cfg = state.cfg
    # mlb.purple("Training data:")
    # print(OmegaConf.to_yaml(cfg.data.train))
    # mlb.purple(f"Original training data was: T{cfg.data.train.T}")
    # mlb.purple(f"Testing on T3 data")
    # with open_dict(cfg): # disable strict mode
    #     cfg.data.test = cfg.data.train # so conditions are the same as during training
    #     cfg.data.test.T = 3
    #     cfg.data.test.num_templates = cfg.data.test.buf_size = 100
    #     cfg.data.test.num_mutated_tasks = 1
    #     cfg.data.test.print_data = True
    #     cfg.data.test.repeat = False
    #     cfg.data.test.threaded = False
    taskloader = DeepcoderTaskloader(
        cfg=cfg,
        mode='test'
        )
    tasks = taskloader.getTasks()
    assert len(tasks) == cfg.data.test.num_templates
    return tasks

def cfg_diff(train_cfg,test_cfg):
    mlb.magenta("Differences between train and test:")
    for key in set(test_cfg.keys()) | set(train_cfg.keys()):
        if key in ['threaded', 'num_templates', 'valid_frac', 'buf_size', 'repeat', 'print_data']:
            continue #ignore these
        if key not in test_cfg:
            mlb.yellow(f"warn: key not in test data config: {key}")
            continue
        elif key not in train_cfg:
            mlb.yellow(f"warn: key not in train data config: {key}")
            continue
        if test_cfg[key] != train_cfg[key]:
            mlb.magenta(mlb.mk_bold(f"\t{key=} {train_cfg[key]=} {test_cfg[key]=}"))



def cleanup():
    path = utils.to_absolute_path('outputs')
    files = os.listdir(path)
    pass # TODO continue

@hydra.main(config_path="conf", config_name='config')
def hydra_main(cfg):
    if cfg.debug.verbose:
        mlb.set_verbose()

    np.seterr(all='raise') # so we actually get errors when overflows and zero divisions happen
    cleanup()

    state = State()

    if cfg.print and cfg.load is None:
        which(cfg)
        print("cfg.print was specified, exiting")
        return

    def on_crash():
        print(os.getcwd())
        # if hasattr(state,'taskloader') and hasattr(state.taskloader, 'lock'):
        #     print("acquiring lock...")
        #     state.taskloader.lock.acquire() # force the other thread to block
        #     #state.taskloader.p.kill()
        #     state.taskloader.lock.release()
        #     print("done")
    #def on_ctrlc():
        #on_crash()
        #print('exiting')
        #sys.exit(1)
         
    with mlb.debug(debug=cfg.debug.mlb_debug, ctrlc=on_crash, crash=on_crash):

        # PLOT
        if cfg.mode == 'plot':
            #assert isinstance(cfg.load, omegaconf.listconfig.ListConfig)
            assert isinstance(cfg.load,str)
            model_results = []
            for file in cfg.load.split(','):
                file = file.strip()
                if file == '':
                    continue
                model_results.extend(torch.load(utils.to_absolute_path('outputs/'+file)))
            plot_model_results(model_results, file=cfg.plot.file, title=cfg.plot.title, save_model_results=False)
            return
        
        # TEST
        elif cfg.mode == 'test':
            original_cfg = None
            assert cfg.test.to_file or cfg.load, "Doesnt make sense to generate data and neither save it nor test it on a loaded state"
            tests_from = cfg.test.from_fn or cfg.test.from_file # use fancy python `or` semantics
            if cfg.test.from_fn is not None:
                if cfg.test.from_fn not in tests.tests:
                    mlb.red(f"from_fn value not recognized. options are: {list(tests.tests.keys())}")
                    return
                test_frontiers = tests.tests[cfg.test.from_fn](cfg)
                mlb.purple(f"got {len(test_frontiers)} test frontiers from {cfg.test.from_fn}()")
                if cfg.test.to_file is not None:
                    print(f"Writing saved tests to {cfg.test.to_file}...")
                    torch.save((test_frontiers,cfg), tests.tests_dir / cfg.test.to_file)
            elif cfg.test.from_file is not None:
                (test_frontiers,original_cfg) = torch.load(tests.tests_dir / cfg.test.from_file)
                # note that original_cfg is just around in case you ever want a record of how the tests were created!
                tests_from = cfg.test.from_file
                mlb.purple(f"loaded {len(test_frontiers)} test frontiers from {cfg.test.from_file} (details in `original_cfg`)")
            else:
                raise ValueError("Specify either test.from_file or test.from_fn")
            assert isinstance(test_frontiers,list) and len(test_frontiers) > 0
            if cfg.load is None:
                print("no state specified to load, exiting")
                return
            ### NOTE: this continues at a later 'test' section
        
        # CMD
        elif cfg.mode == 'cmd':
            mlb.purple("Entered cmd mode")
            os.chdir(utils.to_absolute_path(f'outputs/'))
            print('chdir to outputs/')
            if not os.path.isdir('../outputs_trash'):
                os.mkdir('../outputs_trash')
            import readline # simply by importing readline we now have scrollable history in input()!!
            results = None
            while True: # exit with ctrl-D
                try:
                    line = input('>>> ').strip()
                except EOFError:
                    return
                [cmd, *args] = line.split(' ')
                args_line = ' '.join(args)

                def process(tb):
                    """
                    Process a tensorboard directory to get all its events. Doesn't work recursively from higher directories (that wouldnt really make sense)
                    """
                    try:
                        tboard = EventAccumulator(tb)
                    except Exception as e:
                        print("unable to load file {tb}: {e}")
                        return
                    tboard.Reload() # make sure data is all loaded
                    ret = {}
                    scalars = tboard.Tags()['scalars']
                    for scalar in scalars: # eg scalar = 'SampleDummyValueHead'
                        events = tboard.Scalars(scalar)
                        ret['scalar'] = events
                    return ret


                def glob_all(args):
                    # first glob the results
                    results = []
                    for arg in args:
                        results.extend(glob.glob(f'**/*{arg}*',recursive=True))
                    results = sorted(results)

                    # then filter using predicates
                    for predicate in [arg for arg in args if '=' in arg]:
                        lhs,rhs = predicate.split('=')
                        # TODO WAIT FIRST JUST FOLLOW THIS https://github.com/tensorflow/tensorboard/issues/785
                        # idk it might be better.
                        # TODO first navigate to the actual folder that the tb files are in bc thats 
                        # what process() should take as input (e.g. 'tb' or whatever prefix+name is)
                        process(result)
                        raise NotImplementedError

                    return results

                if cmd == 'list':
                    results = glob_all(args)
                    for result in results:
                        print(result)
                if cmd == 'delete':
                    if len(args) == 0 and result is not None and len(result) > 0:
                        print("deleting result of previous `list` command")
                    else:
                        results = glob_all(args)
                        if len(results) == 0:
                            print('glob returned no files to delete')
                            continue

                    for result in results:
                        dir = os.path.dirname(result)
                        dir = utils.to_absolute_path(f'outputs_trash/{dir}')
                        os.makedirs(dir,exist_ok=True)
                        os.rename(result,dir+'/'+os.path.basename(result))
                        print(f'moved {result} -> {dir}')
            return

        # LOAD OR NEW
        print_overrides = []
        if cfg.load is None:
            print("no file to load from, creating new state...")
            with torch.cuda.device(cfg.device):
                state.new(cfg=cfg)
        else:
            print(f"loading from outputs/{cfg.load}...")
            state.load(
                'outputs/'+cfg.load # 2020-09-06/13-49-11/saves/autosave'
                )
            if cfg.mode == 'device':
                mlb.green(f"DEVICE: {state.cfg.device}")
                return
            print("loaded")
            assert all(['=' in arg for arg in sys.argv[1:]])
            overrides = [arg.split('=')[0] for arg in sys.argv[1:]]
            for override in overrides:
                try:
                    # eg override = 'data.T'
                    dotpath = override.split('.')
                    if dotpath[-1] == 'device':
                        raise NotImplementedError
                    target = state.cfg # the old cfg
                    source = cfg # the cfg that contains the overrides
                    for attr in dotpath[:-1]: # all but the last one (which we'll use setattr on)
                        target = target[attr]
                        source = source[attr]
                    overrided_val = source[dotpath[-1]]
                    print_overrides.append(f'overriding {override} to {overrided_val}')
                    with open_dict(target): # disable strict mode
                        target[dotpath[-1]] = overrided_val
                except Exception as e:
                    mlb.red(e)
                    pass
        print()
        which(state.cfg)

        for string in print_overrides: # just want this to print after the big wall of yaml
            mlb.purple(string)
            
        with torch.cuda.device(state.cfg.device):
            if state.cfg is not None and state.cfg.seed is not None:
                print("Setting evaluation to deterministic (roughly) and seeding RNG")
                torch.manual_seed(state.cfg.seed)
                # warning: these may slow down your model
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                np.random.seed(state.cfg.seed)
                random.seed(state.cfg.seed)

            if cfg.print:
                which(state.cfg)
                print("cfg.print was specified, exiting")
                return
            
            mlb.yellow("===START===")


            # TRAIN
            if cfg.mode == 'resume':
                print("Entering training loop...")
                train_model(**state.as_kwargs)

            # TEST
            elif cfg.mode == 'test':
                ### NOTE: this continues from the earlier 'test' section
                if cfg.test.from_fn == 'deepcoder' or (original_cfg is not None and original_cfg.test.from_fn == 'deepcoder'):
                    cfg_diff(state.cfg.data.train,original_cfg.data.test) # print the differences
                mlb.purple("Running tests")
                model_results = test_models([state.astar],
                                            test_frontiers,
                                            state.g,
                                            timeout=cfg.test.timeout,
                                            verbose=True)
                mlb.purple("plotting results")
                plot_model_results(model_results, file=f'{tests_from}_{cfg.test.timeout}s')

            # PROFILE
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

            # INSPECT
            elif cfg.mode == 'inspect':
                print()
                print("=== Inspecting State ===")
                which(state.cfg)
                print(state)
                raise Exception("Take a look around!") # intentional Exception so you can look at `state` and debug it.
            else:
                raise Exception("Mode not recognized:", cfg.mode)
        # not really sure if this is needed
        #hydra.core.hydra_config.HydraConfig.set_config(cfg)
        mlb.yellow("===END===")
        which(state.cfg)

if __name__ == '__main__':
    hydra_main()
