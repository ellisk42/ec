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
from collections import defaultdict
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
import numpy as np
import random

from dreamcoder.domains.list.main import ListFeatureExtractor
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives,deepcoderPrimitivesPlusPlus
from dreamcoder.valueHead import SimpleRNNValueHead, ListREPLValueHead, BaseValueHead, SampleDummyValueHead, InvalidIntermediatesValueHead
from dreamcoder.policyHead import RNNPolicyHead,BasePolicyHead,ListREPLPolicyHead, NeuralPolicyHead
from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from torch.utils.tensorboard import SummaryWriter
import mlb
import time



import dreamcoder.matt.cmd as cmd
import dreamcoder.matt.plot as plot
from dreamcoder.matt.state import State
import dreamcoder.matt.test as test
import dreamcoder.matt.train as train
import dreamcoder.matt.fix as fix
from dreamcoder.matt.util import *

# IMPORTANT to fix loading from pickles these imports need to be here
from dreamcoder.domains.list.main import ExtractorGenerator
from dreamcoder.matt.state import Poisoned
from dreamcoder.matt.plot import ModelResult

torch.set_num_threads(1) # or else it gets unnecessarily crazy


# doesnt actually do anything i think
def tmux_closed():
    sys.exit(1)
signal.signal(signal.SIGHUP,tmux_closed)

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
    print()


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
            if '___' in cfg.load:
                cfg.load = cfg.load.replace('___',' ')
            paths = outputs_regex(*cfg.load.split(' '))
            # path must at least be DATE/TIME, possibly DATE/TIME/...
            paths = [p for p in paths if len(p.parts) >= 2]

            print("Initial path regex results:")
            for p in paths:
                print(f'\t{p}')

            for i,path in enumerate(paths):
                if 'model_results' not in path.parts:
                    if cfg.plot.suffix is None:
                        mlb.red(f'No plot.suffix provided and the regexed path doesnt contain model_results: {path}')
                        sys.exit(1)
                    paths[i] = get_datetime_path(path) / 'model_results' / cfg.plot.suffix
                    print(f"Path {i} converted to {paths[i]}")

            print("Plotting these paths:")
            for p in paths:
                print(f'\t{p}')

            model_results = []
            for path in paths:
                model_results.extend(torch.load(path))
            plot.plot_model_results(model_results,
                                    file=cfg.plot.file,
                                    title=cfg.plot.title,
                                    toplevel=True,
                                    save_model_results=False)
            return
        
        # TEST
        elif cfg.mode == 'test':
            original_cfg = None
            tests_from = cfg.test.from_fn or cfg.test.from_file # use fancy python `or` semantics
            if cfg.test.from_fn is not None:
                if cfg.test.from_fn not in test.tests.tests:
                    mlb.red(f"from_fn value not recognized. options are: {list(tests.tests.keys())}")
                    return
                test_frontiers = test.tests.tests[cfg.test.from_fn](cfg)
                test_frontiers = preprocess(test_frontiers,cfg)
                mlb.purple(f"got {len(test_frontiers)} test frontiers from {cfg.test.from_fn}()")
                if cfg.test.to_file is not None:
                    print(f"Writing saved tests to {cfg.test.to_file}...")
                    torch.save((test_frontiers,cfg), test.tests.tests_dir / cfg.test.to_file)
            elif cfg.test.from_file is not None:
                (test_frontiers,original_cfg) = torch.load(test.tests.tests_dir / cfg.test.from_file)
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
            cmd.cmd(cfg)

        # LOAD OR NEW
        print_overrides = []
        state = State()
        if cfg.load is None:
            print("no file to load from, creating new state...")
            with torch.cuda.device(cfg.device):
                state.new(cfg=cfg)
        else:

            paths = outputs_regex(cfg.load)
            # path must at least be DATE/TIME, possibly DATE/TIME/...
            paths = [p for p in paths if len(p.parts) >= 2]
            if len(paths) != 1:
                mlb.red(f'load regex `{cfg.load}` yielded multiple possible files:')
                for path in paths:
                    mlb.red(f'\t{path}')
                sys.exit(1)
            [path] = paths

            if 'saves' not in path.parts:
                path = get_datetime_path(path) / 'saves' / 'autosave'
            
            assert all(['=' in arg for arg in sys.argv[1:]])
            overrides = [arg.split('=')[0] for arg in sys.argv[1:]]

            device = None
            if 'device' in overrides:
                device=cfg.device # override the device

            mlb.green(f"loading from {path}...")
            state.load(path, device=device)

            #mlb.purple(f"It looks like the device of the model is {state.phead.output[0].weight.device}")

            if cfg.mode == 'device':
                mlb.green(f"DEVICE: {state.cfg.device}")
            print("loaded")
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
                train.train_model(**state.as_kwargs)

            # TEST
            elif cfg.mode == 'test':
                if cfg.test.model_result_path is None:
                    mlb.red("please specify test.model_result_path")
                    sys.exit(1)
                ### NOTE: this continues from the earlier 'test' section
                if cfg.test.from_fn == 'deepcoder' or (original_cfg is not None and original_cfg.test.from_fn == 'deepcoder'):
                    test.cfg_diff(state.cfg.data.train,original_cfg.data.test) # print the differences
                if cfg.test.max_tasks is not None and len(test_frontiers) > cfg.test.max_tasks:
                    mlb.red(f"Cutting down test frontiers from {len(test_frontiers)} to {cfg.test.max_tasks}")
                    test_frontiers = test_frontiers[:cfg.test.max_tasks]
                vhead = InvalidIntermediatesValueHead(cfg) if cfg.test.validator_vhead else SampleDummyValueHead()
                solver = make_solver(cfg.data.test.solver,vhead,state.phead,cfg.data.test.max_depth)
                mlb.purple("Running tests")
                model_results = test.test_models([solver],
                                            test_frontiers,
                                            state.g,
                                            timeout=cfg.test.timeout,
                                            verbose=True)
                mlb.purple("plotting results")
                plot.plot_model_results(model_results,
                                        model_result_path=cfg.test.model_result_path,
                                        file=f'{tests_from}_{cfg.test.timeout}s'
                                        )

            # PROFILE
            elif cfg.mode == 'profile':
                mlb.purple('[profiling]')
                import cProfile,pstats
                from pstats import SortKey as sort
                cProfile.runctx('train.train_model(**state.as_kwargs)',globals(),locals(),'profiled')
                p = pstats.Stats('profiled')
                p.strip_dirs()
                p.sort_stats(sort.TIME)
                p.reverse_order()
                p.print_stats()
                print('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
                print('tottime: doesnt include subfunctions')
                print('percall: previous column divided by num calls')
                breakpoint()

            # INSPECT
            elif cfg.mode == 'inspect':
                print()
                print("=== Inspecting State ===")
                which(state.cfg)
                #print(state)
                breakpoint()
                raise Exception("take a look around")
            else:
                raise Exception("Mode not recognized:", cfg.mode)
        # not really sure if this is needed
        #hydra.core.hydra_config.HydraConfig.set_config(cfg)
        mlb.yellow("===END===")
        which(state.cfg)

if __name__ == '__main__':
    hydra_main()
