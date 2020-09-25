from dreamcoder.matt.util import *
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
from dreamcoder.valueHead import SimpleRNNValueHead, ListREPLValueHead, BaseValueHead, SampleDummyValueHead
from dreamcoder.policyHead import RNNPolicyHead,BasePolicyHead,ListREPLPolicyHead, NeuralPolicyHead
from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from torch.utils.tensorboard import SummaryWriter
import mlb
import time
import dreamcoder.matt.cmd as cmd
import dreamcoder.matt.plot as plot
import dreamcoder.matt.state as state
import dreamcoder.matt.train as train

def test_models(astars, test_tasks, g, timeout, verbose=True):
    """
    `astars`: a list of one or more Astar objects
        These can be easily made with makeDeepcoderData.make_solver('astar',vhead,phead,maxDepth)
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
        #name = f"{astar.owner.policyHead.__class__.__name__}_&&_{astar.owner.valueHead.__class__.__name__}"
        name = astar.owner.policyHead.cfg.name
        prefix = astar.owner.policyHead.cfg.prefix
        print(f"Testing: {name}")
        search_results = []
        search_failures = []
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
                if verbose:
                    mlb.green(f"solved {task.name} with {len(solns)} solns in {times:.2f}s (searched {num_progs} programs)")
                    t,d,s = get_depth(solns[0].program)
                    print(f"\t-> [T{t}d{d}s{s}] {solns[0].program}")
            else:
                if verbose: mlb.red(f"failed to solve {task.name} (searched {num_progs} programs)")
                search_failures.append(num_progs)
        model_results.append(plot.ModelResult(prefix=prefix,name=name, cfg=astar.owner.policyHead.cfg, search_results=search_results, search_failures=search_failures, timeout=timeout))
        if verbose: mlb.blue(f'solved {len(search_results)}/{len(test_tasks)} tasks ({len(search_results)/len(test_tasks)*100:.1f}%)\n')
    return model_results

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
    taskloader = DeepcoderTaskloader(
        cfg=cfg,
        mode='test'
        )
    tasks = taskloader.getTasks()
    assert len(tasks) == cfg.data.test.num_templates
    return tasks

def joshTasks(w):
    """
    From https://github.com/ellisk42/ec/blob/Josh/dreamcoder/domains/list/makeListTasks.py
    """
    ts = []
    import json
    if w == "1":
        directory = "data/wave1"
    elif w == "2":
        directory = "data/wave2"
    elif w == "3":
        directory = "data/wave3/json"
    elif w == "3.1":
        directory = "data/wave3.1/json"
    elif w == "final":
        directory = "data/final_wave"
    else:
        assert False
    directory = utils.to_absolute_path(directory)
    for fn in os.listdir(directory):
        if not fn.endswith(".json"):continue

        if w == "final":
            if not (fn.endswith("_1.json")):
                continue

        with open(f"{directory}/{fn}") as handle:
            data = json.load(handle)

            ts.append(Task(data.get("name",fn.split(".")[0][1:]),
                           arrow(tlist(tint),tlist(tint)),
                           [((e["i"],),e["o"])
                            for e in data["data"] ]))
    return list(sorted(ts,key=lambda t: t.name))

@tests.test
def josh(cfg):
    tasks = joshTasks(str(cfg.test.josh.wave))
    frontiers = [FakeFrontier(None,task) for task in tasks]
    return frontiers

@tests.test
def lucas(cfg):
    from dreamcoder.domains.list.main import retrieveJSONTasks, sortBootstrap, make_list_bootstrap_tasks
    def get_tasks(f):
        return retrieveJSONTasks(utils.to_absolute_path(f))
    if cfg.test.lucas.version == 1:
        tasks = get_tasks("data/list_tasks2.json")[:105]
    elif cfg.test.lucas.version == 2:
        tasks = get_tasks("data/list_tasks2.json")[:4928]
    elif cfg.test.lucas.version == 3:
        tasks = get_tasks("data/list_tasks2.json")
    elif cfg.test.lucas.version == 'old':
        tasks = get_tasks("data/list_tasks.json") + sortBootstrap()
    elif cfg.test.lucas.version == 'boopstrap':
        tasks = make_list_bootstrap_tasks()
    else:
        raise ValueError
    frontiers = [FakeFrontier(None,task) for task in tasks]
    return frontiers

# def analyze_tasks(tasks):
#     requests = defaultdict(int)
#     for task in tasks:
#         task.request

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

