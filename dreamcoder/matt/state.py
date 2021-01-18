from dreamcoder.em import ExecutionModule
from dreamcoder.matt.util import *
from collections import defaultdict
import pathlib
import contextlib
import multiprocessing as mp
import shutil
import sys,os
import glob
import signal
import dreamcoder.matt.sing as sing

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

from dreamcoder.domains.list.main import ListFeatureExtractor, ExtractorGenerator
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives,deepcoderPrimitivesPlusPlus
from dreamcoder.valueHead import SimpleRNNValueHead, ListREPLValueHead, BaseValueHead, SampleDummyValueHead
from dreamcoder.policyHead import RNNPolicyHead,BasePolicyHead,ListREPLPolicyHead, NeuralPolicyHead, DeepcoderListPolicyHead
from dreamcoder.Astar import Astar
# from likelihoodModel import AllOrNothingLikelihoodModel
from torch.utils.tensorboard import SummaryWriter
import mlb
import time

from dreamcoder.matt.syntax_robustfill import get_robustfill



# import dreamcoder.matt.cmd as cmd
# import dreamcoder.matt.plot as plot
# import dreamcoder.matt.test as test
# import dreamcoder.matt.train as train

class Poisoned: pass

class State:
    def __init__(self):
        self.no_pickle = []
        self.cfg = None
    # these @properties are kinda important. Putting them in __init sometimes gives weird behavior after loading
    @property
    def state(self):
        return self
    @property
    def self(self):
        return self
    @property
    def as_kwargs(self):
        kwargs = {'state': self.state}
        kwargs.update(self.__dict__)
        return kwargs
    def new(self,cfg):
        self.cwd = os.getcwd()
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

        if cfg.data.train.expressive_lambdas:
            assert cfg.data.test.expressive_lambdas
            prims = deepcoderPrimitivesPlusPlus()
        else:
            prims = deepcoderPrimitives()

        sing.cfg = cfg
        sing.num_exs = cfg.data.train.N
        g = Grammar.uniform(prims, g_lambdas = taskloader.g_lambdas)
        extractor = ExtractorGenerator(cfg=cfg, maximumLength = max([taskloader.L_big,testloader.L_big])+2)
        sing.em = ExecutionModule(cfg,g,extractor(0),1)

        #taskloader.check()
        test_frontiers = testloader.getTasks()
        #taskloader.check()
        #del testloader # I dont wanna deal w saving it
        print(f'Got {len(test_frontiers)} testing tasks')
        num_valid = int(cfg.data.test.valid_frac*len(test_frontiers))
        validation_frontiers = test_frontiers[:num_valid]
        test_frontiers = test_frontiers[num_valid:]
        print(f'Split into {len(test_frontiers)} testing tasks and {len(validation_frontiers)} validation tasks')


        

        #taskloader.check()

        if cfg.model.policy:
            phead = {
                'rnn': RNNPolicyHead,
                'repl': ListREPLPolicyHead,
                'dc': DeepcoderListPolicyHead,
                'rb': get_robustfill,
            }[cfg.model.type](cfg=cfg, g=g)
        else:
            phead = BasePolicyHead(cfg)
        if cfg.model.value:
            if cfg.model.tied:
                assert cfg.model.policy
                vhead = phead.vhead
            else:
                vhead = {
                    'rnn': SimpleRNNValueHead,
                    'repl': ListREPLValueHead,
                }[cfg.model.type](cfg=cfg, g=g)
        else:
            vhead = SampleDummyValueHead()

        if hasattr(phead,'vhead'):
            vhead = phead.vhead.validator_vhead
        heads = [vhead,phead]


        if cfg.cuda:
            heads = [h.cuda() for h in heads]
            sing.em.cuda()

        # TEMP
        # prog = Program.parse('(lambda (MAP (lambda (+ 1 $0)) $0))')
        # f = FakeFrontier(prog, test_tasks[0][1])
        # phead.policyLossFromFrontier(f,g)
        # for f in test_tasks:
        #     phead.policyLossFromFrontier(f,g)
        #     print(f"ran on {f._fullProg}")
        #     print()
        sing.vhead = vhead
        sing.phead = phead
        sing.heads = [vhead,phead]

        params = itertools.chain.from_iterable([head.parameters() for head in heads] + [sing.em.parameters()])
        optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, eps=1e-3, amsgrad=True)

        vhead = InvalidIntermediatesValueHead(cfg)
        astar = make_solver(cfg.data.test.solver,vhead,phead,cfg.data.train.max_depth)
        j=0
        frontiers = None

        loss_window = 1
        plosses = []
        vlosses = []

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
    def load(self, path, device=None):
        if device is not None:
            device = torch.device(device)
        path = utils.to_absolute_path(path)
        print("torch.load")
        state = torch.load(path, map_location=device)
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
        if hasattr(self,'w') and self.w is not Poisoned:
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

