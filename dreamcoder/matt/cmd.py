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
import dreamcoder.matt.plot as plot
import dreamcoder.matt.state as state
import dreamcoder.matt.test as test
import dreamcoder.matt.train as train

def cmd(cfg):
    mlb.purple("Entered cmd mode (exit with ctrl-D)")
    os.chdir(outputs_path(''))
    print('chdir to outputs/')
    if not os.path.isdir('../outputs_trash'):
        os.mkdir('../outputs_trash')
    import readline # simply by importing readline we now have scrollable history in input()!!

    results = None
    while True: # exit with ctrl-D
        try:
            line = input('>>> ').strip()
        except EOFError: # ctrl-D
            return

        [cmd, *args] = line.split(' ')
        args_line = ' '.join(args)

        if cmd == 'list':
            results = outputs_regex(*args)
            for result in results:
                print(result)
            if len(results) == 0:
                print('[No matching files]')
                continue
        if cmd == 'delete':
            if len(args) == 0 and result is not None and len(result) > 0:
                print("deleting result of previous `list` command")
            else:
                results = outputs_regex(*args)
                if len(results) == 0:
                    print('[No matching files]')
                    continue
            for result in results:
                print(f"{result} -- implies -> {get_datetime_path(result)}")
            try:
                yes = input('Type `y` to confirm deletion of the above results:').strip()
            except (EOFError,KeyboardInterrupt):
                continue
            if yes != 'y':
                print("aborting...")
                continue
            for result in results:
                result = get_datetime_path(result)
                target_parent = ec_path('outputs_trash') / result.parent.name  # result.parent.name is DATE/
                target_parent.mkdir(parents=True,exist_ok=True) 
                result.rename(target_parent / result.name)
                print(f'moved {result} -> {target_parent / result.name}')
    return

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