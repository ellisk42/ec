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
import dreamcoder.matt.test as test
from dreamcoder.matt.state import which

def window_avg(window):
    window = list(filter(lambda x: x is not None, window))
    return sum(window)/len(window)

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
    plosses_since_print=None,
    vlosses_since_print=None,
    j=0,
    **kwargs,
        ):
    print(f"j:{j}")
    phead.featureExtractor.run_tests()

    if frontiers is None:
        frontiers = []
    time_since_print = None

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
                start = time.time()
                vloss = vhead.valueLossFromFrontier(f, g)
                ploss = phead.policyLossFromFrontier(f, g)
                elapsed = time.time() - start
                print(f'loss {ploss.item():.2f} in {elapsed:.4f}s on {f.p}')
            except InvalidSketchError:
                print(f"Ignoring training program {f._fullProg} because of out of range intermediate computation")
                continue
            loss = vloss + ploss
            loss.backward()

            # for printing later
            if cfg.loop.print_every is not None: # dont gather these if we'll never empty it by printing
                #plosses[j % loss_window] = ploss.item()
                plosses.append(ploss.item())
                #vlosses[j % loss_window] = vloss.item()
                vlosses.append(vloss.item())
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
                rate = len(plosses)/(time.time()-time_since_print) if time_since_print is not None else None
                if rate is None: rate = 0
                time_str = f" ({rate:.2f} steps/sec)"
                vloss_avg = sum(vlosses) / max([len(vlosses),1])
                ploss_avg = sum(plosses) / max([len(plosses),1])
                for head,loss in zip([vhead,phead],[vloss_avg,ploss_avg]): # important that the right things zip together (both lists ordered same way)
                    print(f"[{j}]{time_str} {head.__class__.__name__} {loss}")
                    w.add_scalar('TrainLoss/'+head.__class__.__name__, loss, j)
                print()
                w.flush()
                vlosses = []
                plosses = []
                time_since_print = time.time()

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
                        if ploss.item() == np.inf:
                            breakpoint()
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
                model_results = test.test_models([astar],
                                            validation_frontiers[: cfg.loop.search_valid_num_tasks],
                                            g,
                                            timeout=cfg.loop.search_valid_timeout,
                                            verbose=True)
                accuracy = len(model_results[0].search_results) / len(validation_frontiers[:cfg.loop.search_valid_num_tasks]) * 100
                w.add_scalar('ValidationAccuracy/'+head.__class__.__name__, accuracy, j)
                plot.plot_model_results(model_results, file='validation', w=w, j=j, tb_name=f'ValdiationAccuracy')

            # if mlb.predicate('test'): # NOT REALLY USED
            #     model_results = test_models([astar], test_tasks, g, timeout=cfg.loop.search_valid_timeout, verbose=True)
            #     plot_model_results(model_results, file='test', salt=j)

            j += 1 # increment before saving so we resume on the next iteration
            if cfg.loop.save_every is not None and (j-1) % cfg.loop.save_every == 0: # the j-1 is important for not accidentally repeating a step
                state.save(locals(),'autosave')
