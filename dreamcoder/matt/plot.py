from dreamcoder.matt.util import *
from collections import defaultdict
import pathlib

import torch
import numpy as np
import random

import mlb
import dreamcoder.matt.fix as fix
import matplotlib.pyplot as plot

from dreamcoder.matt.sing import sing

from copy import deepcopy

class SearchTry:
  def __init__(self, time, nodes_expanded, soln):
    self.hit = soln is not None
    self.soln = soln
    self.time = time
    self.nodes_expanded = nodes_expanded


class ModelResult:
    def __init__(self,
                 search_tries,
                 timeout,
                 ):
        self.cfg = deepcopy(sing.cfg) # so we know exactly what produced this
        self.timeout=timeout
        self.search_tries = search_tries
        self.hits = [t for t in search_tries if t.hit]
        self.fails = [t for t in search_tries if not t.hit]
        self.num_tests = len(self.search_tries)
        assert self.num_tests > 0

        # if you crop at this point youll show the full graph, or actually even more
        self.full_xlim_evals = max([t.nodes_expanded for t in self.search_tries])
        self.full_xlim_time  = max([t.time for t in self.search_tries])

        if len(self.fails) > 0:
            # if you stop the x axis at this point you wont miss out on anything
            self.cropped_xlim_evals = min([t.nodes_expanded for t in self.fails])
            self.cropped_xlim_time  = min([t.time for t in self.fails])
        else:
            # show entire line, hide nothing!
            self.cropped_xlim_evals = max([t.nodes_expanded for t in self.hits])
            self.cropped_xlim_time  = max([t.time for t in self.hits])
    def accuracy(self):
        return len(self.hits) / self.num_tests * 100
    def fraction_hit(self, predicate):
        # returns % of total (hit + fail) of searchtries that are hits and ALSO satisfy predicate(searchtry)
        return len([t for t in self.hits if predicate(t)]) / self.num_tests * 100
    def print_dist(self):
        dist = defaultdict(int)
        for t in self.hits:
            raise NotImplementedError # impl get_depth for pnodes
            t,d,astar = get_depth(t.soln)
            dist[(t,d)] += 1
        print(f"{self.prefix}.{self.name} distribution of {len(self.search_results)} solutions:")
        for k,v in dist.items():
            print(f"T{k[0]}d{k[1]}: {v}")

def plot_model_results(model_results, file, toplevel=False, legend=None, cropped=False, model_result_path=None, filetype='png', title=None, salt=None, save_model_results=True, w=None, j=None, tb_name=None, xlim=None):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir('model_results'):
        os.mkdir('model_results')
    assert isinstance(model_results, list)
    assert isinstance(model_results[0], ModelResult)

    if legend is not None:
        assert len(legend) == len(model_results)

    if toplevel:
        assert w is None
        w = SummaryWriter(
            log_dir=toplevel_path(''),
            max_queue=10,
        )


    for i,m in enumerate(model_results):
        if not hasattr(m, 'prefix'):
            model_results[i] = fix.fix_model_result(m)

    if title is None:
        title = file
    
    prefixes = [m.prefix for m in model_results]
    shared_prefix =  all([p == prefixes[0] for p in prefixes]) # all results have same prefix

    if shared_prefix:
        if file is None:
            file = prefixes[0]
        if title is None:
            title = prefixes[0]

    if salt is not None and salt != '':
        salt = f'_{salt}'
    else:
        salt = ''
    if j is not None:
        j_str = f'@{j}'
    else:
        j_str = ''

    time_file = f"plots/{file}_time.{filetype}"
    evals_file = f"plots/{file}_evals{j_str}{salt}.{filetype}"

    if model_result_path is not None:
        model_results_file = f"model_results/{model_result_path}"
        p = pathlib.Path(model_results_file)
        if p.exists():
            move_to = p
            i=0
            while move_to.exists():
                move_to = pathlib.Path(f'{p}.{i}')
                i += 1
            p.rename(move_to)
            mlb.red(f"moved old model result: {p} -> {move_to}")
    else:
        model_results_file = f"model_results/{file}{j_str}{salt}"

    print(f'Plotting {len(model_results)} model results')

    # plot vs time
    plot.figure()
    plot.title(title)
    plot.xlabel('Time (s)')
    plot.ylabel('Percent Solved')
    x_max = max([m.full_xlim_time for m in model_results])
    plot.ylim(bottom=0., top=100.)
    plot.xlim(left=0., right=x_max)
    for m in model_results:
        label = m.name if shared_prefix else m.prefix + '.' + m.name
        xs = list(np.arange(0,m.timeout,0.1)) # start,stop,step
        line, = plot.plot(xs,
                [m.fraction_hit(lambda r: r.time < x) for x in xs],
                label=label,
                linewidth=4)
        if label == 'DeepCoder':
            print("fired")
            line.set_color('C5')
    plot.legend()

    plot.savefig(time_file)
    mlb.yellow(f"saved plot to {printable_local_path(time_file)}")
    if toplevel:
        path = toplevel_path(time_file)
        plot.savefig(path)
        mlb.yellow(f"toplevel: saved plot to {path}")

    # plot vs evaluations
    plot.figure(dpi=200)
    plot.title(title, fontsize=14)
    plot.xlabel('Number of partial programs considered', fontsize=14)
    plot.ylabel('Percent correct', fontsize=14)
    x_max = max([m.full_xlim_evals for m in model_results])
    if xlim is not None:
        x_max = min((x_max,xlim))
        print(f'applied xlim to get new xmax of {x_max}')
    plot.ylim(bottom=0., top=100.)
    plot.xlim(left=0., right=x_max)
    for i,m in enumerate(model_results):
        if legend is None:
            label = m.name if shared_prefix else m.prefix + '.' + m.name
        else:
            label = legend[i]
        xs = list(range(m.earliest_failure))
        line, = plot.plot(xs,
                [m.fraction_hit(lambda r: r.evaluations <= x) for x in xs],
                label=label,
                linewidth=4)
        if label == 'DeepCoder':
            print("fired")
            line.set_color('C5')
        if label == 'RobustFill':
            print("fired")
            line.set_color('C4')
            line.set_zorder(0)
    plot.legend()
    plot.savefig(evals_file)
    mlb.yellow(f"saved plot to {printable_local_path(evals_file)}")

    if w is not None:
        if j is None:
            j=0
        if tb_name is None:
            tb_name = title


        if not cropped:
            fig = plot.gcf() # get current figure
            w.add_figure(tb_name,fig,j)
            print(f"Added figure to tensorboard: {tb_name}")


    if toplevel:
        path = toplevel_path(evals_file)
        plot.savefig(path)
        mlb.yellow(f"toplevel: saved plot to {path}")
        # cropped version
        right = min([m.cropped_xlim_evals for m in model_results]) 
        if xlim is not None:
            right = min((x_max,xlim))
            print(f'applied xlim to get new xmax of {right}')
        plot.xlim(left=0., right=right)
        path = str(toplevel_path(evals_file)).replace(f'.{filetype}',f'_cropped.{filetype}')
        plot.savefig(path)
        mlb.yellow(f"toplevel: saved plot to {path}")
        if cropped:
            fig = plot.gcf() # get current figure
            w.add_figure(tb_name,fig,j)
            print(f"Added cropped figure to tensorboard: {tb_name}")


    
    if toplevel:
        w.flush()
        w.close()

    if save_model_results:
        print(f"saving model_results used in plotting to {printable_local_path(model_results_file)}")
        torch.save(model_results,model_results_file)
    print(os.getcwd())
