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




def handle_elsewhere():
    if toplevel:
        assert w is None
        w = SummaryWriter(
            log_dir=toplevel_path(''),
            max_queue=10,
        )

    if salt is not None and salt != '':
        salt = f'_{salt}'
    else:
        salt = ''
    if j is not None:
        j_str = f'@{j}'
    else:
        j_str = ''

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
        w.flush()
        w.close()


    plot.savefig(time_file)
    mlb.yellow(f"saved plot to {printable_local_path(time_file)}")
    if toplevel:
        path = toplevel_path(time_file)
        plot.savefig(path)
        mlb.yellow(f"toplevel: saved plot to {path}")

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

    if save_model_results:
        if model_result_path is not None:
            # if our target file already exists, move it.
            res_file = model_results_path() / model_result_path
            safe_name = get_unique_path(res_file)
            res_file.rename(safe_name)
            mlb.red(f"moved old model result: {res_file.relative_to(outputs_path())} -> {safe_name.relative_to(outputs_path())}")
        else:
            res_file = model_results_path() / f"{file}{j_str}{salt}"
        print(f"saving model_results used in plotting to {res_file.relative_to(outputs_path())}")
        torch.save(model_results,res_file)

    plot.savefig(evals_file)
    mlb.yellow(f"saved plot to {evals_file.relative_to(outputs_path())}")

    time_file = plots_path() / f"{file}_time.{filetype}"
    evals_file = plots_path() / f"{file}_evals.{filetype}"



def plot_model_results(
    model_results,
    file, # cfg.plot.file
    toplevel=False, # mode=plot
    legend=None, # cfg.plot.legend after preprocessing
    cropped=False, # cfg.plot.cropped
    model_result_path=None,
    filetype='png', # cfg.plot.filetype
    title=None, # cfg.plot.title else argv[2:]
    salt=None,
    save_model_results=True, # false if mode=plot
    w=None,
    j=None,
    tb_name=None, # cfg.plot.tb_name
    xlim=None): # cfg.plot.xlim

    if isinstance(model_results, ModelResult):
        model_results = [model_results]

    if legend is not None:
        assert len(legend) == len(model_results)

    print(f'Plotting {len(model_results)} model results')



def evals_plot():

    #############
    # * EVALS * #
    #############

    font_size = 14

    plot.figure(dpi=200)
    plot.title(title, fontsize=font_size)
    plot.xlabel('Number of partial programs considered', fontsize=font_size)
    plot.ylabel('Percent correct', fontsize=font_size)
    x_max = max([m.full_xlim_evals for m in model_results])
    if xlim is not None:
        x_max = min((x_max,xlim))
        print(f'applied xlim to get new xmax of {x_max}')
    plot.ylim(bottom=0., top=100.)
    plot.xlim(left=0., right=x_max)
    for i,m in enumerate(model_results):
        label = legend[i] if legend else f'{m.cfg.job_name}.{m.cfg.run_name}'
        xs = list(range(m.earliest_failure))
        line, = plot.plot(xs,
                [m.fraction_hit(lambda r: r.evaluations <= x) for x in xs],
                label=label,
                linewidth=4)
        if label == 'DeepCoder':
            line.set_color('C5')
        if label == 'RobustFill':
            line.set_color('C4')
            line.set_zorder(0)
    plot.legend()



    print(os.getcwd())


def time_plot():

    ############
    # * TIME * #
    ############

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
            line.set_color('C5')
    plot.legend()