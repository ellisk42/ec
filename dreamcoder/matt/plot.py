from dreamcoder.matt.util import *
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from dreamcoder.matt.sing import sing
from copy import deepcopy
import dill

class SearchTry:
  def __init__(self, time, nodes_expanded, soln):
    self.hit = soln is not None
    self.soln = soln
    if soln is not None:
        soln.reset_cache()
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
        self.full_x_max_evals = max([t.nodes_expanded for t in self.search_tries])
        self.full_x_max_time  = max([t.time for t in self.search_tries])

        if len(self.fails) == 0:
            # rare case where it never fails so the cropped graph is same as the full graph
            self.cropped_x_max_evals = self.full_x_max_evals
            self.cropped_x_max_time  = self.full_x_max_time
        else:
            # if you crop the x axis to this point it will be fair. If you go beyond this point
            # then there will have been a SearchTry that failed after fewer nodes than this so
            # it'll be a bit misleading
            self.cropped_x_max_evals = min([t.nodes_expanded for t in self.fails])
            self.cropped_x_max_time  = min([t.time for t in self.fails])
    def accuracy(self):
        return len(self.hits) / self.num_tests * 100
    def percent(self, predicate):
        return len([t for t in self.search_tries if predicate(t)]) / self.num_tests * 100
    def print_dist(self):
        dist = defaultdict(int)
        for t in self.hits:
            raise NotImplementedError # impl get_depth for pnodes
            t,d,astar = get_depth(t.soln)
            dist[(t,d)] += 1
        print(f"{self.prefix}.{self.name} distribution of {len(self.search_results)} solutions:")
        for k,v in dist.items():
            print(f"T{k[0]}d{k[1]}: {v}")
    def save(self, file):
        # if our target file already exists, move it. We never wanna accidentally overwrite this file
        path = with_ext(model_results_path() / file, 'res')
        move_existing(path)
        torch.save(self,path, pickle_module=dill)
        print(f"saved model result to {path.relative_to(outputs_path())}")

def load_model_results(load):
    """
    load :: str | [str]
        * will filter anything thats not '.res' out
        * will maintain the ordering of the list of load regexes in case you're relying on that eg in order to make it align with a plot legend
    """
    paths = path_search(train_path(), load, sort=False, ext='res')
    model_results = [torch.load(p,pickle_module=dill) for p in paths]
    print(f"Loaded {len(paths)} model results:")
    for p,m in zip(paths,model_results):

        print(f"\t{m.cfg.job_name}.{m.cfg.run_name} from {p}")
    return model_results

def main():

    plot = sing.cfg.plot

    model_results = load_model_results(sing.cfg.load)
    print(f'Plotting {len(model_results)} model results')

    if plot.file is None:
        plot.file = f'{sing.cfg.start_time_filename}.{plot.filetype}'
    if plot.title is None:
        plot.title = f'{plot.file}\nload={sing.cfg.load}'
    
    fig = evals_plot(
        model_results,
        title=plot.title,
        cropped=plot.cropped,
        legend=plot.legend,
        font_size=plot.font_size,
        linewidth=plot.linewidth,
        )
    
    # Do any modifications to the plot

    [ax] = fig.axes
    lines = ax.get_lines()
    colors = {
        'blended':'C0',
        'neural':'C1',
        'rnn':'C2',
        'robustfill':'C4',
        'deepcoder':'C5',
    }
    if plot.iclr_paper:
        for line in lines:
            label = line.get_label().lower().strip()
            if label in colors:
                line.set_color(colors[label])
            if label == 'robustfill':
                line.set_zorder(0)

    if plot.x_max:
        (xmin,xmax) = ax.get_xlim()
        fig.xlim(left=xmin, right=plot.x_max)

    # Add plot to tensorboard
    w = SummaryWriter(log_dir=toplevel_plots_path())
    w.add_figure(plot.file, fig)
    w.flush()
    w.close()
    green(f"Added plot to tensorboard at http://localhost:6696/#images&tagFilter={plot.file}&regexInput=toplevel")

    # Save plot to file
    path = toplevel_plots_path() / plot.file
    fig.savefig(path, dpi=200)
    green(f"saved fig to {path.relative_to(toplevel_path())}")

    # Save args used to generate plot to file
    with open(f'{path}.argv','w') as f:
        f.write(' '.join(sys.argv))
    print(f"saved argv to {path.relative_to(toplevel_path())}.argv")

    print(os.getcwd())

def evals_plot(
    model_results,
    title,
    cropped=False,
    legend=None,
    font_size=14,
    linewidth=4
        ):

    """
    Pass in `model_results :: ModelResult | [ModelResult]`

    """

    if isinstance(model_results, ModelResult):
        model_results = [model_results]
    if legend:
        assert len(legend) == len(model_results)

    #############
    # * EVALS * #
    #############

    fig = plt.figure()
    plt.title(title, fontsize=font_size)
    plt.xlabel('Number of partial programs considered', fontsize=font_size)
    plt.ylabel('Percent correct', fontsize=font_size)
    x_max = min([m.cropped_x_max_evals for m in model_results]) if cropped else max([m.full_x_max_evals for m in model_results])
    plt.ylim(bottom=0., top=100.)
    plt.xlim(left=0., right=x_max)
    for i,m in enumerate(model_results):
        label = legend[i] if legend else m.cfg.job_info
        xs = list(range(m.cropped_x_max_evals)) # 0..earliest_failure
        ys = [m.percent(lambda search_try: search_try.hit and search_try.nodes_expanded <= x) for x in xs]
        plt.plot(
            xs,
            ys,
            label=label,
            linewidth=linewidth)
    plt.legend()
    return fig


def time_plot():
    raise NotImplementedError

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