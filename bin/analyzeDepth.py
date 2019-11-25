try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.utilities import *
from dreamcoder.program import *

from scipy.stats.stats import pearsonr

from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import os

MARKERONE = '*'
MARKERTWO = '+'

FONTSIZE = 18
LEGENDSIZE = 12
TICKFONTSIZE = 14

def primitiveDepth(e):
    if isinstance(e,Invented):
        return 1 + primitiveDepth(e.body)
    if isinstance(e,Application):
        return max(primitiveDepth(e.f),primitiveDepth(e.x))
    if isinstance(e,Abstraction):
        return primitiveDepth(e.body)
    return 1

def loadCheckpoint(r):
    domain = os.path.basename(r).split("_")[0]
    domain = {"logo": "LOGO",
              "tower": "Towers",
              "text": "Text",
              "list": "List",
              "rational": "Regression"}[domain]
    result = loadPickle(r)

    if arguments.every:
        hits = [len(tst)/result.numTestingTasks for tst in result.testingSearchTime]
        depths = [ [primitiveDepth(e) for e in g.primitives ] for g in result.grammars]
        return domain,list(zip(range(999),hits,depth))
    
    g = result.grammars[-1]
    ds = [primitiveDepth(e) for e in g.primitives ]

    hitRatio = len(result.testingSearchTime[-1])/result.numTestingTasks

    return domain,ds,hitRatio
        
    
    
    
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--recognition", nargs='+', default=[],
                        help="List of checkpoints that use the full model.")
    parser.add_argument("--generative", nargs='+', default=[],
                        help="List of checkpoints that use the no-recognition lesion.")
    parser.add_argument("--every", default=False, action='store_true')
    parser.add_argument("--legend",default=[],nargs='+')
    arguments = parser.parse_args()

    if arguments.every:
        fullResults = {}
        for r in arguments.recognition:
            domain, data = loadCheckpoint(r)
            fullResults[domain] = fullResults.get(domain,[]) + data

        lesionResults = {}
        for r in arguments.generative:
            domain, data = loadCheckpoint(r)
            lesionResults[domain] = lesionResults.get(domain,[]) + data

        for mode in ["MEAN","SIZE"]:
            # points and their colors
            X = []
            Y = []
            C = []
            for results in [fullResults,lesionResults]:
                for domain, data in results:
                    for i,y,d in data:
                        Y.append(y) # fraction of solved tasks
                        if mode == "MEAN":
                            x = sum(d)/len(d)
                        elif mode == "SIZE":
                            x = sum(d_ > 1 for d_ in d)
                        else:
                            assert False
                        X.append(x)

                        if result is fullResults:
                            c = (1.,0.,i/20.)
                        elif result is lesionResults:
                            c = (0.,1.,i/20.)
                        C.append(c)

            plot.figure(figsize=(4,2.5))
            plot.scatter(X,Y,color=C,alpha=0.5)
            plot.savefig(f"figures/depthVersusAccuracy_revision_{mode}.png")
        sys.exit()

                
                            

                




    fullResults = {}
    for r in arguments.recognition:
        domain, depths, hits = loadCheckpoint(r)
        fullResults[domain] = fullResults.get(domain,[]) + [(depths, hits)]

    lesionResults = {}
    for r in arguments.generative:
        domain, depths, hits = loadCheckpoint(r)
        lesionResults[domain] = lesionResults.get(domain,[]) + [(depths, hits)]

    

    for mode in ["MAX","MEAN","SIZE"]:
        plot.figure(figsize=(4,2.5))
        colors = ["red","green","blue","purple","cyan"]

        X = []
        Y = []
        def scatter(dh, c, style):
            xs = []
            ys = []
            for depths, hits in dh:
                ys.append(hits)
                if mode == "MAX":
                    xs.append(max(depths) + random.random()*0.2 - 0.1)
                elif mode == "MEAN":
                    xs.append(sum(depths)/len(depths))
                elif mode == "SIZE":
                    xs.append(sum(d >= 1 for d in depths))
                else:
                    assert False

                X.append(xs[-1])
                Y.append(ys[-1])

            print(mode,pearsonr(xs,ys))
            plot.scatter(xs,[100*y for y in ys],color=c,marker=style)
                

        legend = []
        for domain in set(fullResults.keys())|set(lesionResults.keys()):
            if domain in fullResults: scatter(fullResults[domain], colors[0], MARKERONE)
            if domain in lesionResults: scatter(lesionResults[domain], colors[0], MARKERTWO)
            legend.append((domain, colors[0]))
            colors = colors[1:]

        plot.ylabel("% Test Solved",
                    fontsize=FONTSIZE)
        plot.xlabel({"MAX": "Max depth",
                     "MEAN": "Avg. depth",
                     "SIZE": "Library size"}[mode],
                    fontsize=FONTSIZE)
        plot.xticks(fontsize=TICKFONTSIZE)
        plot.yticks(fontsize=TICKFONTSIZE)
        if mode in {"MAX"}: #"SIZE"
            plot.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        r,p = pearsonr(X,Y)
        print(mode,r,p)

        handles = [mlines.Line2D([], [], color='k', marker=marker, ls='None',
                                 label=label)
                         for marker, label in [(MARKERONE,"Full model"),
                                               (MARKERTWO,"No Rec")]]
        if arguments.legend: handles = []
        handles.extend([mlines.Line2D([],[],color=color,marker='o',ls='None',label=label)
                        #mpatches.Patch(color=color, label=label)
                        for label, color in legend])


        if mode in arguments.legend:
            plot.legend(handles=handles,
                        fontsize=LEGENDSIZE,
                        borderpad=0.,
                        handletextpad=0.05,
#                        bbox_to_anchor=(1,0.5),
                        labelspacing=0.1,
                        loc='best')
#                        loc='lower right')

        
        
        plot.savefig(f"figures/depthVersusAccuracy_mainPaper_{mode}.eps",
                     bbox_inches='tight')
        plot.figure()
        legend = plot.legend(handles=handles,
                             handletextpad=0.05,
                             bbox_to_anchor=(1,0.5),
                             loc='center',
                             ncol=1)
        f = legend.figure
        f.canvas.draw()
        bb = legend.get_window_extent().transformed(f.dpi_scale_trans.inverted())        
        f.savefig("figures/depthVersusAccuracy_mainPaper_legend.eps",
                  bbox_inches=bb)

