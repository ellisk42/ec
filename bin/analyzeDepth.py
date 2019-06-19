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
MARKERTWO = 'v'

FONTSIZE = 18
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
    arguments = parser.parse_args()


    fullResults = {}
    for r in arguments.recognition:
        domain, depths, hits = loadCheckpoint(r)
        fullResults[domain] = fullResults.get(domain,[]) + [(depths, hits)]

    lesionResults = {}
    for r in arguments.generative:
        domain, depths, hits = loadCheckpoint(r)
        lesionResults[domain] = lesionResults.get(domain,[]) + [(depths, hits)]

    

    for mode in ["MAX","MEAN","SIZE"]:
        plot.figure(figsize=(4,4))
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
            plot.scatter(xs,ys,color=c,marker=style)
                

        legend = []
        for domain in set(fullResults.keys())|set(lesionResults.keys()):
            if domain in fullResults: scatter(fullResults[domain], colors[0], MARKERONE)
            if domain in lesionResults: scatter(lesionResults[domain], colors[0], MARKERTWO)
            legend.append((domain, colors[0]))
            colors = colors[1:]

        plot.ylabel("% Testing Tasks Solved",
                    fontsize=FONTSIZE)
        plot.xlabel({"MAX": "Maximum library depth",
                     "MEAN": "Average library depth",
                     "SIZE": "Library size"}[mode],
                    fontsize=FONTSIZE)
        plot.xticks(fontsize=TICKFONTSIZE)
        plot.yticks(fontsize=TICKFONTSIZE)
        if mode in {"MAX","SIZE"}:
            plot.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        r,p = pearsonr(X,Y)
        print(mode,r,p)

        plot.legend(fancybox=True, shadow=True,
                    handles=[mpatches.Patch(color=color, 
                                            label=label)
                               for label, color in legend] + \
                    [mlines.Line2D([], [], color='k', marker=marker, ls='None',
                                   label=label)
                     for marker, label in [(MARKERONE,"Full model"),
                                           (MARKERTWO,"No Rec")]],
                    bbox_to_anchor=(1,0.5),
                    loc='center left')

        
        
        plot.savefig(f"figures/depthVersusAccuracy_{mode}.png",
                     bbox_inches='tight')
