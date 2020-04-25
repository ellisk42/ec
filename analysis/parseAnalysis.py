""" more detailed analysis of parseing """
from analysis.utils import *
import numpy as np
from dreamcoder.domains.draw.primitives import *

def walkParse(parse):
    
    outdict = []
    def _walkParse(p, level, branch):
        from dreamcoder.domains.draw.primitives import Parse, Chunk
        import numpy as np

        if False:
            print('---')
            print(p)
            print(type(p))
            print(level)
            print(branch)
            print('---')

        if isinstance(p, set):
            assert False, "this is a set of parses! pick out a parse first"
#             p = list(p)
        elif isinstance(p, (Parse, Chunk)):
            p = p.l
        elif isinstance(p, np.ndarray):
            # save it
            outdict.append({
                "branch":branch,
                "level":level,
                "stroke":p
            })
            return None    

        for i, pp in enumerate(p):
    #         print("lower levels:")
    #         print(type(pp))
            if isinstance(pp, (Parse, Chunk)):
                next_lev = level
#                 branchthis = branch
            else:
                next_lev = level+1
#             branchthis = branch + [i]
            if len(p)>1:
                # then branches, should note it down.
                branchthis = branch + [i]
            else:
                branchthis = branch
            _walkParse(pp, next_lev, branchthis)

    _walkParse(parse, 0, [])
    
    print("DONE")
    return outdict



def plotParse(outdict, depth, ax=None, plot_branches_separate=False):
    """ for a given depth, split chunks at this depth:
    e.g., if two branches at start, then depth=0 will make two
    plots, one for eahc of hte branches
    - plot_branches_separate then each branch will not be another color on same 
    plot (default) but will be diff plots
    """
    
    from pythonlib.tools.plottools import makeColors
    
    def _plot(OD, title="", color="k", ax=None):
        """ plot each stroke overlaid on same plot, that is in OD"""
        if ax is None:
            ax = plot([])
        ax.set_title(title)
        for o in OD:
            plotOnAxes([o["stroke"]], ax, color=color)
        return ax

    def _go(OD, depth, pos):
        """obsolete - I used previuosly when would pick out stroke to plot here"""
        for o in OD:
            OD = [o for o in OD if o["branch"][depth]==pos]
        _plot(OD)
    
    def _pos2str(OD, depth):
        """ converts branches (list of nums) to a str. uindicates the branch for this 
        stroke at each level. each unique string represents part of a chunk
        that should be plotted. e.g, two strokes are [0,0,0] and [0,0,1]. if
        consider them as same chunk for plotting, then choose depth=1 and they will become:
        '00' and '00'. if choose depth2, they become '000' and '001' and will be plotted
        separately."""
        positions = ["".join([str(o["branch"][i]) for i in range(depth+1) if i < len(o["branch"])]) for o in OD]
        return positions
            
    positions = _pos2str(outdict, depth)
    for pos, o in zip(positions, outdict):
        o["poscode"] = pos

    plotcols = makeColors(len(set(positions)), alpha=0.75, cmap="jet")
    if ax is None and not plot_branches_separate:
        ax = plot([])
    if plot_branches_separate:
        # then generate a new subplots
        ncols = 6
        nrows = np.ceil(len(set(positions))/ncols)
        fig = plt.figure(figsize=(6*3, nrows*3))
    for n, (pos, pcol) in enumerate(zip(set(positions), plotcols)):
#         _go(outdict, depth=depth, pos = pos)
        OD = [o for o in outdict if o["poscode"]==pos]
        if plot_branches_separate:
            ax = plt.subplot(nrows, ncols, n+1)
            pcol = "k"
        _plot(OD, f"{depth} depth, {len(plotcols)} chunks", color=pcol, ax=ax)
    if plot_branches_separate:
        return fig
    else:
        return None

        
def getMaxDepth(outdict):
    """ can use this to dcide how many plots to make.
    i.e,, each time plot choose a different depth to split chunks"""
    return max([len(o["branch"]) for o in outdict])-1
    
    
def plotProgramTree(DAT, task, pnum=0, plot_branches_separate=False):
    """ not using parse, but instead plotting the chunk structure
    of teh dc program
    - pnum = 0 is OK.
    """
    from analysis.parse import getBestFrontierProgram
    
    tname = task.name
    solved = DAT['taskresultdict'][tname]
    print(f"solved? {solved}")

    # 1) Load best frontier program
    frontier = getBestFrontierProgram(DAT["result"], task, returnfrontier=True)
    p = frontier.program

    # 2) Load previuosly saved parses
    # parse_old = [D["parse"] for D in DAT["parses"] if D["name"]==tname][0] # this is flattened parses...

    parses = DATloadParse(DAT, tname, flattened=False)
    assert len(parses)>0, "did not find parse..."
    # parse = getParses(p)
    # P = list(parse)[i]

    # 3) Convert a parse into heirarchical format
    outdict = walkParse(list(parses)[pnum])

    # 4) Plot for this dreamcoder solution
    numplots = getMaxDepth(outdict)
    if not plot_branches_separate:
        ncols = 6
        nrows = np.ceil(numplots/ncols)
        fig = plt.figure(figsize=(6*3, nrows*3))
    else:
        fig =[]
    for n in range(numplots):
        if not plot_branches_separate:
            ax = plt.subplot(nrows, ncols, n+1)
            plotParse(outdict, depth=n, ax=ax)
        else:
            fig.append(plotParse(outdict, depth=n, plot_branches_separate=True))
    return fig