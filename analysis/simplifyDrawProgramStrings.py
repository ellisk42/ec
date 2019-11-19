""" find and replace strings to make it more human readable"""
"""Should incorporate this into summary script"""

## loading checkpoint files

%load_ext autoreload
%autoreload 2

from analysis.utils import *
from analysis.parse import *
from analysis.analy import *
from analysis.graphs import *

# ==== LOAD DREAMCODER
ECTRAIN = "S12.10.test4"
DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=False, loadbehavior=False)

# === 4) Load tools to work with tasks libraries
%load_ext autoreload
%autoreload 2

from analysis.utils import *
from analysis.parse import *
from analysis.analy import *
from analysis.graphs import *
import dreamcoder.domains.draw.primitives as P


# GET PRIMITIVES TO RENAME
p_to_ignore = ['line', 'circle', 'transmat', 'transform', 
               'reflect', 'connect', 'repeat', "None", "Some"]
               
primcodes = {p.name:p.evaluate([]) for p in DP.primitives[::-1] if p.name not in p_to_ignore}
primcodes["None"]="N"
primcodes["Some"]=''
primcodes = {key:str(value)[:5] for key, value in primcodes.items()}

if ECTRAIN in ["S12.10.test4", "S13.10.test4"]:
    primcodes.update({
        "#(lambda (lambda (transmat N N $0 ( 0.28) $1)))":"#[MOVE X $ AND UP 0.28]",
        "#(lambda (transmat N N ( $0) N N))":"#[MOVE X $0]",
        "#(transform line (transmat ( 4.0) ( 1.570) ( -1.75) ( -2.0) N))":"#[VERTLINE LEFT]",
        "#(transform line (#[MOVE X $0] -0.5))":"#[HORIZLINE]",
        "#(lambda (lambda (transform (repeat $0 $1 (#[MOVE X $0] 1.1)) (#[MOVE X $0] -1.75))))":"#[PRIM $ REPEAT TO RIGHT]",
        "#(lambda (repeat #[VERTLINE LEFT] $0 (#[MOVE X $0] 1.1)))":"#[GRATING]",
        "#(lambda (transform (#[PRIM $ REPEAT TO RIGHT] $0 line) (#[MOVE X $0] -0.5)))":"#[HLINE REP RIGHT $]",
        "#(transform #[HORIZLINE] (#[MOVE X $0] -0.65))":"#[HLINE at -0.65]",
        "#(lambda (lambda (transmat N N $0 ( 1.0) $1)))":"#[MOVE X $ and Y 1]",
        "#(repeat #[HLINE at -0.65] 2)":"#[REPEAT 2 HLINE LEFT]",
        "#(lambda (lambda (connect (transform circle (#[MOVE X $ and Y 1] ( $0) ( $1))))))":"#[CIRCLE X$ Y1 + P$]",
        "#(lambda (lambda (connect (#[GRATING] $0) (reflect $1 0.0))))":"#[GRATING + REFLECT P$ 0]",
        "#(#[CIRCLE X$ Y1 + P$] -0.65)":"#[CIRCLE -0.65 1 + P$]",
        "#(lambda (lambda (#[CIRCLE X$ Y1 + P$] -1.75 $0 (reflect $1 0.0))))":"#[CIRCLE -1.75,1 + P$]",
        "#(lambda (lambda (#[GRATING + REFLECT P$ 0] (#[CIRCLE X$ Y1 + P$] 1.55 $1 $0) 4)))":"#[GRATING + CIRCLE 1.55,-1 + P$]",
        "#(lambda (lambda (connect (#[PRIM $ REPEAT TO RIGHT] 1 (connect (transform #[HORIZLINE] (transmat N N N ( 1.0) $0)) circle)) (#[GRATING] $1))))":"#[GRATING + LINE + CIRCLE]",
        "#(#[CIRCLE X$ Y1 + P$] 0.45 trs)":"#[CIRCLE + $]",
        "#(lambda (transform line (transmat ( $0) ( 1.570) ( -1.75) ( -2.0) N)))":"#[VLINE $S at -1.75,-2]",
        "#(#(lambda (lambda (connect (transform $0 (transmat N N $1 ( -1.0) N))))) ( -1.75) circle)":"#[CIRCLE -1.75, -1]",
        "#(lambda (lambda (repeat (#[VLINE $S at -1.75,-2] $1) $0 (#[MOVE X $0] 1.1))))":"#[REP VLINE $S -1.75,-2 by X 1.1 $R]",
        "#(connect (transform circle (#[MOVE X $0] -1.75)))":"#[CIRCLE + $P -1.75,0]",
        "#(lambda (lambda (connect (transform $0 (transmat N N $1 ( -1.0) N)))))":"#[$P $X -1]",
        "#(lambda (lambda (connect $1 (transform $0 (#[MOVE X $0] 1.1)))))":"#[$P + $P at X1.1]",
        "#(lambda (lambda (#[$P + $P at X1.1] (#[REP VLINE $S -1.75,-2 by X 1.1 $R] $1 4) (#[CIRCLE + $P -1.75,0] $0))))":"#[$P + $P at X1.1 + 4GRATING, variable height + CIRCLE at -1.75,0]",
        "#(lambda (lambda (connect (transform $0 (transmat None None $1 (Some dist3) None)))))":"#[$P -1X]",
        "#(lambda (lambda (connect $1 (transform $0 (#(lambda (transmat None None (Some $0) None None)) dist18)))))":"#[$P + $P moved X1.1]",
        "#(transform (#(lambda (transform (repeat (transform line (#[MOVE X $0] -1.75)) $0 (#[MOVE X $0] 1.1)) (#[MOVE X $0] -0.5))) 1) (#[MOVE X $0] 1.1))":"[REP HLINE MOVE START -1.75 BY X1.1 + $P at X-0.5 and $P at X1.1]"
    })


# note will do in order from top to bottom
result=DAT["result"]

# ==== go thru all primitives and replace if possible
P = result.grammars[-1].primitives
for i, p in enumerate(P):
    pp = str(p)
    print(i)
    print(pp)
    print( )
    for key in primcodes.keys():
        pp = pp.replace(key, primcodes[key])
    print(pp)
    print('-------')


# GO THRU ALL PRIMITIVES
    
    
# --- save
fname = "{}/{}_solutions_simple.txt".format(DAT["summarysavedir"], testortrain)
print(fname)
stringall = []
# ==== go thru all primitives and replace if possible
P = result.grammars[-1].primitives
for i, p in enumerate(P):
    pp = str(p)
       
    stringall.append(i)
    stringall.append(t.name)
    stringall.append(pp)
    stringall.append(" ")
    for key in primcodes.keys():
        pp = pp.replace(key, primcodes[key])
    stringall.append(pp)
    stringall.append('-------')
with open(fname, "w") as f:
    for s in stringall:
        print(s)
        f.write(str(s)+"\n")


# GO THRU ALL SOLUTIONS
# for stim in stimlist:
#     t = DATgetTask(stim, DAT)
#     result.frontiersOverTime(t)[-1]
#     raise
    
testortrain = "train"
if testortrain=="train":
    tasks = DAT["tasks"]
else:
    tasks = DAT["testtasks"]
    
# --- save
fname = "{}/{}_solutions_{}_simple.txt".format(DAT["summarysavedir"], DAT["trainset"], testortrain)
# fname = "/tmp/test.txt"
print(fname)
stringall = []
for i, t in enumerate(tasks):
    if result.frontiersOverTime[t][-1].empty:
        continue
        
    pp = str(result.frontiersOverTime[t][-1].bestPosterior.program)
    
    stringall.append(i)
    stringall.append(t.name)
    stringall.append(pp)
    stringall.append(" ")
    for key in primcodes.keys():
        pp = pp.replace(key, primcodes[key])
    stringall.append(pp)
    stringall.append('-------')
with open(fname, "w") as f:
    for s in stringall:
        print(s)
        f.write(str(s)+"\n")
