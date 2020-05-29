from collections import OrderedDict
import datetime
import json
import os
import pickle
import random as random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.domains.logo.makeLogoTasks import makeTasks, montageTasks, drawLogo, manualLogoTasks, makeLogoUnlimitedTasks, generateLogoDataset, loadLogoDataset, sampleSupervised
from dreamcoder.domains.logo.logoPrimitives import primitives, turtle, tangle, tlength
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.program import Program, Index
from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.utilities import eprint, testTrainSplit, loadPickle

def saveVisualizedTasks(tasks, visualizeTasks):
    # Store the high resolution tasks.
    import imageio
    import os.path as osp
    for n, t in enumerate(tasks):
        logo_safe_name = t.name.replace("=","_").replace(' ','_').replace('/','_').replace("-","_").replace(")","_").replace("(","_")
        logo_name = 'logo_{}_name_{}.png'.format(n, logo_safe_name)
        a = t.highresolution
        w = int(len(a)**0.5)
        img = np.array([a[i:i+w]for i in range(0,len(a),w)])
        img_name = osp.join(visualizeTasks, logo_name)
        imageio.imwrite(img_name, img)
        os.system(f"convert {img_name} -channel RGB -negate {img_name}")

def animateSolutions(allFrontiers, animate, checkpoint=None):
    programs = []
    filenames = []
    sorted_frontiers = sorted(allFrontiers.items(), key=lambda f: f[0].name)
    for n,(t,f) in sorted_frontiers:
        if f.empty: continue

        programs.append(f.bestPosterior.program)
        # Write them into the checkpoint dir
        if checkpoint is None: 
            outputDir = '/tmp'
        else:
            import os.path as osp
            outputDir = osp.join(osp.dirname(checkpoint), 'solutions')
        filenames.append(osp.join(outputDir, f'logo_animation_{n}'))
    
    drawLogo(*programs, pretty=True, smoothPretty=True, resolution=128, animate=animate,
             filenames=filenames)

def get_unigram_uses(grammar, programs):
    """Note that this only returns a summary of unigrams actually used in the programs."""
    from itertools import chain
    closed_summaries = [grammar.closedLikelihoodSummary(request, program) for (request, program) in programs]
    all_primitives = set(chain(*[list(ls.uses.keys()) for ls in closed_summaries]))
    primitives_to_idx = {p : i for (i, p) in enumerate(sorted(all_primitives, key=lambda p:str(p)))}
    
    unigram_uses = []
    for ls in closed_summaries:
        uses = [ls.uses[p] if p in ls.uses else 0.0 for p in primitives_to_idx]
        unigram_uses.append(uses)
    
    primitives_to_idx = {str(p) : i for (p, i) in primitives_to_idx.items()}
    return primitives_to_idx, unigram_uses

def get_bigram_uses(contextual_grammar, grammar, programs):
    """This returns an overapproximated transition matrix of counts, similar to that 
    used by the recognition model."""
    closed_summaries = [contextual_grammar.closedLikelihoodSummary(request, program) for (request, program) in programs]
    bigram_uses = []
    
    library = {}
    n_grammars = 0
    for prim in grammar.primitives:
        numberOfArguments = len(prim.infer().functionArguments())
        idx_list = list(range(n_grammars, n_grammars+numberOfArguments))
        library[prim] = idx_list
        n_grammars += numberOfArguments
    # Extra grammar for when there is no parent and for when the parent is a variable
    n_grammars += 2
    G = len(grammar) + 1
    for summary in closed_summaries:
        uses = np.zeros((n_grammars,len(grammar)+1))
        for e, ss in summary.library.items():
            for g,s in zip(library[e], ss):
                assert g < n_grammars - 2
                for p, production in enumerate(grammar.primitives):
                    uses[g,p] = s.uses.get(production, 0.)
                uses[g,len(grammar)] = s.uses.get(Index(0), 0)
                
        # noParent: this is the last network output
        for p, production in enumerate(grammar.primitives):            
            uses[n_grammars - 1, p] = summary.noParent.uses.get(production, 0.)
        uses[n_grammars - 1, G - 1] = summary.noParent.uses.get(Index(0), 0.)

        # variableParent: this is the penultimate network output
        for p, production in enumerate(grammar.primitives):            
            uses[n_grammars - 2, p] = summary.variableParent.uses.get(production, 0.)
        uses[n_grammars - 2, G - 1] = summary.variableParent.uses.get(Index(0), 0.)
        bigram_uses.append((list(np.ravel(uses)), uses.shape))
    
    library = {str(p) : i for (p, i) in library.items()}
    return library, bigram_uses
    
def solutionPrimitiveCounts(allFrontiers, grammar, checkpoint):
    # TODO: @CathyWong: should print the checkpoint iter somewhere
    # TODO: cluster on non-invented (rewritten); rewrite in several other older grammars? (how do)= -- maybe just original.
    
    sorted_frontiers = sorted(allFrontiers.items(), key=lambda f: f[0].name)
    sorted_frontiers = [(t, f) for (t, f) in sorted_frontiers if not f.empty]
    frontiers_to_idx = {t.name : i for (i, (t, f)) in enumerate(sorted_frontiers)}
    
    from collections import defaultdict
    counts = {n : defaultdict() for n in range(len(allFrontiers))}
    g, contextual_g = grammar[-1], ContextualGrammar.fromGrammar(grammar[-1])
    
    best_programs = [(f.task.request, f.bestPosterior.program) for n,(t,f) in enumerate(sorted_frontiers)]
    
    # Rewrite in initial DSL.
    beta_normal = [(r, p.betaNormalForm()) for (r, p) in best_programs]

    object = {}
    for (name, programs) in zip(['final_dsl', 'beta_normal'], [best_programs, beta_normal]):
    
        unigram_to_idx, unigram_uses = get_unigram_uses(g, programs)
        bigram_to_idx, bigram_uses = get_bigram_uses(contextual_g, g, programs)
        metrics = {
            'frontiers_to_idx' : frontiers_to_idx,
            'unigram_to_idx' : unigram_to_idx,
            'bigram_to_idx' : bigram_to_idx,
            'unigram_uses' : unigram_uses,
            'bigram_uses' : bigram_uses
        }
        object[name] = metrics
    import json
    import os.path as osp
    outputDir = osp.join(osp.dirname(checkpoint), 'solutions')
    with open(osp.join(outputDir, 'primitive_counts.json'), 'w') as f:
        json.dump(object, f)
    with open(osp.join(outputDir, 'primitive_counts.json'), 'r') as f:
        json.load(f)
    
def dreamFromGrammar(g, directory, N=100):
    if isinstance(g,Grammar):
        programs = [ p
                     for _ in range(N)
                     for p in [g.sample(arrow(turtle,turtle),
                                        maximumDepth=20)]
                     if p is not None]
    else:
        programs = g
    drawLogo(*programs,
             pretty=False, smoothPretty=False,
             resolution=512,
             filenames=[f"{directory}/{n}.png" for n in range(len(programs)) ],
             timeout=1)
    drawLogo(*programs,
             pretty=True, smoothPretty=False,
             resolution=512,
             filenames=[f"{directory}/{n}_pretty.png" for n in range(len(programs)) ],
             timeout=1)
    drawLogo(*programs,
             pretty=False, smoothPretty=True,
             resolution=512,
             filenames=[f"{directory}/{n}_smooth_pretty.png" for n in range(len(programs)) ],
             timeout=1)
    for n,p in enumerate(programs):
        with open(f"{directory}/{n}.dream","w") as handle:
            handle.write(str(p))        
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class LogoFeatureCNN(nn.Module):
    special = "LOGO"
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(LogoFeatureCNN, self).__init__()

        self.sub = prefix_dreams + str(int(time.time()))

        self.recomputeTasks = False

        def conv_block(in_channels, out_channels, p=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.Conv2d(out_channels, out_channels, 3, padding=1),
                # nn.ReLU(),
                nn.MaxPool2d(2))

        self.inputImageDimension = 128
        self.resizedDimension = 128
        assert self.inputImageDimension % self.resizedDimension == 0

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 256

        
    def forward(self, v):
        assert len(v) == self.inputImageDimension*self.inputImageDimension
        floatOnlyTask = list(map(float, v))
        reshaped = [floatOnlyTask[i:i + self.inputImageDimension]
                    for i in range(0, len(floatOnlyTask), self.inputImageDimension)]
        v = variable(reshaped).float()
        # insert channel and batch
        v = torch.unsqueeze(v, 0)
        v = torch.unsqueeze(v, 0)
        v = maybe_cuda(v, next(self.parameters()).is_cuda)/256.
        window = int(self.inputImageDimension/self.resizedDimension)
        v = F.avg_pool2d(v, (window,window))
        v = self.encoder(v)
        return v.view(-1)

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.highresolution)

    def tasksOfPrograms(self, ps, types):
        images = drawLogo(*ps, resolution=128)
        if len(ps) == 1: images = [images]
        tasks = []
        for i in images:
            if isinstance(i, str): tasks.append(None)
            else:
                t = Task("Helm", arrow(turtle,turtle), [])
                t.highresolution = i
                tasks.append(t)
        return tasks        

    def taskOfProgram(self, p, t):
        return self.tasksOfPrograms([p], None)[0]

def list_options(parser):
    parser.add_argument("--proto",
                        default=False,
                        action="store_true",
                        help="Should we use prototypical networks?")
    parser.add_argument("--target", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--taskDataset", type=str,
                        choices=[
                            "logo_unlimited_1300",
                            "logo_unlimited_1000",
                            "logo_unlimited_500",
                            "logo_unlimited_200"],
                        default=None,
                        help="Load pre-generated task datasets.")
    parser.add_argument("--reduce", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--save", type=str,
                        default=None,
                        help="Filepath output the grammar if this is a child")
    parser.add_argument("--prefix", type=str,
                        default="experimentOutputs/",
                        help="Filepath output the grammar if this is a child")
    parser.add_argument("--dreamCheckpoint", type=str,
                        default=None,
                        help="File to load in order to get dreams")
    parser.add_argument("--dreamDirectory", type=str,
                        default=None,
                        help="Directory in which to dream from --dreamCheckpoint")
    parser.add_argument("--visualize",
                        default=None, type=str)
    parser.add_argument("--split",
                        default=1., type=float)
    parser.add_argument("--animate",
                        default=None, type=str)
    parser.add_argument("--visualizeSolutions",
                        default=None, type=str)
    parser.add_argument("--visualizeTasks",
                        default=None, type=str)
    parser.add_argument("--visualizePlay",
                        default=None, type=str)
    parser.add_argument("--solutionPrimitiveCounts",
                        default=None, type=str)
    parser.add_argument("--taskDatasetDir",
                        default="data/logo/tasks")
    parser.add_argument("--languageDatasetDir",
                        default="data/logo/language")
    parser.add_argument("--generateTaskDataset",
                        choices=[
                            "logo_classic",
                            "logo_unlimited_1000",
                            "logo_unlimited_500",
                            "logo_unlimited_200"],
                        default=None,
                        help="Generates pre-cached dataset and stores it under the top-level path.")
    parser.add_argument("--generateLanguageDataset",
                        choices=[
                            "logo_unlimited_1000",
                            "logo_unlimited_500",
                            "logo_unlimited_200"],
                        default=None,
                        help="Generates language dataset and stores it under the top-level path.")
    parser.add_argument("--iterations_as_epochs",
                        default=1,
                        type=int)
    parser.add_argument("--sample_n_supervised",
                        default=0, type=int)
    parser.add_argument("--om_original_ordering",
                        default=0, type=int,
                        help="Uses the original ordering for the initial run for experiments in the paper.")

def outputDreams(checkpoint, directory):
    from dreamcoder.utilities import loadPickle
    result = loadPickle(checkpoint)
    eprint(" [+] Loaded checkpoint",checkpoint)
    g = result.grammars[-1]
    if directory is None:
        randomStr = ''.join(random.choice('0123456789') for _ in range(10))
        directory = "/tmp/" + randomStr
    eprint(" Dreaming into",directory)
    os.system("mkdir  -p %s"%directory)
    dreamFromGrammar(g, directory)

def enumerateDreams(checkpoint, directory):
    from dreamcoder.dreaming import backgroundHelmholtzEnumeration
    from dreamcoder.utilities import loadPickle
    result = loadPickle(checkpoint)
    eprint(" [+] Loaded checkpoint",checkpoint)
    g = result.grammars[-1]
    if directory is None: assert False, "please specify a directory"
    eprint(" Dreaming into",directory)
    os.system("mkdir  -p %s"%directory)
    frontiers = backgroundHelmholtzEnumeration(makeTasks(None,None), g, 100,
                                               evaluationTimeout=0.01,
                                               special=LogoFeatureCNN.special)()
    print(f"{len(frontiers)} total frontiers.")
    MDL = 0
    def L(f):
        return -list(f.entries)[0].logPrior
    frontiers.sort(key=lambda f: -L(f))
    while len(frontiers) > 0:
        # get frontiers whose MDL is between [MDL,MDL + 1)
        fs = []
        while len(frontiers) > 0 and L(frontiers[-1]) < MDL + 1:
            fs.append(frontiers.pop(len(frontiers) - 1))
        if fs:
            random.shuffle(fs)
            print(f"{len(fs)} programs with MDL between [{MDL}, {MDL + 1})")

            fs = fs[:500]
            os.system(f"mkdir {directory}/{MDL}")
            dreamFromGrammar([list(f.entries)[0].program for f in fs],
                             f"{directory}/{MDL}")
        MDL += 1

def visualizePrimitives(primitives, export='/tmp/logo_primitives.png'):
    from itertools import product
    from dreamcoder.program import Index,Abstraction,Application
    from dreamcoder.utilities import montageMatrix,makeNiceArray
    from dreamcoder.type import tint
    import scipy.misc
    from dreamcoder.domains.logo.makeLogoTasks import parseLogo

    angles = [Program.parse(a)
              for a in ["logo_ZA",
                        "logo_epsA",
                        "(logo_MULA logo_epsA 2)",
                        "(logo_DIVA logo_UA 4)",
                        "(logo_DIVA logo_UA 5)",
                        "(logo_DIVA logo_UA 7)",
                        "(logo_DIVA logo_UA 9)",
                        ] ]
    specialAngles = {"#(lambda (lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT (logo_MULL logo_UL 3) (logo_MULA $2 4) $0))) $1)))":
                     [Program.parse("(logo_MULA logo_epsA 4)")]+[Program.parse("(logo_DIVA logo_UA %d)"%n) for n in [7,9] ]}
    numbers = [Program.parse(n)
               for n in ["1","2","5","7","logo_IFTY"] ]
    specialNumbers = {"#(lambda (#(lambda (lambda (lambda (lambda (logo_forLoop $2 (lambda (lambda (logo_FWRT $5 (logo_DIVA logo_UA $3) $0))) $0))))) (logo_MULL logo_UL $0) 4 4))":
                      [Program.parse(str(n)) for n in [1,2,3] ]}
    distances = [Program.parse(l)
                 for l in ["logo_ZL",
                           "logo_epsL",
                           "(logo_MULL logo_epsL 2)",
                           "(logo_DIVL logo_UL 2)",
                           "logo_UL"] ]
    subprograms = [parseLogo(sp)
                   for sp in ["(move 1d 0a)",
                              "(loop i infinity (move (*l epsilonLength 4) (*a epsilonAngle 2)))",
                              "(loop i infinity (move (*l epsilonLength 5) (/a epsilonAngle 2)))",
                              "(loop i 4 (move 1d (/a 1a 4)))"]]

    entireArguments = {"#(lambda (lambda (#(#(lambda (lambda (lambda (logo_forLoop $2 (lambda (lambda (logo_FWRT $2 $3 $0))))))) logo_IFTY) (logo_MULA (#(logo_DIVA logo_UA) $1) $0) (#(logo_MULL logo_UL) 3))))":
                       [[Program.parse(str(x)) for x in xs ]
                        for xs in [("3", "1", "$0"),
                                   ("4", "1", "$0"),
                                   ("5", "1", "$0"),
                                   ("5", "3", "$0"),
                                   ("7", "3", "$0")]]}
    specialDistances = {"#(lambda (lambda (logo_forLoop 7 (lambda (lambda (#(lambda (lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $2 (lambda (lambda (logo_FWRT $2 $3 $0))))))) 7 $1 $2 $0)))) $3 logo_epsA $0))) $0)))":
                        [Program.parse("(logo_MULL logo_epsL %d)"%n) for n in range(5)]}
    
    matrix = []
    for p in primitives:
        if not p.isInvented: continue
        t = p.tp
        eprint(p,":",p.tp)
        if t.returns() != turtle:
            eprint("\t(does not return a turtle)")
            continue

        def argumentChoices(t):
            if t == turtle:
                return [Index(0)]
            elif t == arrow(turtle,turtle):
                return subprograms
            elif t == tint:
                return specialNumbers.get(str(p),numbers)
            elif t == tangle:
                return specialAngles.get(str(p),angles)
            elif t == tlength:
                return specialDistances.get(str(p),distances)
            else: return []

        ts = []
        for arguments in entireArguments.get(str(p),product(*[argumentChoices(t) for t in t.functionArguments() ])):
            eprint(arguments)
            pp = p
            for a in arguments: pp = Application(pp,a)
            pp = Abstraction(pp)
            i = np.reshape(np.array(drawLogo(pp, resolution=128)), (128,128))
            if i is not None:
                ts.append(i)
            

        if ts == []: continue

        matrix.append(ts)
        if len(ts) < 6: ts = [ts]
        else: ts = makeNiceArray(ts)
        r = montageMatrix(ts)
        fn = "/tmp/logo_primitive_%d.png"%len(matrix)
        eprint("\tExported to",fn)
        scipy.misc.imsave(fn, r)
        
    matrix = montageMatrix(matrix)
    scipy.misc.imsave(export, matrix)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on LOGO tasks.
    """

    # The below legacy global statement is required since prefix_dreams is used by LogoFeatureCNN.
    # TODO(lcary): use argument passing instead of global variables.
    global prefix_dreams

    # The below global statement is required since primitives is modified within main().
    # TODO(lcary): use a function call to retrieve and declare primitives instead.
    global primitives

    visualizeCheckpoint = args.pop("visualize")
    if visualizeCheckpoint is not None:
        with open(visualizeCheckpoint,'rb') as handle:
            primitives = pickle.load(handle).grammars[-1].primitives
        visualizePrimitives(primitives)
        sys.exit(0)

    dreamCheckpoint = args.pop("dreamCheckpoint")
    dreamDirectory = args.pop("dreamDirectory")

    proto = args.pop("proto")

    if dreamCheckpoint is not None:
        #outputDreams(dreamCheckpoint, dreamDirectory)
        enumerateDreams(dreamCheckpoint, dreamDirectory)
        sys.exit(0)

    animateCheckpoint = args.pop("animate")
    if animateCheckpoint is not None:
        animateSolutions(loadPickle(animateCheckpoint).allFrontiers, animate=True)
        sys.exit(0)
    
    visualizeCheckpoint = args.pop("visualizeSolutions")
    if visualizeCheckpoint is not None:
        animateSolutions(loadPickle(visualizeCheckpoint).allFrontiers, animate=False, checkpoint=visualizeCheckpoint)
        sys.exit(0)
    
    solutionCheckpoint = args.pop("solutionPrimitiveCounts")
    if solutionCheckpoint is not None:
        checkpoint = loadPickle(solutionCheckpoint)
        solutionPrimitiveCounts(checkpoint.allFrontiers, checkpoint.grammars, solutionCheckpoint)
        sys.exit(0)
    
    visualizeTasks = args.pop("visualizeTasks")
    if visualizeTasks is not None:
        tasks = manualLogoTasks(resolution=[28,1024])
        saveVisualizedTasks(tasks, visualizeTasks)
        sys.exit(0)
    
    visualizePlay = args.pop("visualizePlay")
    if visualizePlay is not None:
        tasks = makeLogoUnlimitedTasks(resolution=[28,1024])
        saveVisualizedTasks(tasks, visualizePlay)
        sys.exit(0)
    
    ### Dataset generation.
    generateTaskDataset = args.pop("generateTaskDataset")
    generateLanguageDataset = args.pop("generateLanguageDataset")
    if generateLanguageDataset is not None and generateTaskDataset is None:
        print("Error: cannot generate language without generating task dataset.")
        assert False
    if generateLanguageDataset is not None and not (generateTaskDataset == generateLanguageDataset):
        print("Error: task and language datasets must be the same.")
    if generateTaskDataset is not None:
        generateLogoDataset(task_dataset=generateTaskDataset,
                            task_dataset_dir=args.pop("taskDatasetDir"),
                            language_dataset=generateLanguageDataset,
                            language_dataset_dir=args.pop("languageDatasetDir"))
        assert False
    
    target = args.pop("target")
    save = args.pop("save")
    prefix = args.pop("prefix")
    split = args.pop("split")
    prefix_dreams = prefix + "/dreams/" + ('_'.join(target)) + "/"
    prefix_pickles = prefix + "/logo." + ('.'.join(target))
    if not os.path.exists(prefix_dreams):
        os.makedirs(prefix_dreams)
    
    om_original_ordering = args.pop("om_original_ordering")
    sample_n_supervised = args.pop("sample_n_supervised")
    task_dataset = args.pop("taskDataset")
    task_dataset_dir=args.pop("taskDatasetDir")
    if task_dataset:
        train, test = loadLogoDataset(task_dataset=task_dataset, task_dataset_dir=task_dataset_dir,
        om_original_ordering=om_original_ordering)
        eprint(f"Loaded dataset [{task_dataset}]: [{len(train)}] train and [{len(test)}] test tasks.")
        if sample_n_supervised > 0:
            eprint(f"Sampling n={sample_n_supervised} supervised tasks.")
            train = sampleSupervised(train, sample_n_supervised)    
    else: 
        tasks = makeTasks(target, proto)
        eprint("Generated", len(tasks), "tasks")

        os.chdir("prototypical-networks")
        subprocess.Popen(["python","./protonet_server.py"])
        time.sleep(3)
        os.chdir("..")

        test, train = testTrainSplit(tasks, split)
        eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))
        if test: montageTasks(test,"test_")    
        montageTasks(train,"train_")

    red = args.pop("reduce")
    if red is not []:
        for reducing in red:
            try:
                with open(reducing, 'r') as f:
                    prods = json.load(f)
                    for e in prods:
                        e = Program.parse(e)
                        if e.isInvented:
                            primitives.append(e)
            except EOFError:
                eprint("Couldn't grab frontier from " + reducing)
            except IOError:
                eprint("Couldn't grab frontier from " + reducing)
            except json.decoder.JSONDecodeError:
                eprint("Couldn't grab frontier from " + reducing)

    primitives = list(OrderedDict((x, True) for x in primitives).keys())
    baseGrammar = Grammar.uniform(primitives, continuationType=turtle)

    eprint(baseGrammar)

    timestamp = datetime.datetime.now().isoformat()
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(".", "-")
    outputDirectory = "experimentOutputs/logo/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    
    use_epochs = bool(args.pop("iterations_as_epochs"))
    if use_epochs and (args["taskBatchSize"] is not None):
        eprint("Using iterations as epochs over full training set.")
        multiplier = (int(len(train) / args["taskBatchSize"]))
        original_iterations = int(args["iterations"])
        args["iterations"] = original_iterations * multiplier
        
        eprint(f'Now running for n={args["iterations"]} iterations.')
    
    generator = ecIterator(baseGrammar, train,
                           taskDataset=task_dataset,
                           testingTasks=test,
                           outputPrefix="%s/logo"%outputDirectory,
                           evaluationTimeout=0.01,
                           **args)

    r = None
    for result in generator:
        iteration = len(result.learningCurve)
        
        # Skip dreaming.
        if False:
            dreamDirectory = "%s/dreams_%d"%(outputDirectory, iteration)
            os.system("mkdir  -p %s"%dreamDirectory)
            eprint("Dreaming into directory",dreamDirectory)
            dreamFromGrammar(result.grammars[-1],
                             dreamDirectory)
        r = result

    needsExport = [str(z)
                   for _, _, z
                   in r.grammars[-1].productions
                   if z.isInvented]
    if save is not None:
        with open(save, 'w') as f:
            json.dump(needsExport, f)
