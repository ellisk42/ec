# coding: utf8

import json
import os
import random
import sys
import dill
import numpy as np
from collections import defaultdict

from dreamcoder.domains.logo.logoPrimitives import primitives, turtle
from dreamcoder.task import Task
from dreamcoder.program import Abstraction, Application, Index, Program
from dreamcoder.type import arrow
from dreamcoder.utilities import eprint, jsonBinaryInvoke, random_seed, montage
from dreamcoder.grammar import Grammar


def drawLogo(*programs,
             timeout=None,
             resolution=None,
             pretty=False, smoothPretty=False,
             filenames=[],
             animate=False):
    message = {}
    if pretty: message["pretty"] = pretty
    if smoothPretty: message["smoothPretty"] = smoothPretty
    if timeout: message["timeout"] = timeout
    assert resolution is not None, "resolution not provided in drawLogo"
    if isinstance(resolution, list):
        assert len(resolution) == len(programs), "must provide a resolution for each program"
    elif isinstance(resolution, int):
        resolution = [resolution]*len(programs)
    else: assert False
    jobs = []
    for p, size in zip(programs, resolution):
        entry = {"program": str(p),
                 "size": size}
        if animate: entry["animate"] = True
        if len(filenames) > 0:
            entry["export"] = filenames[0]
            filenames = filenames[1:]
        jobs.append(entry)        
    message["jobs"] = jobs
    response = jsonBinaryInvoke("./logoDrawString", message)

    if len(programs) == 1:
        return response[0]
    return response

def makeTasks(subfolders, proto):
    tasks =  manualLogoTasks()
    lens = [t.token_len for t in tasks]
    print(f"Max len: {max(lens)}, average len: {np.mean(lens)}")
    return tasks

def parseLogo(s):
        
    _ua = Program.parse("logo_UA")
    _ul = Program.parse("logo_UL")

    _za = Program.parse("logo_ZA")
    _zl = Program.parse("logo_ZL")

    _da = Program.parse("logo_DIVA")
    _ma = Program.parse("logo_MULA")
    _dl = Program.parse("logo_DIVL")
    _ml = Program.parse("logo_MULL")

    _aa = Program.parse("logo_ADDA")
    _sa = Program.parse("logo_SUBA")
    _al = None#Program.parse("logo_ADDL")
    _sl = None#Program.parse("logo_SUBL")

    _pu = None#Program.parse("logo_PU")
    _pd = None#Program.parse("logo_PD")
    _p = Program.parse("logo_PT")
    _move = Program.parse("logo_FWRT")
    _embed = Program.parse("logo_GETSET")

    _addition = Program.parse("+")
    _infinity = Program.parse("logo_IFTY")
    _ea = Program.parse("logo_epsA")
    _el = Program.parse("logo_epsL")
    _loop = Program.parse("logo_forLoop")

    from sexpdata import loads, Symbol
    s = loads(s)
    def command(k, environment, continuation):
        assert isinstance(k,list)
        if k[0] == Symbol("move"):
            return Application(Application(Application(_move,
                                                       expression(k[1],environment)),
                                           expression(k[2],environment)),
                               continuation)
        if k[0] == Symbol("for") or k[0] == Symbol("loop"):
            v = k[1]
            b = expression(k[2], environment)
            newEnvironment = [None, v] + environment
            body = block(k[3:], newEnvironment, Index(0))
            return Application(Application(Application(_loop,b),
                                           Abstraction(Abstraction(body))),
                               continuation)
        if k[0] == Symbol("embed"):
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(_embed,Abstraction(body)),continuation)
        if k[0] == Symbol("p"):
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(_p,Abstraction(body)),continuation)

        assert False
    def expression(e, environment):
        for n, v in enumerate(environment):
            if e == v: return Index(n)

        if isinstance(e,int): return Program.parse(str(e))

        mapping = {"1a": _ua,
                   "1d": _ul, "1l": _ul,
                   "0a": _za,
                   "0d": _zl, "0l": _zl,
                   "/a": _da,
                   "/l": _dl, "/d": _dl,
                   "*a": _ma,
                   "*l": _ml, "*d": _ml,
                   "+a": _aa,
                   "+d": _al, "+l": _al,
                   "-a": _sa,
                   "-d": _sl, "-l": _sl,
                   "+": _addition,
                   "infinity": _infinity,
                   "epsilonAngle": _ea,
                   "epsilonDistance": _el,
                   "epsilonLength": _el}
        if e == float('inf'): return _infinity
        for name, value in mapping.items():
            if e == Symbol(name): return value
            
        assert isinstance(e,list), "not a list %s"%e
        for name, value in mapping.items():
            if e[0] == Symbol(name):
                f = value
                for argument in e[1:]:
                    f = Application(f, expression(argument, environment))
                return f
        assert False
        
    def block(b, environment, continuation):
        if len(b) == 0: return continuation
        return command(b[0], environment, block(b[1:], environment, continuation))

    try: return Abstraction(command(s, [], Index(0)))
    except: return Abstraction(block(s, [], Index(0)))

def token_length(p):
    return len(str(p).replace("lambda", " ").replace(")", " ").replace("(", " ").split())
    
def manualLogoTask(name, expression, proto=False, needToTrain=False,
                   supervise=False, lambdaCalculus=False, resolution=[28,128],
                   language=None):
    p = Program.parse(expression) if lambdaCalculus else parseLogo(expression)
    from dreamcoder.domains.logo.logoPrimitives import primitives
    from dreamcoder.grammar import Grammar
    g = Grammar.uniform(primitives, continuationType=turtle)
    gp = Grammar.uniform(primitives)
    try:
        l = g.logLikelihood(arrow(turtle,turtle),p)
        lp = gp.logLikelihood(arrow(turtle,turtle),p)
        assert l >= lp
        groundTruthLogLikelihood = -l
        eprint(name,-l,"nats")
        
    except: 
        groundTruthLogLikelihood = 10000
        eprint("WARNING: could not calculate likelihood of manual logo",p)

    attempts = 0
    while True:
        [output, highresolution] = drawLogo(p, p, resolution=[28,128])
        if output == "timeout" or highresolution == "timeout":
            attempts += 1
        else:
            break
    if attempts > 0:
        eprint(f"WARNING: Took {attempts} attempts to render task {name} within timeout")
            
    shape = list(map(int, output))
    highresolution = list(map(float, highresolution))
    t = Task(name, arrow(turtle,turtle),
             [(([0]), shape)])
    t.mustTrain = needToTrain
    t.proto = proto
    t.specialTask = ("LOGO", {"proto": proto})
    t.groundTruthLogLikelihood = groundTruthLogLikelihood 
    t.highresolution = highresolution
    t.groundTruthProgram = p
    t.expression = expression
    
    
        
    t.token_len = token_length(p)

    if supervise:
        t.supervisedSolution = p
    
    t.language = [language]

    return t

def dSLDemo():
    n = 0
    demos = []
    def T(source):
        demos.append(manualLogoTask(str(len(demos)), source,
                                    lambdaCalculus="lambda" in source))
    # this looks like polygons - verify and include
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 4) 3)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 6) 4)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 5) 5)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 3) 6)")
    T("(#(lambda (lambda (#(lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) $1 $0 logo_IFTY))) $1 (logo_DIVA logo_UA $0)))) (logo_MULL logo_UL 2) 7)")

    # Spirals!
    for spiralSize in [1,2,3,4,5]:
        T(f"((lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT (logo_MULL logo_epsL $1) (logo_MULA logo_epsA $2) $0))))) {spiralSize})")
    for spiralSize in [5,6,7,8,9]:
        #T(f"(lambda (#(lambda (logo_forLoop $0 (lambda (lambda (#(lambda (logo_FWRT (logo_MULL logo_UL $0) (logo_DIVA logo_UA 4))) $1 $0))))) {spiralSize} $0))")
        T("(loop i " + str(spiralSize) + " (move (*d 1l i) (/a 1a 4)))")# (#(lambda (logo_forLoop $0 (lambda (lambda (#(lambda (logo_FWRT (logo_MULL logo_UL $0) (logo_DIVA logo_UA 4))) $1 $0))))) {spiralSize} $0))")

    # CIRCLES
    #(lambda (#(lambda (logo_forLoop 6 (lambda (lambda (#(lambda (lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT $2 $3 $0)))))) logo_epsA (logo_MULL logo_epsL $2) $0))))) 6 $0))
    for circleSize in [1,3,5,7,9]:
        T(f"(lambda (#(lambda (logo_forLoop 6 (lambda (lambda (#(lambda (lambda (logo_forLoop logo_IFTY (lambda (lambda (logo_FWRT $2 $3 $0)))))) logo_epsA (logo_MULL logo_epsL $2) $0))))) {circleSize} $0))")
    
    T("(loop i 3 (move (*d 1l 3) (/a 1a 4)))")
    T("(loop i 5 (move (*d 1l 5) (/a 1a 5)))")
    T("(loop i infinity (move (*d epsilonDistance 5) (/a epsilonAngle 3)))")
    T("(loop i infinity (move (*d epsilonDistance 9) (/a epsilonAngle 2)))")
    T("(loop i infinity (move (*d epsilonLength i) (*a epsilonAngle 3)))")
    T("(loop i 9 (move (*d 1l i) (/a 1a 4)))")
    T("(move 1d 0a)")
    T("(loop i infinity (move (*d epsilonLength 6) epsilonAngle))")
    T("(loop i infinity (move (*d epsilonLength 8) epsilonAngle))")
    T("(loop k 2 (loop i infinity (move (*d epsilonLength 4) epsilonAngle)))")
    T("(loop k 2 (loop i infinity (move (*d epsilonLength 8) epsilonAngle)))")
    T("(loop s 4 (move (*d 1d 3) (/a 1a 4)))")
    T("(loop s 4 (move (*d 1d 6) (/a 1a 4)))")
    T("""
          (loop j 5
          (move 0d (/a 1a 5))
          (embed (loop i infinity
          (move (*d epsilonLength 6) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength 6) epsilonAngle))))""")
    T("""
              (loop j 5
              (embed (loop s 4 (move (*d 1d 3) (/a 1a 4))))
              (move 0d (/a 1a 5)))""")
    return demos

def rotationalSymmetryDemo():
    demos = []
    def T(source):
        demos.append(manualLogoTask(str(len(demos)), source))
            
    body = {"dashed": "(p (move 1d 0a)) (move 1d 0a) (p (move 1d 0a)) (move 1d 0a)",
            "lonely circle": "(p (move (*d 1d 2) 0a)) (loop k 2 (loop i infinity (move (*d epsilonLength 2) epsilonAngle)))",
            "square dashed": "(p (move 1d 0a)) (loop s 4 (move 1d (/a 1a 4)))",
            "square": "(loop s 4 (move (*d 1d 2) (/a 1a 4)))",
            "semicircle": "(loop i infinity (move (*d epsilonLength 4) epsilonAngle))"}
    for name in body:
        for n in [3,4,5,6,7]:
            T("""
              (loop j %d
              (embed %s)
              (move 0d (/a 1a %d)))"""%(n,body[name],n))
    return demos

def generateLogoDataset(task_dataset, task_dataset_dir, language_dataset, language_dataset_dir):
    """task_dataset: [logo_classic | "logo_unlimited_1300",
    "logo_unlimited_500",
    "logo_unlimited_200"]
        task_dataset_dir: dataset will be saved to {task_dataset_dir}/{task_dataset}/{train | test}. Generates numbered pickle files for each task. 
        Images are saved to {task_dataset_dir}/{task_dataset}/{images}.
        language_dataset_dir: language dataset will be saved to {language_dataset_dir}/{language_dataset}.
    """
    print(f"Generating logo task sdataset: [{task_dataset}] and saving to: [{task_dataset_dir}]")
    if language_dataset is not None:
        print(f"Generating logo dataset language and saving to: [{language_dataset}]")
    
    language_data = None
    if task_dataset == 'logo_classic':
        if language_dataset is not None:
            print("Error: no language defined for logo_classic dataset.")
            assert False
        tasks = makeTasks()
    elif task_dataset == 'logo_unlimited_1000':
        tasks = makeLogoUnlimitedTasks(n_tasks=1000)
    elif task_dataset == 'logo_unlimited_500':
        tasks = makeLogoUnlimitedTasks(n_tasks=500)
    elif task_dataset == 'logo_unlimited_200':
        tasks = makeLogoUnlimitedTasks(n_tasks=200)
    
    writeLogoDataset(tasks, task_dataset, task_dataset_dir)
    if language_dataset is not None:
        writeLanguageDataset(tasks, language_dataset, language_dataset_dir)

def writeLogoDataset(tasks, task_dataset, task_dataset_dir):
    print(f"Writing n={len(tasks)} to {task_dataset_dir}/{task_dataset}")
    from pathlib import Path
    dataset_path = os.path.join(task_dataset_dir, task_dataset)
    image_path = os.path.join(dataset_path, 'images')
    # Remove the dataset if it exists.

    Path(image_path).mkdir(parents=True, exist_ok=True)
    train_tasks = [t for t in tasks if t.mustTrain]
    test_tasks = [t for t in tasks if not t.mustTrain]
    print(f"Writing n=[{len(train_tasks)}] train and n=[{len(test_tasks)}] test.")
    for split, tasks in [("train", train_tasks), ("test", test_tasks)]:
        split_path = os.path.join(dataset_path, split)
        Path(split_path).mkdir(parents=True, exist_ok=True)
        for i, task in enumerate(tasks):
            task_path = os.path.join(split_path, f"{i}.p")
            with open(task_path, 'wb') as f:
                dill.dump(task, f)
    
    for split, tasks in [("train", train_tasks), ("test", test_tasks)]:
        split_path = os.path.join(image_path, split)
        Path(split_path).mkdir(parents=True, exist_ok=True)
        saveVisualizedTasks(tasks, split_path)   

def loadLogoDataset(task_dataset, task_dataset_dir, om_original_ordering):
    dataset_path = os.path.join(task_dataset_dir, task_dataset)
    tasks = {"train": [], "test": []}
    for split in ("train", "test"):
        split_path = os.path.join(dataset_path, split)
        task_files = sorted([os.path.join(split_path, t) for t in os.listdir(split_path) if '.p' in t], key=lambda p:os.path.basename(p))
        for task_file in task_files:
            with open(task_file, 'rb') as f:
                t = dill.load(f)
                t.nearest_name = None
                tasks[split].append(t)
    if bool(om_original_ordering):
        # Silly hack to load the original ordering used in the paper @CathyWong
        with open(os.path.join(dataset_path, 'ordering_om')) as f:
            ordering = [l.strip() for l in f.readlines()]
        name_to_task = {
            t.name : t
            for t in tasks['train']
        }
        tasks["train"] = [name_to_task[n] for n in ordering]
    return tasks["train"], tasks["test"]
        
    
def writeLanguageDataset(tasks, language_dataset, language_dataset_dir):
    """
    Writes synthetic language to a dict of {task_name : [language]}
    """
    
    print(f"Writing synthetic language to {language_dataset_dir}/{language_dataset}/synthetic")
    from pathlib import Path
    dataset_path = os.path.join(language_dataset_dir, language_dataset, 'synthetic')
    
    # Write a vocabulary.
    vocabulary = set()
    for t in tasks:
        for sentence in t.language:
            vocabulary.update(sentence.split())
    print(f"Writing a synthetic vocabulary of len=[{len(vocabulary)}]")
    
    # Write the language data.
    train_tasks = [t for t in tasks if t.mustTrain]
    test_tasks = [t for t in tasks if not t.mustTrain]
    for split, tasks in [("train", train_tasks), ("test", test_tasks)]:
        split_path = os.path.join(dataset_path, split)
        Path(split_path).mkdir(parents=True, exist_ok=True)
        language_data = {
            t.name : t.language
            for t in tasks
        }
        with open(os.path.join(split_path, 'language.json'), 'w') as f:
            json.dump(language_data, f)
        with open(os.path.join(split_path, 'vocab.json'), 'w') as f:
            json.dump(sorted(list(vocabulary)), f)
            
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
    
def makeLogoUnlimitedTasks(resolution=[28,128], n_tasks=None, split=0.75):
    """
    Generates logo_unlimited tasks.
        n_tasks = bounds the number of compositional tasks.
        Compositional tasks will be split across train and test.
    """
    tasks = defaultdict(list)
    
    def T(name, source, needToTrain=False, supervise=False, language=None, template_name=None):
        tasks[template_name].append(manualLogoTask(name, source, supervise=supervise,
        needToTrain=needToTrain, resolution=resolution,
        language=language))
    
    def copy_tasks(original_tasks, n_needed):
        tasks = []
        for i in range(n_needed):
            random.shuffle(original_tasks)
            copy = original_tasks[0]
            tasks.append(manualLogoTask(f"{copy.name}_copy_{i}", copy.expression, supervise=False,
            needToTrain=False, resolution=resolution,
            language=copy.language[0]))
        return tasks

    
    # Language constants for synthetic language grammar.
    LANGUAGE_CONSTANTS = {
        'size': {1 : "small", 2: "medium", 5 : "big"},
        'separation' : {1: "short", 2: "medium", 5: 'big'}
    }
    
    #### Base shape primitives. All in training set.
    snowflake_bodies = {
        'ngon' : {},
        'circles' : {},
        'line' : {}
    }
    snowflake_small_bodies = set() # Smallest shapes
    
    def slant(n):
        return f"(move 0d (/a 1a {n}))"
    
    # N-gon base shapes. 
    # a {small  | medium } {n} gon
    language = None
    for n in range(3,10):
        for length in [1, 2]: 
            l = f"(*d 1l {length})"
            body = f"""
            (loop i {n}
              (move {l} (/a 1a {n})))
            """
            size = LANGUAGE_CONSTANTS['size'][length]
            if n == 3:
                shape_name = 'triangle'
            elif n == 4:
                shape_name = 'square'
            else:
                shape_name = f"{n} gon"
            name = f"{size} {shape_name}"  
            language = f"a {name}"
            T(language, body, needToTrain=True, language=language, template_name='base')
            if length == 1 or (length == 2 and n < 6):
                snowflake_bodies['ngon'][name] = body
            if length == 1 and n < 6:
                snowflake_small_bodies.add(name)
    # Circle and semi-circle base shapes. 
    # a {small  | medium | large} {semicircle | circle}
    for n in [1,2,5]:
        circles = [
            (f'{LANGUAGE_CONSTANTS["size"][n]} semicircle', 
            f"(loop i infinity (move (*d epsilonLength {n}) epsilonAngle))"),
            (f'{LANGUAGE_CONSTANTS["size"][n]} circle',
            """
            (loop i infinity
            (move (*d epsilonLength %d) epsilonAngle))
            (loop i infinity
            (move (*d epsilonLength %d) epsilonAngle))
            """%(n,n)
            )
        ]
        for (name, body) in circles:
            full_body = f"({body})"
            language = f"a {name}"
            T(language,full_body,needToTrain=True, language=language, template_name='base')
            if n <=2:
                snowflake_bodies['circles'][name] = body
                snowflake_small_bodies.add(name)
    
    # Line segment base shapes. 
    name = "short line"
    body = "(move 1d 0a)"
    language = f"a {name}"
    T(language, body, needToTrain=True, language=language, template_name='base')
    snowflake_bodies['line'][name] = body
    snowflake_small_bodies.add(name)
    
    
    ## Template 1: 'separated by a space', 'connected by a line'
    # Build base shapes next to other shapes
    for body_type_name in snowflake_bodies:
        for body_base_name, body_base in snowflake_bodies[body_type_name].items():
            for body_type_name_2 in snowflake_bodies:
                for body_base_name_2, body_base_2 in snowflake_bodies[body_type_name_2].items():
                    if body_base_name not in snowflake_small_bodies and body_base_name_2 not in snowflake_small_bodies: pass
                    for dash_dist in [2, 5]:
                        if dash_dist <= 2:
                            # Sometimes these overlap
                            name = f"a {body_base_name} next to a {body_base_name_2}"
                        else:
                            sep_name = LANGUAGE_CONSTANTS['separation'][dash_dist]
                            name = f"a {body_base_name} separated by a {sep_name} space from a {body_base_name_2}"
                        T(name,
                          f"""
                          ({body_base}
                          (p (move (*d 1d {dash_dist}) 0a))
                          {body_base_2})
                          """,
                          needToTrain=False, 
                          language=name,
                          template_name='separated by distance')
                    for dash_dist in [2, 5]:
                        sep_name = LANGUAGE_CONSTANTS['separation'][dash_dist]
                        name = f"a {body_base_name} connected by a {sep_name} line to a {body_base_name_2}"
                        T(name,
                          f"""
                          ({body_base}
                          (move (*d 1d {dash_dist}) 0a)
                          {body_base_2})
                          """,
                          needToTrain=False,
                          language=name,
                          template_name='connected_by_a_line')
    
    #### Template 2: 'In a row'
    for body_type_name in snowflake_bodies:
        for body_base_name, body_base in snowflake_bodies[body_type_name].items():
            if body_base_name in snowflake_small_bodies:
                for n in [3,4,5,6]:
                    name = f"{n} {body_base_name} s in a row"
                    row = f"""
                    (loop j {n}
                    (embed {body_base})
                    (p (move 1d 0a)))"""
                    T(name, row, needToTrain=False, language=name,
                    template_name='in_a_row')
    
    # # Build snowflakes
    for rotations in [3,5,6,7,8]:
        for body_type_name in snowflake_bodies:
            for body_base_name, body_base in snowflake_bodies[body_type_name].items():
                # Template: snowflakes with no arm connecting them to the center.
                body = f"{body_base}"
                name = f"{rotations} sided snowflake with a {body_base_name} as arms"
                T(name,
                  """
                  (loop j %d
                  (embed %s)
                  (move 0d (/a 1a %d)))"""%(rotations,body,rotations),
                  needToTrain=False,
                  template_name="snowflake shape arms",
                  language=name)
    
                # Template 3: snowflakes with an 'arm' connecting them to the center
                # Snowflakes with an 'arm' connecting them to the center.
                for arm_length in [1, 2]:
                    body = f"(move (*d 1d {arm_length}) 0a) {body_base}"
                    name = f"{rotations} sided snowflake with a {LANGUAGE_CONSTANTS['separation'][arm_length]} line and a {body_base_name} as arms"
                    T(name,
                      """
                      (loop j %d
                      (embed %s)
                      (move 0d (/a 1a %d)))"""%(rotations,body,rotations),
                      needToTrain=False,
                      template_name="snowflake line shape arms",
                      language=name)
    
                # Snowflakes with distance from the center.
                for dash_dist in [1, 2]:
                    name = f"{rotations} sided snowflake with a {LANGUAGE_CONSTANTS['separation'][dash_dist]} space and a {body_base_name} as arms"
                    body = f"(p (move (*d 1d {dash_dist}) 0a)) {body_base}"
                    T(name,
                      """
                      (loop j %d
                      (embed %s)
                      (move 0d (/a 1a %d)))"""%(rotations,body,rotations),
                      template_name="snowflake space shape arms",
                      needToTrain=False,
                      language=name)
                # Snowflakes with a dash, line, then a shape.
                if body_base_name in snowflake_small_bodies:
                    name = f"{rotations} sided snowflake with a short space and a short line and a short space and a {body_base_name} as arms"
    
                    body = f"(p (move 1d 0a)) (move 1d 0a) (p (move 1d 0a)) {body_base}"
                    T(name,
                      """
                      (loop j %d
                      (embed %s)
                      (move 0d (/a 1a %d)))"""%(rotations,body,rotations),
                      needToTrain=False,
                      language=name,
                      template_name="snowflake space line shape arms")
                    # Cross product of two shapes.
                    for body_type_name_2 in snowflake_bodies:
                        for body_base_name_2, base_2 in snowflake_bodies[body_type_name_2].items():
                            if body_base_name_2 in snowflake_small_bodies:
                                if body_type_name_2 == 'ngon' and '1l' in body_base_name_2: continue # Too hard to distinguish
                                body = f"(p (move 1d 0a)) {body_base}  (p (move 1d 0a)) {base_2}"
    
                                if body_base_name == body_base_name_2:
                                    name = f"{rotations} sided snowflake with 2 {body_base_name} s as arms"
                                else:
                                    name = f"{rotations} sided snowflake with a {body_base_name} and a {body_base_name_2} as arms"
    
                                T(name,
                                """
                                (loop j %d
                                (embed %s)
                                (move 0d (/a 1a %d)))"""%(rotations,body,rotations),
                                needToTrain=False,
                                language=name,
                                template_name="snowflake two shape arms")
    
    ### Miscellaneous other tasks.
    name = "a vertical short line"
    T(name, "((move 0d (/a 1a 4)) (move 1d 0a))",
      needToTrain=True, template_name='base', language=name)
    
    # Note the off by one in the loop
    for i in [6,7,8,9]:
        name = f"a greek spiral with {i-1} turns"
        T(name,
          """
          (loop i %d
          (move (*l 1l i) (/a 1a 4)))
          """%i,
          needToTrain=False,
          template_name="greek spiral",
          language=name)
    
    for i in [5,7,9]:
        name = f"a {i} pointed star"
        T(name,
          """
          (loop i %d (move (*d 1d 4) (-a (/a 1a 2) (/a (/a 1a 2) %s))))
          """%(i,i),
          needToTrain=False,
          template_name="star",
          language=name)
    
    # Note that there is an off by one in the number of circles actually drawn
    for n in [4,5,6,7,8,9]:
        name = f"{n-1} concentric square s"
        T(name,
          """
          (for i %d
          (embed (loop j 4 (move (*d 1d i) (/a 1a 4)))))
          """%n,
          needToTrain=False,
          language=name,
          template_name='concentric squares')
    
    # Note that there is an off by one in the number of circles actually drawn
    for n in [3,4,5,6,7,8,9]:
        name = f"{n-1} concentric circle s"
        T(name,
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle)))"""%n,
          needToTrain=False,
          language=name,
          template_name='concentric circles')
    for n in [3,4,5,6,7,8]:
        name = f"a {n} stepped staircase"
        T(name,
          """
          (loop i %d
          (move 1d (/a 1a 4))
          (move 1d (/a 1a 4))
          (move 0d (/a 1a 2)))
          """%n,
          needToTrain=False,
          language=name,
          template_name='staircase')
    for n in [2,3,4,5]:#range(1,5):
        name = f"a {n} stepped zigzag"
        T(name,
          """
          ((move 0d (/a 1a 8))
          (loop i %d
          (move 1d (/a 1a 4)) 
          (move 1d (+a (/a 1a 2) (/a 1a 4)))))
          """%n,
         needToTrain=False,
         language=name,
         template_name='zigzag')
    
    with random_seed(42): 
        total_tasks = sum(len(tasks[template_name]) for template_name in tasks)
        n_task_templates = len([k for k in tasks.keys() if k != 'base'])
        sparse_templates = [template_name for template_name in tasks if len(tasks[template_name]) <= 10 and template_name != 'base']
        
        # First, get all the tasks that we need to train on.
        must_train_templates = ["base"] + sparse_templates
        other_templates = [template_name for template_name in tasks if template_name not in must_train_templates]
        
        base_train_tasks = []
        train_tasks, test_tasks = {}, {}
        for template_name in must_train_templates:
            template_tasks = tasks[template_name]
            if template_name == 'base':
                base_train_tasks = template_tasks
            else:
                if len(template_tasks) <= 5:
                    num_test_tasks = 1
                elif len(template_tasks) <= 10:
                    num_test_tasks = 4
                random.shuffle(template_tasks)
                test_tasks[template_name] = template_tasks[:num_test_tasks]
                train_tasks[template_name] = template_tasks[num_test_tasks:]
        for template_name in other_templates:
            template_tasks = tasks[template_name]
            # Allocate train and testing tasks separately
            num_test_tasks = 12
            random.shuffle(template_tasks)
            test_tasks[template_name] = template_tasks[:num_test_tasks]
            train_tasks[template_name] = template_tasks[num_test_tasks:]
        
        final_train_tasks = []
        final_test_tasks = []
        # Now, go through and take the appropriate number of tasks until we have enough.
        total_test_tasks = sum([len(test_tasks[t]) for t in test_tasks])
        for template_name in test_tasks:
            final_test_tasks += test_tasks[template_name]
            percentage_test = len(test_tasks[template_name]) / total_test_tasks
            num_train_tasks = int(percentage_test * n_tasks) + 1
            available_train_tasks = train_tasks[template_name]
            if len(available_train_tasks) >= num_train_tasks:
                print(f"Taking {num_train_tasks}/{len(available_train_tasks)} train from {template_name}")
                random.shuffle(available_train_tasks)
                final_train_tasks += available_train_tasks[:num_train_tasks]
            else:
                # Make more of the tasks.
                needed_tasks = num_train_tasks - len(available_train_tasks)
                print(f"Taking available {len(available_train_tasks)} and generating {needed_tasks} from {template_name}")
                final_train_tasks += available_train_tasks
                final_train_tasks += copy_tasks(available_train_tasks, needed_tasks)
        
        # Truncate tasks as necessary.
        print(f"Generated n={len(final_train_tasks)} train tasks, truncating to {n_tasks}")
        print(f"Generated n={len(final_test_tasks)} test tasks.")
        random.shuffle(final_train_tasks)
        final_train_tasks = base_train_tasks + final_train_tasks
        final_train_tasks = final_train_tasks[:n_tasks]
        
        for test_task in final_test_tasks:
            test_task.mustTrain = False
        for train_task in final_train_tasks:
            train_task.mustTrain = True
        return final_train_tasks + final_test_tasks

def sampleSupervised(tasks, n):
    """Heuristic for sampling n supervised; requires that we have at least one from each tricky class."""
    must_have = "snowflake"
    while True:
        selection = random.sample(range(0, len(tasks)), n)
        
        has_must_have = False
        for s in selection:
            if must_have in tasks[s].name:
                has_must_have = True
        if has_must_have:
            for s in selection:
                tasks[s].use_supervised = True
            return tasks

def manualLogoTasks(resolution=[28,128]):
    tasks = []
    def T(name, source, needToTrain=False, supervise=False):
        tasks.append(manualLogoTask(name, source, supervise=supervise,
                                    needToTrain=needToTrain, resolution=resolution))
    if False:
        for d,a,s in [('1l','0a','(loop i infinity (move epsilonLength epsilonAngle))'),
                      ('epsilonLength','0a','(loop i infinity (move epsilonLength epsilonAngle))'),
                      ('(*d 1l 3)','0a','(move 1l 0a)'),
                      ('epsilonLength','0a','(move (*d 1l 2) 0a)'),
                      ('(*d epsilonLength 9)','0a','(move epsilonLength 0a)'),
                      ('(/d 1l 2)','0a','(move 1l 0a)')]:
            #            'epsilonLength']:
            # for a in ['epsilonAngle','0a']:
            #     for s in ['(move 1l 0a)',
            #               '(move epsilonLength 0a)',
            #               '(loop i infinity (move epsilonLength epsilonAngle))']:
            #         if d == 'epsilonLength' and s == '(move epsilonLength 0a)': continue
            T("pu: %s/%s/%s"%(d,a,s),
              """
              (pu (move %s %s) pd %s)
              """%(d,a,s))
        return tasks

    def slant(n):
        return f"(move 0d (/a 1a {n}))"

    for n,l,s in [(3,"1l",8),
                  (4,"(*d 1d 3)",None),
                  (5,"1l",None),
                  (6,"(*d 1d 2)",5),
                  (7,"1l",None),
                  (8,"(/d 1d 2)",None)]:
        T(f"{n}-gon {l}{'' if s is None else ' slanted '+str(s)}",
          f"""
          ({'' if s is None else slant(s)}
           (loop i {n}
            (move {l} (/a 1a {n}))))
          """,
          needToTrain=True)
    for n,l,s in [(3,"(*d 1l 2)",None),
                (4,"(*d 1d 4)",None),
                (5,"(*d 1d 2)",None),
                (6,"1l",None),
                (7,"(*d 1d 3)",None),
                (8,"1l",3)]:
        T(f"{n}-gon {l}{'' if s is None else ' slanted '+str(s)}",
          f"""
          ({'' if s is None else slant(s)}
           (loop i {n}
            (move {l} (/a 1a {n}))))
          """,
          needToTrain=False)
        

    T("upwards", "((move 0d (/a 1a 4)) (move 1d 0a))",
      needToTrain=True)
    T("right angle", "((move (*d 1d 2) (/a 1a 4)) (move 1d 0a))",
      needToTrain=True)
    T("right angle epsilon", "((move epsilonLength (/a 1a 4)) (move epsilonLength 0a))",
      needToTrain=True)

    T("line segment", "(move 1d 0a)",
      needToTrain=True)

    T("square slanted by 2pi/3",
      """((move 0d (/a 1a 3))
      (loop k 4 (move 1d (/a 1a 4))))""",
      needToTrain=True)
    T("semicircle slanted by 2pi/5",
      """((move 0d (/a 1a 5))
      (loop i infinity
      (move (*d epsilonLength 4) epsilonAngle)))""",
      needToTrain=True)
    T("Greek spiral slanted by 2pi/6",
      """((move 0d (/a 1a 6))
      (loop i 7 (move (*l 1l i) (/a 1a 4))))""",
      needToTrain=True)
    T("Hook slanted by 2pi/7",
      """((move 0d (/a 1a 7))
      (move 1d 0a)
      (loop i infinity
      (move (*d epsilonLength 4) epsilonAngle)))""",
      needToTrain=True)
    T("""slanted line""",
      """((move 0d (/a 1a 8))
      (move (*d 1l 3) 0a))""",
      needToTrain=True)
    

    for i in [6,7,8,9]:
        T("Greek spiral %d"%i,
          """
          (loop i %d
          (move (*l 1l i) (/a 1a 4)))
          """%i,
          needToTrain=i in [7,8])
    for i in [2,3,4,5]:
        T("smooth spiral %d"%i,
          """
          (loop i infinity 
          (move (*d epsilonLength i) (*a epsilonAngle %d)))
          """%i,
          needToTrain=i in [3,5])

    T("smooth spiral 4 slanted by 2pi/2",
      """
          ((move 0d (/a 1a 2))
      (loop i infinity 
          (move (*d epsilonLength i) (*a epsilonAngle 4))))
      """,
      needToTrain=True)

    for i in [3,5,7,9]:
        T("star %d"%i,
          """
          (loop i %d (move (*d 1d 4) (-a (/a 1a 2) (/a (/a 1a 2) %s))))
          """%(i,i),
          needToTrain=i in [5,9])

    T("leaf iteration 1.1",
      """
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      """,
      needToTrain=True)
    T("leaf iteration 1.2",
      """
      ((move 0d (/a 1a 2))
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2))))
      """,
      needToTrain=True)
    T("leaf iteration 2.1",
      """
      (loop n 2
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      (move 0d (/a 1a 4)))
      """,
      needToTrain=True)
    T("leaf iteration 2.2",
      """
      ((move 0d (/a 1a 2))
      (loop n 2
      (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
      (move 0d (/a 1a 4))))
      """,
      needToTrain=True)
    for n in range(3,8):
        T("flower %d"%n,
          """
          (loop j %d
          (loop n 2
          (loop i infinity (move epsilonDistance (/a epsilonAngle 2)))
          (move 0d (/a 1a 4)))
          (move 0d (/a 1a %d)))
          """%(n,n),
          needToTrain=n in range(3,5))        


    for n in [5,6]:
        T("staircase %d"%n,
          """
          (loop i %d
          (move 1d (/a 1a 4))
          (move 1d (/a 1a 4))
          (move 0d (/a 1a 2)))
          """%n,
          needToTrain=n in [5])

    for n in range(1,6):
        T("blocks zigzag %d"%n,
          """
          (loop i %d
          (move 1d (/a 1a 4)) (move 1d (/a 1a 4))
          (move 1d (+a (/a 1a 2) (/a 1a 4))) (move 1d (+a (/a 1a 2) (/a 1a 4))))
          """%n,
          needToTrain=n in [1,2,3])
    for n in [3,4]:#range(1,5):
        T("diagonal zigzag %d"%n,
          """
          ((move 0d (/a 1a 8))
          (loop i %d
          (move 1d (/a 1a 4)) 
          (move 1d (+a (/a 1a 2) (/a 1a 4)))))
          """%n,
          needToTrain=n == 4)

    

    for n in [1,2,3,4,5,6]:
        T("right semicircle of size %d"%n,
          """
          (loop i infinity
          (move (*d epsilonLength %d) (-a 0a epsilonAngle)))
          """%n,
          needToTrain=n%2 == 0)
        T("left semicircle of size %d"%n,
          f"""
          ({'' if n != 1 else slant(8)}
           (loop i infinity
            (move (*d epsilonLength {n}) epsilonAngle)))
          """,
          needToTrain=n%2 == 1)
        T("circle of size %d"%n,
              """
              ((loop i infinity
              (move (*d epsilonLength %d) epsilonAngle))
              (loop i infinity
              (move (*d epsilonLength %d) epsilonAngle)))
              """%(n,n),
          needToTrain=n in [1,4,3,5,6])

    for n in [5,6]:
        T("%d enclosed circles"%n,
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength j) epsilonAngle)))"""%n,
          needToTrain=n == 5)

    for n,l in [(4,2),
                (5,3),
                (6,4),
                (3,1)]:
        T("%d-circle flower l=%d"%(n,l),
          """
          (loop j %d
          (move 0d (/a 1a %d))
          (embed (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))))"""%(n,n,l,l),
          needToTrain=(n,l) in [(6,4),(3,1)])

    for n,l in [(3,1),(2,2),(1,3),
                (2,1),(1,2),(1,1)]:
        T("%d-semicircle sequence L=%d"%(n,l),
          """
          (loop j %d
          (loop i infinity
          (move (*d epsilonLength %d) epsilonAngle))
          (loop i infinity
          (move (*d epsilonLength %d) (-a 0a epsilonAngle))))
          """%(n,l,l),
          needToTrain=(n,l) in [(3,1),(2,2),(1,3)])

    for n,l in [(2,"1d"),
                (3,"1d")]:
        T("row of %d circles"%n,
          """
          (loop j %d
          (embed (loop k 2 (loop i infinity (move epsilonLength epsilonAngle))))
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 2)
    for n,l in [(2,"1d"),
                (3,"1d")]:
        T("row of %d lines"%n,
          """
          (loop j %d
          (move 1d 0a)
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 2)
    T("line next to semicircle",
      """
      ((move 1d 0a) (p (move 1d 0a)) (loop i infinity (move epsilonLength epsilonAngle)))
      """,
      needToTrain=True)
    for n,l in [(3,"(/d 1d 2)"),
                (4,"(/d 1d 3)")]:
        T("%d dashed lines of size %s"%(n,l),
          """(loop i %d (p (move 1d 0a)) (move %s 0a))"""%(n,l),
          needToTrain=n == 3)
    T("broken circle",
      """
      ((loop i infinity (move epsilonLength epsilonAngle)) (p (move 1d 0a)) (loop i infinity (move epsilonLength epsilonAngle)))
      """,
      needToTrain=True)
    T("circle next to semicircle",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (loop i infinity (move epsilonLength epsilonAngle)))
      """,
      needToTrain=True)
    T("semicircle next to square",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (loop i infinity (move 1d (/a 1a 4))))
      """,
      needToTrain=False)
    T("circle next to square",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (loop i infinity (move 1d (/a 1a 4))))
      """,
      needToTrain=False)
    T("circle next to line",
      """
      ((loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))
      (p (move 1d 0a))
      (move 1d 0a))
      """,
      needToTrain=True)
    T("line next to circle",
      """
      ((move 1d 0a)
      (p (move 1d 0a))
      (loop i infinity (move epsilonLength epsilonAngle))
      (loop i infinity (move epsilonLength epsilonAngle))      
      (move 1d 0a))
      """,
      needToTrain=True)
    for n,l in [(4,"1d"),
                (5,"1d")]:
        T("row of %d dashes"%n,
          """
          (loop j %d
          (embed (move 0d (/a 1a 4)) (move 1d 0a))
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 4)        
    for n,l in [(5,"1d"),(6,"1d")]:
        T("row of %d semicircles"%n,
          """
          (loop j %d
          (embed (loop i infinity (move epsilonLength epsilonAngle)))
          (p (move %s 0a)))"""%(n,l),
          needToTrain=n == 5)

    with random_seed(42): # carefully selected for maximum entropy
        for n in [3,4,5,6,7]:
            body = {"empty": "(move 1d 0a)",
                    "spiral": "(loop i infinity (move (*d epsilonLength i) (*a epsilonAngle 2)))",
                    "dashed": "(p (move 1d 0a)) (move 1d 0a)",
                    "circle": "(move 1d 0a) (loop k 2 (loop i infinity (move epsilonLength epsilonAngle)))",
                    "lonely circle": "(p (move 1d 0a)) (loop k 2 (loop i infinity (move epsilonLength epsilonAngle)))",
                    "square dashed": "(p (move 1d 0a)) (loop s 4 (move 1d (/a 1a 4)))",
                    "square": "(move 1d 0a) (loop s 4 (move 1d (/a 1a 4)))",
                    "close large semicircle": "(loop i infinity (move (*d epsilonLength 2) epsilonAngle))",
                    "close semicircle": "(loop i infinity (move epsilonLength epsilonAngle))",
                    "semicircle": "(move 1d 0a) (loop i infinity (move epsilonLength epsilonAngle))",
                    "double dashed": "(p (move 1d 0a)) (move 1d 0a) (p (move 1d 0a)) (move 1d 0a)",
                    "Greek": "(loop i 3 (move (*l 1l i) (/a 1a 4)))"}
            for name in body:
                if name == "spiral" and n not in [3,5]: continue
                if name == "square" and n not in [5,3,6,7]: continue
                if name == "semicircle" and n not in [5,3,4,6]: continue
                if name == "Greek" and n not in [3,5]: continue
                if name == "double dashed" and n not in [6,4,3]: continue
                
                mustTrain = False

                mustTrain = mustTrain or (n == 3 and name == "Greek")
                mustTrain = mustTrain or (n == 7 and name == "empty")
                mustTrain = mustTrain or (n == 5 and name == "dashed")
                mustTrain = mustTrain or (n == 7 and name == "circle")
                mustTrain = mustTrain or (n == 6 and name == "circle")
                mustTrain = mustTrain or (n == 6 and name == "lonely circle")
                mustTrain = mustTrain or (n == 5 and name == "square")
                mustTrain = mustTrain or (n == 7 and name == "square")
                mustTrain = mustTrain or (n == 5 and name == "semicircle")
                mustTrain = mustTrain or (n == 3 and name == "square dashed")
                mustTrain = mustTrain or (n == 6 and name == "close semicircle")
                mustTrain = mustTrain or (n == 5 and name == "close large semicircle")
                mustTrain = mustTrain or (n == 3 and name == "spiral")
                mustTrain = mustTrain or (n == 6 and name == "double dashed")
                mustTrain = mustTrain or (n == 3 and name == "double dashed")
                #mustTrain = mustTrain or (n == 6 and name == "empty")

                #mustTrain = mustTrain or (random.random() < 0.07) # calibrated to give 70 training tasks
                

                # # cap number of super easy snowflakes
                # if name == "empty" and n not in [7]: mustTrain = False
                # if name == "dashed" and n not in [4]: mustTrain = False
                

                T("%d-%s snowflake"%(n,name),
                  """
                  (loop j %d
                  (embed %s)
                  (move 0d (/a 1a %d)))"""%(n,body[name],n),
                  needToTrain=mustTrain)

    for n in [3,4]:#2,3,4]:
        T("%d-row of squares"%n,
          """
          (loop i %d
          (embed (loop k 4 (move 1d (/a 1a 4))))
          (move 1d 0a))
          """%n,
          needToTrain=n == 4)
    T("2x2 grid",
    """
    (for x 2 (embed (for y 2
       (embed (loop k 4 (move 1d (/a 1a 4))))
       (move 1d 0a)))
       (move 0d (/a 1a 4)) (move 1d (-a 0a (/a 1a 4))))
    """)
    T("slanted squares",
      """
      ((embed (loop k 4 (move 1d (/a 1a 4))))
      (move 0d (/a 1a 8))
      (loop k 4 (move 1d (/a 1a 4))))
      """)
    for l in range(1,6):
        T("square of size %d"%l,
          """
          (for i 4
          (move (*d 1d %d) (/a 1a 4)))
          """%l,
          needToTrain=l in range(4))
    for n in [5,7]:
        T("%d-concentric squares"%n,
          """
          (for i %d
          (embed (loop j 4 (move (*d 1d i) (/a 1a 4)))))
          """%n,
          needToTrain=n == 5)
    return tasks

def montageTasks(tasks, prefix="", columns=None, testTrain=False):
    import numpy as np
    
    w = 128
    arrays = [t.highresolution for t in tasks]
    for a in arrays:
        assert len(a) == w*w

    if testTrain:
        arrays = [a for a,t in zip(arrays, tasks) if t.mustTrain ] + [a for a,t in zip(arrays, tasks) if not t.mustTrain ]
        
    arrays = [np.array([a[i:i + w]
                        for i in range(0, len(a), w) ])
              for a in arrays]
    i = montage(arrays, columns=columns)
    try:
        import imageio     
        imageio.imwrite('/tmp/%smontage.png'%prefix, i)
        if testTrain:
            trainingTasks = arrays[:sum(t.mustTrain for t in tasks)]
            testingTasks = arrays[sum(t.mustTrain for t in tasks):]
            random.shuffle(trainingTasks)
            random.shuffle(testingTasks)
            arrays = trainingTasks + testingTasks
        else:
            random.shuffle(arrays)
        imageio.imwrite('/tmp/%srandomMontage.png'%prefix, montage(arrays, columns=columns))
    except:
        import scipy.misc        
        scipy.misc.imsave('/tmp/%smontage.png'%prefix, i)
        if testTrain:
            trainingTasks = arrays[:sum(t.mustTrain for t in tasks)]
            testingTasks = arrays[sum(t.mustTrain for t in tasks):]
            random.shuffle(trainingTasks)
            random.shuffle(testingTasks)
            arrays = trainingTasks + testingTasks
        else:
            random.shuffle(arrays)
        scipy.misc.imsave('/tmp/%srandomMontage.png'%prefix, montage(arrays, columns=columns))

def demoLogoTasks():
    import scipy.misc
    import numpy as np

    g0 = Grammar.uniform(primitives, continuationType=turtle)
    eprint("dreaming into /tmp/dreams_0...")
    N = 1000
    programs = [ p
                     for _ in range(N)
                     for p in [g0.sample(arrow(turtle,turtle),
                                         maximumDepth=20)]
                     if p is not None]
    os.system("mkdir  -p /tmp/dreams_0")
    for n,p in enumerate(programs):
        with open(f"/tmp/dreams_0/{n}.dream","w") as handle:
            handle.write(str(p))
    drawLogo(*programs, pretty=True, smoothPretty=False,
             resolution=512,
             filenames=[f"/tmp/dreams_0/{n}_pretty.png"
                        for n in range(len(programs)) ],
             timeout=1)
    
    if len(sys.argv) > 1:
        tasks = makeTasks(sys.argv[1:],proto=False)
    else:
        tasks = makeTasks(['all'],proto=False)
    montageTasks(tasks,columns=16,testTrain=True)
    for n,t in enumerate(tasks):
        a = t.highresolution
        w = int(len(a)**0.5)
        scipy.misc.imsave('/tmp/logo%d.png'%n, np.array([a[i:i+w]
                                                         for i in range(0,len(a),w) ]))
        logo_safe_name = t.name.replace("=","_").replace(' ','_').replace('/','_').replace("-","_") + ".png"
        #os.system(f"convert /tmp/logo{n}.png -morphology Dilate Octagon /tmp/{logo_safe_name}")
        os.system(f"convert /tmp/logo{n}.png -channel RGB -negate /tmp/{logo_safe_name}")
    eprint(len(tasks),"tasks")
    eprint(sum(t.mustTrain for t in tasks),"need to be trained on")

    for t in dSLDemo():
        a = t.highresolution
        w = int(len(a)**0.5)
        scipy.misc.imsave('/tmp/logoDemo%s.png'%t.name, np.array([a[i:i+w]
                                                                  for i in range(0,len(a),w) ]))
        os.system(f"convert /tmp/logoDemo{t.name}.png -morphology Dilate Octagon /tmp/logoDemo{t.name}_dilated.png")

    tasks = [t for t in tasks if t.mustTrain ]
    random.shuffle(tasks)
    montageTasks(tasks[:16*3],"subset",columns=16)

    montageTasks(rotationalSymmetryDemo(),"rotational")

    
