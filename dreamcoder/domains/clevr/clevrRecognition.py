"""
clevrRecognition.py | Author: Catherine Wong.
Contains the domain-specific recognition model(s) used for the CLEVR symbolic scene-graph dataset.

This feature extractor is used as a front-end encoder for the I/O examples on a given task. 
It takes the place of the 'featureExtractor' for sleep_recognition in dreamcoder.py

All featureExtractors must follow a featureExtractor(tasks, testingTasks=testingTasks, cuda=cuda) API. They have been overloaded to also contain additional attributes and methods that must be defined:
    featuresOfTask(task) that runs the forward pass for the actual features.
    
    recomputeTasks attribute: whether to 'recompute' I/O examples for a Helmholtz example during training. If true, we will continually 'clear' and resample input/output entries.
    taskOfProgram(program, type): if defined, generates a 'task' for a given Helmholtz program. Note that this involves actually executing the program itself.
    tasksOfPrograms(prgrams, types): if defined, performs taskOfProgram in a batch.
They are passed to the RecognitionModel defined in recognition.py, which assembles the full stack architecture.
"""
import dreamcoder.domains.clevr.makeClevrTasks as makeClevrTasks
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.type import *
from dreamcoder.domains.clevr.clevrPrimitives import *

class ClevrFeatureExtractor(RecurrentFeatureExtractor):
    """
    Domain-specific recognition model for the CLEVR scene tasks.
    This also contains domain-specific flags used for serializing the tasks themselves.
    
    tasks: a list of all of the tasks. This is used to extract all of their I/O examples, which we will use to execute Helmholtz programs later on.
    
    """
    special = 'clevr' # Special flag passed to the OCaml Helmholtz enumerator.
    serialize_special = makeClevrTasks.serialize_clevr_object # Special flag passed to the OCaml Helmholtz enumerator.
    maximum_helmholtz = 5000  # Special flag passed to cap how many Helmholtz we enumerate.
    
    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.recomputeTasks = True
        self.useTask = True # This means that the tokenize function is called on a Task object, not just examples.
        self.max_examples = 2
        self.max_examples_length = 200
        
        # Lexicon for all tokens we can ever see in the I/O examples.
        lexicon = clevr_lexicon() + ["OBJ_START", "OBJ_END"]
        super(ClevrFeatureExtractor, self).__init__(lexicon=lexicon,
                                                      H=64,
                                                      tasks=tasks, \
                                                      bidirectional=True,
                                                      cuda=cuda,
                                                      helmholtzTimeout=0.5,
                                                      helmholtzEvaluationTimeout=0.25)
                                                      
    def tokenize(self, task):
        """Tokenizes the I/O examples of a single task. In particular, a task contains examples in the form (xs, y) where (xs) is a single list of objects, and y is either an attribute, or itself a list of objects.
        
        """
        tokenized = []
        return_type, _ = makeClevrTasks.infer_return_type([task.examples[0][-1]])
        def tokenize_obj_list(obj_list):
            flattened = []
            for o in obj_list:
                o_attrs = []
                for k, v in o.items():
                    if k in ["id", "color", "shape", "size", "material"]:
                        o_attrs += [k, str(v)]
                    else: # Relations
                        o_attrs += [k] + [str(rel) for rel in v]
                flattened += ["OBJ_START"] + o_attrs + ["OBJ_END"]
            return flattened
        def blank_task():
            blank_xs = (["OBJ_START", "OBJ_END"],)
            blank_y = ["OBJ_START", "OBJ_END"]
            return (blank_xs, blank_y)
        
        def example_len(xs, y):
            return sum([len(x) for x in xs]) + len(y)
        
        # To limit recognition backward pass times, we cap the length of these inputs
        for xs, y in task.examples:
            xs = [tokenize_obj_list(obj_list) for obj_list in xs]
            if return_type in [tint, tbool, tclevrsize, tclevrcolor, tclevrshape, tclevrsize, tclevrmaterial]:
                y = [str(y)]
            else:
                y = tokenize_obj_list(y)
            tokenized.append([xs, y])
        sorted_tokenized = sorted(tokenized, key = lambda e: example_len(e[0], e[1]))
        
        sorted_tokenized = [e if example_len(e[0], e[1]) <= self.max_examples_length else blank_task() for e in sorted_tokenized]
        tokenized = sorted_tokenized[:self.max_examples]
        return tokenized

    def taskOfProgram(self, p, tp):
        # Uses the RNN random input sampling
        t = super(ClevrFeatureExtractor, self).taskOfProgram(p, tp)
        if t is not None:
            xs, y = t.examples[0]
            if type(y) == list: # Sort and dedup any object list
                t.examples = [(xs, sort_and_dedup_obj_list(y)) for xs, y in t.examples]
            t.examples = t.examples[:self.max_examples]
        return t