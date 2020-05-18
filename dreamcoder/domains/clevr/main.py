from dreamcoder.utilities import *
from dreamcoder.recognition import *
from dreamcoder.enumeration import *
from dreamcoder.dreamcoder import ecIterator, default_wake_generative

from dreamcoder.domains.clevr.clevrPrimitives import *
from dreamcoder.domains.clevr.makeClevrTasks import *

import os
import datetime
import random


class ClevrFeatureExtractor(RecurrentFeatureExtractor):
    special = 'clevr'
    serialize_special = serialize_clevr_object
    maximum_helmholtz = 20000
    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.recomputeTasks = True
        self.useTask = True
        lexicon = clevr_lexicon() + ["OBJ_START", "OBJ_END"]
        super(ClevrFeatureExtractor, self).__init__(lexicon=lexicon,
                                                      H=64,
                                                      tasks=tasks,
                                                      bidirectional=True,
                                                      cuda=cuda,
                                                      helmholtzTimeout=0.5,
                                                      helmholtzEvaluationTimeout=0.25)
                                                      
    def tokenize(self, task):
        tokenized = []
        return_type, _ = infer_return_type([task.examples[0][-1]])
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
            
        for xs, y in task.examples:
            xs = [tokenize_obj_list(obj_list) for obj_list in xs]
            if return_type in [tint, tbool, tclevrsize, tclevrcolor, tclevrshape, tclevrsize, tclevrmaterial]:
                y = [str(y)]
            else:
                y = tokenize_obj_list(y)
            tokenized.append([xs, y])
        return tokenized

    def taskOfProgram(self, p, tp):
        # Uses the RNN random input sampling
        t = super(ClevrFeatureExtractor, self).taskOfProgram(p, tp)
        if t is not None:
            xs, y = t.examples[0]
            if type(y) == list: # Sort and dedup any object list
                t.examples = [(xs, sort_and_dedup_obj_list(y)) for xs, y in t.examples]
        return t

all_train_questions = [
    "1_zero_hop",
    '1_one_hop',
    '1_compare_integer',
    '1_same_relate',
    '1_single_or',
    '2_remove',
    '2_transform'
]

def clevr_options(parser):
    # Dataset loading options.
    parser.add_argument("--curriculumDatasets", type=str, nargs="*",
                        default=["curriculum"],
                        help="A list of curriculum datasets, stored as JSON CLEVR question files. These will be used in ")
    parser.add_argument("--taskDatasets", type=str, nargs="+",
                        default=all_train_questions,
                        help="Which task datasets to load, stored as JSON CLEVR question files.")
    parser.add_argument("--taskDatasetDir",
                        default="data/clevr",
                        help="Top level directory for the dataset.")
    parser.add_argument("--languageDatasetDir",
                        default="data/clevr/language/")
    parser.add_argument("--trainInputScenes",
                        default="CLEVR_train_scenes_1000",
                        help="Input scene graphs for all of the training questions.")
    parser.add_argument("--testInputScenes",
                        default="CLEVR_val_scenes_500",
                        help="Input scene graphs for all of the test questions.")

    # Primitive loading options.
    parser.add_argument("--primitives",
                        nargs="*",
                        default=["clevr_bootstrap", "clevr_map_transform"],
                        help="Which primitives to use. Choose from: [clevr_original, clevr_bootstrap, clevr_map_transform, clevr_filter, clevr_filter_except, clevr_difference]")
    parser.add_argument("--run_python_test",
                        action='store_true')
    parser.add_argument("--generate_ocaml_definitions",
                        action='store_true')
    parser.add_argument("--run_ocaml_test",
                        action='store_true')
    parser.add_argument("--run_recognition_test",
                        action='store_true')
    parser.add_argument("--iterations_as_epochs",
                        default=True)
                        
def main(args):
    
    # Load the curriculum and datasets.
    curriculum_datasets = args.pop("curriculumDatasets")
    task_dataset_dir=args.pop("taskDatasetDir")
    train_scenes, test_scenes = args.pop("trainInputScenes"), args.pop("testInputScenes")
    
    if len(curriculum_datasets) > 0:
        curriculum, _ = loadCLEVRDataset(task_datasets=curriculum_datasets, task_dataset_dir=task_dataset_dir, train_scenes=train_scenes, test_scenes = test_scenes, seed=args["seed"], is_curriculum=True)
    
    task_datasets = args.pop("taskDatasets")
    train, test = loadCLEVRDataset(task_datasets=task_datasets, task_dataset_dir=task_dataset_dir, train_scenes=train_scenes, test_scenes = test_scenes, seed=args["seed"])
    eprint(f"Loaded datasets: [{task_datasets}]: [{len(train)}] total train and [{len(test)}] total test tasks.")
    
    # Generate language dataset directly from the loaded tasks.
    args.pop("languageDataset")
    languageDataset = curriculum_datasets + task_datasets
    
    # Load the primitives and optionally run tests with the primitive set.
    primitive_names = args.pop("primitives")
    primitives = load_clevr_primitives(primitive_names)
    baseGrammar = Grammar.uniform(primitives)
    
    if args.pop("run_python_test"):
        run_clevr_primitives_test(primitive_names, curriculum)
        assert False
    
    if args.pop("generate_ocaml_definitions"):
        generate_ocaml_definitions(primitive_names)
        assert False
    
    if args.pop("run_ocaml_test"):
        # Test the Helmholtz enumeratio n
        # tasks = [buildClevrMockTask(train[0])]
        tasks = train
        if True:
            from dreamcoder.dreaming import backgroundHelmholtzEnumeration
            print(baseGrammar)
            helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, 
                                                                baseGrammar, 
                                                                timeout=5,
                                                                evaluationTimeout=0.05,
                                                                special='clevr',
                                                                executable='helmholtz',
                                                                serialize_special=serialize_clevr_object,
                                                                maximum_size=20000) # TODO: check if we need special to check tasks later
            f = helmholtzFrontiers()
            helmholtzFrontiers = backgroundHelmholtzEnumeration(train, 
                                                                baseGrammar, 
                                                                timeout=5,
                                                                evaluationTimeout=0.05,
                                                                special='clevr',
                                                                executable='helmholtz',
                                                                serialize_special=serialize_clevr_object,
                                                                maximum_size=20000) # TODO: check if we need special to check tasks later
            f = helmholtzFrontiers()
        if True:
            # Check enumeration.
            tasks = [buildClevrMockTask(train[0])]
            default_wake_generative(baseGrammar, tasks, 
                                maximumFrontier=5,
                                enumerationTimeout=1,
                                CPUs=1,
                                solver='ocaml',
                                evaluationTimeout=0.05)
    
    if args.pop("run_recognition_test"):
        tasks = [buildClevrMockTask(train[0])]
        featurizer = ClevrFeatureExtractor(tasks=tasks, testingTasks=[], cuda=False)
        for t in tasks:
            featurizer.featuresOfTask(t)
    
    
    use_epochs = args.pop("iterations_as_epochs")
    if use_epochs and args["taskBatchSize"] is not None:
        eprint("Using iterations as epochs")
        args["iterations"] *= int(len(train) / args["taskBatchSize"]) 
        eprint(f"Now running for n={args['iterations']} iterations.")

    timestamp = datetime.datetime.now().isoformat()
    # Escape the timestamp.
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(".", "-")
    outputDirectory = "experimentOutputs/clevr/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    evaluationTimeout = 1.0
    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/clevr"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           languageDataset=languageDataset,
                           **args)
    for result in generator:
        pass
        