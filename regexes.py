#analog of list.py for regex tasks. Responsible for actually running the task.

from ec import explorationCompression, commandlineArguments, Task
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeRegexTasks import makeTasks
from regexPrimitives import basePrimitives, altPrimitives
#from program import *
from recognition import HandCodedFeatureExtractor, MLPFeatureExtractor, RecurrentFeatureExtractor, JSONFeatureExtractor
import random
#class MyJSONFeatureExtractor(JSONFeatureExtactor):
	#TODO
#	def _featuresOfProgram(self, program, tp):
		#TODO

class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    H = 16
    USE_CUDA = False
    def tokenize(self,examples):
        def sanitize(l): return [ z if z in self.lexicon else "?"
                                  for z_ in l
                                  for z in (z_ if isinstance(z_, list) else [z_]) ]

        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"]+y+["LIST_END"]
            else:
                y = [y]
            y = sanitize(y)
            if len(y) > self.maximumLength: return None

            serializedInputs = []
            for xi,x in enumerate(xs):
                if isinstance(x, list):
                    x = ["LIST_START"]+x+["LIST_END"]
                else:
                    x = [x]
                x = sanitize(x)
                if len(x) > self.maximumLength: return None
                serializedInputs.append(x)

            tokenized.append((tuple(serializedInputs),y))

        return tokenized
    def __init__(self, tasks):
        self.lexicon = set(flatten((t.examples for t in tasks), abort=lambda x:isinstance(x, str))).union({"LIST_START", "LIST_END", "?"})

        # Calculate the maximum length
        self.maximumLength = POSITIVEINFINITY
        self.maximumLength = max( len(l)
                                  for t in tasks
                                  for xs,y in self.tokenize(t.examples)
                                  for l in [y] + [ x for x in xs ] )
        
        super(LearnedFeatureExtractor, self).__init__(lexicon=list(self.lexicon),
                                                      tasks=tasks,
                                                      cuda=self.USE_CUDA,
                                                      H=self.H,
                                                      bidirectional=True)
class MyJSONFeatureExtractor(JSONFeatureExtractor):
    N_EXAMPLES = 5
    def _featuresOfProgram(self, program, tp):
        e = program.evaluate([])

        examples = []
        if isListFunction(tp):
            sample = lambda: random.sample(xrange(30), random.randint(0, 8))
        elif isIntFunction(tp):
            sample = lambda: random.randint(0, 20)
        else:
            return None
        for _ in xrange(self.N_EXAMPLES*5):
            x = sample()
            try:
                y = e(x)
                #eprint(tp, program, x, y)
                examples.append((x, y))
            except: continue
            if len(examples) >= self.N_EXAMPLES: break
        else:
            return None
        return examples #changed to list_features(examples) from examples


def regex_options(parser):
	parser.add_argument("--maxTasks", type=int,
		default=500,
		help="truncate tasks to fit within this boundary")
	parser.add_argument("--maxExamples", type=int,
		default=10,
		help="truncate number of examples per task to fit within this boundary")
	parser.add_argument("--primitives",
						default="base",
						help="Which primitive set to use",
						choices=["base","alt1"])
	parser.add_argument("--extractor", type=str,
		choices=["hand", "deep", "learned", "json"],
		default="json") #if i switch to json it breaks
	parser.add_argument("--split", metavar="TRAIN_RATIO",
		type=float,
		default=0.2,
		help="split test/train")
	parser.add_argument("-H", "--hidden", type=int,
		default=16,
		help="number of hidden units")
	parser.add_argument("--likelihoodModel", 
						default="probabilistic",
						help="likelihood Model",
						choices=["probabilistic", "all-or-nothing"])

#Lucas recommends putting a struct with the definitions of the primitives here.
#TODO:
#Build likelihood funciton
#modify NN
#make primitives 
#make tasks

if __name__ == "__main__":
	args = commandlineArguments(
        frontierSize=None, activation='sigmoid', iterations=10,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=10.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
        extras=regex_options)

	tasks = makeTasks() #TODO
	eprint("Generated", len(tasks),"tasks")

	maxTasks = args.pop("maxTasks")
	if len(tasks) > maxTasks:
		eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
		random.seed(42)
		random.shuffle(tasks)
		del tasks[maxTasks:]



	maxExamples = args.pop("maxExamples")
	for task in tasks:
		if len(task.examples) > maxExamples:
			task.examples = task.examples[:maxExamples]

	split = args.pop("split")
	test, train = testTrainSplit(tasks, split)
	eprint("Split tasks into %d/%d test/train"%(len(test),len(train)))



    #from list stuff 
	prims = {"base": basePrimitives,
             "alt1": altPrimitives}[args.pop("primitives")]
    
	extractor = {
		"hand": HandCodedFeatureExtractor,
		"learned": LearnedFeatureExtractor,
		"json": MyJSONFeatureExtractor
	}[args.pop("extractor")]

	extractor.H = args.pop("hidden")
	extractor.USE_CUDA = args["cuda"]

	args.update({
		"featureExtractor": extractor,
		"outputPrefix": "experimentOutputs/regex",
		"evaluationTimeout": 0.005,
		"topK": 5,
		"maximumFrontier": 5,
		"solver": "python",
		"compressor": "rust"
	})
    ####


	baseGrammar = Grammar.uniform(prims())

	explorationCompression(baseGrammar, train,
							testingTasks = test,
							**args)




