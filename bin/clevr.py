try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.clevr.main import main, clevr_options
from dreamcoder.domains.clevr.clevrRecognition import ClevrFeatureExtractor
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

if __name__ == '__main__':
    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=1.5,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=ClevrFeatureExtractor,
        pseudoCounts=30.0,
        extras=clevr_options)
    main(arguments)