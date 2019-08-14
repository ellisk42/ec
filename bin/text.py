try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.text.main import main, LearnedFeatureExtractor, text_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs


if __name__ == '__main__':
    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=LearnedFeatureExtractor,
        pseudoCounts=30.0,
        extras=text_options)
    main(arguments)
