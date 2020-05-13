try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.re2.main import main, re2_options, StringFeatureExtractor
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

if __name__ == '__main__':
    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=5,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=StringFeatureExtractor,
        pseudoCounts=30.0,
        extras=re2_options)
    main(arguments)