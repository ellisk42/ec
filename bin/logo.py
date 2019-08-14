
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.logo.main import main, list_options, LogoFeatureCNN
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs


if __name__ == '__main__':
    args = commandlineArguments(
        structurePenalty=1.5,
        recognitionTimeout=3600,
        a=3,
        topK=2,
        iterations=10,
        useRecognitionModel=True,
        helmholtzRatio=0.5,
        featureExtractor=LogoFeatureCNN,
        maximumFrontier=5,
        CPUs=numberOfCPUs(),
        pseudoCounts=30.0,
        activation="tanh",
        extras=list_options)
    main(args)
