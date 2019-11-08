try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.regex.main import main, regex_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs


if __name__ == '__main__':
    args = commandlineArguments(
        activation='relu', iterations=10,
        a=3, maximumFrontier=5, topK=2, pseudoCounts=30.0,  # try 1 0.1 would make prior uniform
        helmholtzRatio=0.5, structurePenalty=1.0,  # try
        CPUs=numberOfCPUs(),
        extras=regex_options)
    main(args)
