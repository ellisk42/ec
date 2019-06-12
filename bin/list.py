
try:
    import binutil  # required to import from lib modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from eclib.domains.list.main import main, list_options
from eclib.ec import commandlineArguments
from eclib.utilities import numberOfCPUs


if __name__ == '__main__':
    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh', iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
        extras=list_options)
    main(args)
