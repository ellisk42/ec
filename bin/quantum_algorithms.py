try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.quantum_algorithms.main import main
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs


if __name__ == '__main__': 
    arguments = commandlineArguments(
        featureExtractor=None, # it was TowerCNN
        CPUs=numberOfCPUs(),
        helmholtzRatio=0.5,
        recognitionTimeout=3600,
        iterations=6,
        a=3,
        structurePenalty=1,
        pseudoCounts=10,
        topK=2,
        maximumFrontier=5,
        extras=None,
        solver="python", 
        useRecognitionModel=False,
        enumerationTimeout=500,#-g
        compressor="pypy")   #ocaml, python, pypy  
    main(arguments)

# -g (no neural network
# --solver python  (or pypy)

# python bin/quantum_algorithms.py -t 5 --compressor=pypy
## TASKS