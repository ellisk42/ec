import matplotlib.pyplot as plot
from utilities import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("checkpoints",nargs='+',default=[])
    arguments = parser.parse_args()

    for ck in arguments.checkpoints:
        result = loadPickle(ck)
        tasks = list(result.recognitionTaskMetrics.keys())
        for t in tasks:
            print(t)
            print(result.recognitionTaskMetrics[t]["MonteCarloStatistics"])
            print()
    
