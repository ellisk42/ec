"""Utility to summarize statistics on logs, given an experiments file with starred experiment names."""
import os
from datetime import datetime
from collections import defaultdict
import re

def getExperimentNames(experimentsFile):
    """Experiment names should be marked with an *"""
    with open(experimentsFile) as f:
        lines = f.readlines()

    experimentNames = []
    for line in lines:
        if "*" in line:
            name = line[line.index("*"):].split()[0][1:] #Split from * to first space, then remove asterisk.
            experimentNames.append(name)

    print("Found experiments:")
    for e in experimentNames:
        print(e)
    return experimentNames

def getMostRecent(ex1, ex2):
    date1, date2 = ex1.split('_')[-1], ex2.split('_')[-1]
    dateFormat = '%Y-%m-%dT%H.%M.%S'
    date1, date2 = datetime.strptime(date1, dateFormat), datetime.strptime(date2, dateFormat)

    return ex1 if date1 > date2 else ex2


def getExperimentLogs(experiments, logsDirectory):
    """Get experiment logs; if most recent, get name."""
    allLogs = os.listdir(logsDirectory)
    experimentLogs = {}
    for e in experiments:
        for log in allLogs:
            if e in log:
                if e not in experimentLogs:
                    experimentLogs[e] = log
                else:   
                    # Tiebreak based on date.
                    experimentLogs[e] = getMostRecent(log, experimentLogs[e])
    
    print("Found experiment logs: ")
    for e in experimentLogs:
        print("%s : %s" % (e, experimentLogs[e]))
    return experimentLogs

def parseLogFile(experimentLog, logsDirectory):
    totalIters, currentIter, bestTest, lastTest = 0, 0, 0, 0
    with open(os.path.join(logsDirectory, experimentLog)) as f:
        lines = f.readlines()

    possiblyBroken = ""
    testingStats = defaultdict(list)
    for i, line in enumerate(lines):
        testing_iteration = re.match('Evaluating on held out testing tasks for iteration: (\d+)', line)
        testing_hits = re.match('Hits (\d+)/(\d+) testing tasks', line)
        if 'iterations  =' in line:
            iterations = re.match('.*iterations  =  (\d+)', line)
            total_iterations = int(iterations.group(1))
        if ('Uncaught exception:' in line) or ('Fatal' in line) or ('fatal' in line):
            possiblyBroken = "POSSIBLY BROKEN"
        if testing_iteration:
            testingStats['test_iteration'].append(int(testing_iteration.group(1)))
        if testing_hits:
            testingStats['test_hits'].append(int(testing_hits.group(1)))
            totalTasks = int(testing_hits.group(2))
            testingStats['test_total_tasks'].append(totalTasks)


    is_done = "DONE" if (testingStats['test_iteration'][-1] == total_iterations - 1) else ""
    last_few_test_hits = testingStats['test_hits'][-1], testingStats['test_hits'][-2] if len(testingStats['test_hits']) > 2 else -1, testingStats['test_hits'][-3] if len(testingStats['test_hits']) > 3 else -1
    print("%s %s Total iterations: %d, on test iteration %d, best test is %d/%d, last few tests are: %d, %d, %d" % \
        (is_done, 
            possiblyBroken,
            total_iterations, 
            testingStats['test_iteration'][-1]+1, 
            max(testingStats['test_hits']), 
            testingStats['test_total_tasks'][-1],
            *last_few_test_hits))


def summarizeLogs(experimentsFile, logsDirectory):
    experiments = getExperimentNames(experimentsFile)
    experimentLogs = getExperimentLogs(experiments, logsDirectory)
    print("Summarizing logs:")

    for e in experimentLogs:
        print("Summarizing %s (%s)" % (e, experimentLogs[e]))
        parseLogFile(experimentLogs[e], logsDirectory)



if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--experimentsFile", type=str)
    parser.add_argument("--logsDirectory", type=str)

    arguments = parser.parse_args()
    
    summarizeLogs(arguments.experimentsFile, 
        arguments.logsDirectory)