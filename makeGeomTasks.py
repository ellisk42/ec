# coding: utf8

from task import Task
from type import *
from geomPrimitives import tcanvas
import png
import os
import sys
import random

rootdir = "./dataGeom/"



def fileToArray(fname):
    r = png.Reader(filename=fname)
    array = [[(1 - (1 - (y > 0))) for y in x[3::4]] for x in r.read()[2]]
    flatten = [item for sublist in array for item in sublist]
    return flatten


def pretty_print(y):  # Assumption, y is 1 bit depth png, 64*64
    print "_"*132
    for j in range(64):
        sys.stdout.write("|")
        for i in range(64):
            if y[j*64 + (i % 64)] == 1:
                sys.stdout.write("  ")
            else:
                sys.stdout.write("██")
        print "|"
    print "_"*132


def makeTasks():
    problems = []

    def problem(n, examples, needToTrain=False):
        inputType = tuple([])
        outputType = tcanvas
        task = Task(n,
                    outputType,
                    [((), y) for _, y in examples])
        # if needToTrain: task.mustTrain = True         WAT?!
        task.mustTrain = needToTrain
        problems.append(task)

    for _, _, files in os.walk(rootdir):
        for f in files:
            if f.endswith("dash.png"):
                problem("Generate '" + f + "'",
                        [([], fileToArray(rootdir + '/' + f))],
                        needToTrain=True)

    return problems


if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        print t.name
        print t.request
        x, y = t.examples[4]
        pretty_print(y)
        print
