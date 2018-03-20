# coding: utf8

from task import Task
from geomPrimitives import tcanvas
import png
import os
import sys

rootdir = "./data/geom/"


def fileToArray(fname):
    r = png.Reader(filename=fname)
    array = [[(1 - (1 - (y > 0))) for y in x[3::4]] for x in r.read()[2]]
    flatten = [item for sublist in array for item in sublist]
    return flatten


def pretty_print(y):
    size = 16
    print ""
    sys.stdout.write("┍")
    sys.stdout.write("━"*(16*2))
    sys.stdout.write("┑")
    print ""
    for j in range(size):
        sys.stdout.write("│")
        for i in range(size):
            if int(y[j*size + (i % size)]) == 0:
                sys.stdout.write("░░")
            else:
                sys.stdout.write("██")
        sys.stdout.write("│")
        print ""
    sys.stdout.write("┕")
    sys.stdout.write("━"*(16*2))
    sys.stdout.write("┙")
    print ""


def makeTasks():
    problems = []

    def problem(n, examples, needToTrain=False):
        outputType = tcanvas
        task = Task(n,
                    outputType,
                    [((), y) for _, y in examples])
        # if needToTrain: task.mustTrain = True         WAT?!
        task.mustTrain = needToTrain
        problems.append(task)

    for _, _, files in os.walk(rootdir):
        for f in files:
            if f.endswith(".png"):
                problem(f,
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
